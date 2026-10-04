"""Single-GPU K0 diagnostic, embedded verbatim in the user's Bash notebook."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

MODEL_PATH = '/kaggle/input/models/fengjixing/llama3-2-1b-instruct-latent-grpo-top10/other/default/1/LLaMA3.2-1B-Instruct-Latent-GRPO-Top10'
DATA_PATH = '/kaggle/input/datasets/fengjixing/latent-grpo-data/data/GSM8k-Aug-oss-dup-all.parquet'
SOURCE_SHA256 = '3766e3a83cd82ddd686392d8bc6ef6f262821490a09b694b2caed44f1a482501'
CANDIDATE_ROW = 7335  # member of the previously fixed train split; no resampling
PROBLEM_ID = f'gsm8k_aug:{SOURCE_SHA256}:{CANDIDATE_ROW}'
PINNED_COMMIT = '0b7e85f15e9859033653964282348e518f9f6291'
EXPECTED_PACKAGE_SHA256 = 'f6af41f4b2d682a7da3eaf6993e77ddc4f706fc96c0ee8be5184df1a438fa586'
EXPECTED_SCORING_SHA256 = 'fb95e3ca593869c89c81fbba50cd4fc791db642b4d2e333ebbe2b6d702b018d4'
EXPECTED_SOURCE_SHA256 = {'srt/layers/sampler.py': 'cc332d0f121b416a90108f341bffde891e5835ab8b68494afacdabe9fb069695', 'srt/layers/logits_processor.py': '6238a5f1848c0e247d56f5c371ade77a7dc566d5e6b305f9c992201b4b4165a8', 'srt/model_executor/model_runner.py': '71ab5a7e902c6883efec1f2d477c22d90b074429df2c50f69a111e618ffdb392', 'srt/managers/schedule_batch.py': '1d119b65c586129d2871e3aae4fcc9999e3757d8dd428ee70c65426005574516', 'srt/sampling/sampling_params.py': 'ec36d5eea0691142e24570bcddad56f88b96ce0f7559b401618b7f1fa18a77c4', 'srt/entrypoints/engine.py': '7b904f0487b03d3429625935736782e02ff1c90809de680d126d9283719f576b', 'srt/server_args.py': '0229b571dd5107367bc789f780097ab3506fb4999ea26923e42e626d9ed737a9'}
SETUP_SOURCE_SHA256 = {'1': 'd61c544c62cb5c88925ff781b7044e92e36af4cd968ec9016e93a6dc45f90f52', '2': '3ebdc13202223903b48d68ea04669dceb1ca699043be75013981b5e561852d1f', '3': '2e54626cfa084cb4c367d9a56ab24b8f009c70e3e4000329dc21eb03768aff2d', '4': '4d8ce5ec73780379557e5434a011829cf89772054e7f61bf90672de941171b30', '5': '0a77d759020a3d2cfe78e92f53274b4e17aeb77d5f5970a76aa95a3eb69dfdb4', '6': 'f393104aa30d9ec98e995f8f6ceadb412c50f6604b3d55fb5654ace59a9d4363', '7': 'a2329ed7e98bc8b459634eea0794a14b69380727e8c0eae34ab0e76b6cc96eca', '8': 'b7076b5cff4956a1da2bb1d9ebd6c24dd2abbe5e9fe7201ededd4830fc865011'}
GENERATION_CONFIG = dict(temperature=0.6, top_p=0.95, max_new_tokens=128,
                         gumbel_softmax_temperature=1.0, noise_scale=1.0,
                         add_noise_gumbel_softmax=True, use_one_sided_gumbel_noise=False)
EFFECTIVE_GENERATION_CONFIG = dict(GENERATION_CONFIG, max_topk=10)

CAPTURE_SOURCE = r'''
import hashlib
import json
import os
from pathlib import Path
import torch

def install(sampler_class):
    original = sampler_class.forward
    def captured(self, output, info, *args, enable_latent=False, **kwargs):
        directory = os.environ.get('ADANOISE_SMOKE_EVENTS')
        if not directory or not enable_latent:
            return original(self, output, info, *args, enable_latent=enable_latent, **kwargs)
        if output.next_token_logits.shape[0] != 1 or info.latent_modes.numel() != 1:
            raise ValueError('K0 requires exactly one request row')
        mode = bool(info.latent_modes.item())
        record = dict(generation_idx=getattr(self, '_smoke_step', 0),
                      latent_mode=mode, worker_pid=os.getpid(), seed=torch.initial_seed())
        # Read only: no RNG calls, no full-vocabulary copy/history.
        if mode:
            hidden = output.hidden_states
            if hidden is None or hidden.ndim != 2 or hidden.shape[0] != 1:
                raise ValueError('LAST hidden missing or not aligned to single request')
            scores, ids = torch.topk(output.next_token_logits.float(), k=10, dim=-1)
            record.update(hidden_state=hidden[0].detach().float().cpu().tolist(),
                          topk_probs=scores.softmax(-1)[0].detach().cpu().tolist(),
                          clean_topk_ids=ids[0].detach().cpu().tolist())
            del scores, ids, hidden
        tokens = original(self, output, info, *args, enable_latent=enable_latent, **kwargs)
        token = int(tokens[0].item())
        record.update(next_token_id=token,
                      is_mixture_step=mode and token != int(os.environ['ADANOISE_SMOKE_END_ID']))
        if record['is_mixture_step']:
            record.update(topk_ids=output.topk_indices[0].detach().cpu().tolist(),
                          mixture_probs=output.topk_probs[0].detach().float().cpu().tolist())
        if output.next_token_logits.is_cuda:
            device = output.next_token_logits.device
            with torch.cuda.device(device):
                record['cuda_seed'] = torch.cuda.initial_seed()
            record.update(worker_peak_allocated_mb=torch.cuda.max_memory_allocated(device)/2**20,
                          worker_peak_reserved_mb=torch.cuda.max_memory_reserved(device)/2**20)
        path = Path(directory) / f'{os.getpid()}.jsonl'
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        self._smoke_step = record['generation_idx'] + 1
        return tokens
    sampler_class.forward = captured
'''


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def package_hash(package):
    files = {p.relative_to(package).as_posix(): file_hash(p)
             for p in sorted(Path(package).rglob('*.py'))}
    return sha256_bytes(json.dumps(files, sort_keys=True).encode())


def rollout_seed(rollout_id):
    payload = json.dumps([12345, PROBLEM_ID, '1', rollout_id],
                         ensure_ascii=True, separators=(',', ':')).encode('ascii')
    return int.from_bytes(hashlib.sha256(b'adanoise.probe-seeds.v1:' + payload).digest()[:4], 'big')


def check_probabilities(values):
    if (len(values) != 10 or any(not math.isfinite(x) or x < 0 for x in values)
            or not math.isclose(sum(values), 1.0, abs_tol=1e-5)):
        raise ValueError('invalid top10 probabilities')


def validate_events(records, output_ids, seed, end_id, expected_hidden_size=None):
    if not records or [r['generation_idx'] for r in records] != list(range(len(records))):
        raise ValueError('missing or discontinuous sampler events')
    if [r['next_token_id'] for r in records] != output_ids:
        raise ValueError('sampler token sequence differs from returned output_ids')
    if (any(r['seed'] != seed or ('cuda_seed' in r and r['cuda_seed'] != seed) for r in records)
            or len({r['worker_pid'] for r in records}) != 1):
        raise ValueError('worker seed or single-worker identity mismatch')
    if not records[0]['latent_mode']:
        raise ValueError('first generation step did not enter latent mode')
    steps = []
    hidden_size = None
    exited = False
    for record in records:
        if record['latent_mode']:
            if exited:
                raise ValueError('latent mode reentered after explicit generation')
            mixture = record['next_token_id'] != end_id
            if record['is_mixture_step'] != mixture:
                raise ValueError('latent exit and mixture mask mismatch')
        else:
            exited = True
            if record['is_mixture_step']:
                raise ValueError('explicit token mislabeled as mixture step')
        if not record['is_mixture_step']:
            continue
        hidden = record['hidden_state']
        if not hidden or any(not math.isfinite(x) for x in hidden):
            raise ValueError('empty or nonfinite hidden vector')
        hidden_size = hidden_size or len(hidden)
        if len(hidden) != hidden_size or (expected_hidden_size is not None and len(hidden) != expected_hidden_size):
            raise ValueError('hidden dimension changed within trajectory')
        check_probabilities(record['topk_probs'])
        check_probabilities(record['mixture_probs'])
        for name in ('clean_topk_ids', 'topk_ids'):
            ids = record[name]
            if len(ids) != 10 or len(set(ids)) != 10 or any(type(x) is not int or x < 0 for x in ids):
                raise ValueError('invalid sparse top10 token IDs')
        if record['topk_ids'][0] != record['next_token_id']:
            raise ValueError('mixture representative token mismatch')
        steps.append(record)
    if not steps:
        raise ValueError('no actual latent mixture steps')
    return {'num_latent_steps': len(steps), 'hidden_size': hidden_size,
            'latent_exit_observed': any(r['latent_mode'] and r['next_token_id'] == end_id for r in records)}


def load_question():
    import pyarrow.parquet as pq
    path = Path(DATA_PATH)
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_hash(path) != SOURCE_SHA256:
        raise ValueError('mounted GSM8K file differs from fixed candidate-pool source')
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
        if offset <= CANDIDATE_ROW < offset + batch.num_rows:
            row = batch.slice(CANDIDATE_ROW-offset, 1).to_pylist()[0]
            if not row.get('prompt') or not row.get('reward_model', {}).get('ground_truth'):
                raise ValueError('selected fixed train candidate lacks prompt/answer')
            return row
        offset += batch.num_rows
    raise ValueError('fixed train candidate row missing')


def prepare_overlay(root):
    spec = importlib.util.find_spec('sglang')
    if spec is None or not spec.origin:
        raise ModuleNotFoundError('author custom SGLang is required')
    package = Path(spec.origin).resolve().parent
    checks = {}
    for relative, expected in EXPECTED_SOURCE_SHA256.items():
        path = package / relative
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f'installed author source differs from pinned reference: {path}')
        checks[relative] = expected
    if package_hash(package) != EXPECTED_PACKAGE_SHA256:
        raise ValueError('installed author Python package differs from pinned reference')
    scoring_path = package.parents[2] / 'eval/eval_low_tasks_sglang.py'
    if not scoring_path.is_file():
        raise FileNotFoundError(scoring_path)
    if file_hash(scoring_path) != EXPECTED_SCORING_SHA256:
        raise ValueError('installed author grading source differs from pinned reference')
    overlay = root / 'runtime_overlay'
    overlay.mkdir()
    shutil.copytree(package, overlay / 'sglang', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (overlay / 'adanoise_smoke_capture.py').write_text(CAPTURE_SOURCE)
    sampler = overlay / 'sglang/srt/layers/sampler.py'
    with sampler.open('a') as stream:
        stream.write('\nfrom adanoise_smoke_capture import install as _smoke_install\n_smoke_install(Sampler)\n')
    return overlay, scoring_path, checks


def grade_output(text, truth, scoring_path):
    names = ('fix_fracs', 'fix_a_slash_b', 'remove_right_units', 'fix_sqrt', 'strip_string',
             'remove_boxed', 'last_boxed_only_string', 'extract_answer',
             'normalize_answer_text', '_as_float', 'check_is_correct')
    tree = ast.parse(Path(scoring_path).read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if set(names) - functions.keys():
        raise ValueError('required author grading helpers missing')
    ns = {'re': re}
    exec(compile(ast.Module(body=[functions[n] for n in names], type_ignores=[]), str(scoring_path), 'exec'), ns)
    answer = ns['extract_answer'](text) if text.strip() else ''
    invalid = 'empty_output' if not text.strip() else ('answer_unextractable' if not answer.strip() else None)
    return {'reward': int(ns['check_is_correct'](answer, truth)) if invalid is None else 0,
            'extracted_answer': answer, 'invalid_reason': invalid}


class MemoryMonitor:
    """Sample entire GPU0 memory, including worker allocations; not a true peak."""
    def __init__(self):
        self.stop_event = threading.Event()
        self.samples = []
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                text = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                    text=True, timeout=10)
                value = next(float(line.split(',')[1]) for line in text.splitlines()
                             if int(line.split(',')[0]) == 0)
                self.samples.append(value)
            except Exception as error:
                self.error = repr(error)
                return
            self.stop_event.wait(0.5)

    def stop(self):
        self.stop_event.set()
        self.thread.join()


def run_once(root, run_id, seed):
    import asyncio
    import torch
    import sglang as sgl
    from transformers import AutoTokenizer
    context = json.loads((root / 'input.json').read_text())
    run_dir = root / run_id
    run_dir.mkdir()
    event_dir = run_dir / 'events'
    event_dir.mkdir()
    os.environ['ADANOISE_SMOKE_EVENTS'] = str(event_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    text = tokenizer.apply_chat_template(context['question']['prompt'], tokenize=False, add_generation_prompt=True)
    if not text.rstrip().endswith('<think>'):
        text += '<think>'
    end_ids = tokenizer.encode('</think>', add_special_tokens=False)
    if not end_ids:
        raise ValueError('empty latent end marker tokenization')
    end_id = end_ids[0]  # author's marker convention, not the final token of the marker
    os.environ['ADANOISE_SMOKE_END_ID'] = str(end_id)
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    write_json(run_dir / 'prompt.json', dict(text=text, input_ids=input_ids,
                                           end_marker_ids=end_ids, latent_end_token_id=end_id))
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'Tesla T4':
        raise RuntimeError('confirmed T4 GPU0 is not available')
    monitor = MemoryMonitor()
    monitor.thread.start()
    engine = None
    loop = None
    start = time.perf_counter()
    try:
        engine = sgl.Engine(model_path=MODEL_PATH, trust_remote_code=True, dtype='float16',
                            kv_cache_dtype='auto', tp_size=1, base_gpu_id=0, random_seed=seed,
                            enable_latent=True, latent_end_token_id=end_id, disable_cuda_graph=True,
                            disable_overlap_schedule=True, mem_fraction_static=0.90,
                            sampling_backend='flashinfer', max_running_requests=1,
                            log_level='info', skip_tokenizer_init=True, max_topk=10)
        loop = asyncio.new_event_loop()  # preserve user's working uvloop ordering
        asyncio.set_event_loop(loop)
        ready = time.perf_counter()
        output = engine.generate(input_ids=input_ids, sampling_params=GENERATION_CONFIG)
        generation_sec = time.perf_counter() - ready
        monitor.stop()
        if monitor.error or not monitor.samples:
            raise RuntimeError(f'GPU memory measurement failed: {monitor.error}')
        paths = list(event_dir.glob('*.jsonl'))
        if len(paths) != 1:
            raise ValueError('expected exactly one sampler worker event stream')
        records = [json.loads(line) for line in paths[0].read_text().splitlines()]
        if any('cuda_seed' not in r for r in records):
            raise ValueError('CUDA sampling worker seed observation missing')
        ids = output['output_ids']
        checks = validate_events(records, ids, seed, end_id, context['expected_hidden_size'])
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        score = grade_output(decoded, context['question']['reward_model']['ground_truth'], context['scoring_path'])
        finish = output.get('meta_info', {}).get('finish_reason')
        reason = finish.get('type') if isinstance(finish, dict) else finish
        truncated = reason in ('length', 'max_tokens') if reason is not None else None
        identity = dict(schema_version=1, problem_id=PROBLEM_ID, scale=1.0,
                        rollout_id=0 if run_id != 'different' else 1, seed=seed,
                        model_identity=context['model_identity'], generation_config=EFFECTIVE_GENERATION_CONFIG)
        mixed = [r for r in records if r['is_mixture_step']]
        features = dict(identity, feature_schema='raw-latent-v1', feature_source='pre_gumbel',
                        steps=[dict(step_idx=i, hidden_state=r['hidden_state'], topk_probs=r['topk_probs'])
                               for i, r in enumerate(mixed)])
        trajectory = dict(identity, output_text=decoded, explicit_token_ids=[r['next_token_id'] for r in records if not r['latent_mode']],
                          all_output_ids=ids, reward=score['reward'],
                          steps=[dict(step_idx=i, topk_ids=r['topk_ids'], mixture_probs=r['mixture_probs'])
                                 for i, r in enumerate(mixed)])
        write_json(run_dir / 'features.json', features)
        write_json(run_dir / 'trajectory.json', trajectory)
        # Exclude process IDs, timing and memory from repeatability comparisons.
        stochastic_trace = [{k: r[k] for k in ('next_token_id', 'latent_mode', 'is_mixture_step',
                                               'topk_ids', 'mixture_probs') if k in r} for r in records]
        trace_hash = sha256_bytes(json.dumps(stochastic_trace, sort_keys=True).encode())
        result = dict(identity, status='passed', **checks, **score, output_text=decoded,
                      output_ids=ids, output_hash=sha256_bytes(decoded.encode()), trace_hash=trace_hash,
                      finish_reason=finish, truncated=truncated, budget_hit=len(ids) >= 128,
                      hidden_hash=sha256_bytes(json.dumps(features['steps'], sort_keys=True).encode()),
                      startup_sec=ready-start, instrumented_generation_sec=generation_sec,
                      worker_peak_allocated_mb=max(r.get('worker_peak_allocated_mb', 0) for r in records),
                      worker_peak_reserved_mb=max(r.get('worker_peak_reserved_mb', 0) for r in records),
                      sampled_device_memory_max_mb=max(monitor.samples), memory_sample_interval_sec=0.5,
                      gpu_name=torch.cuda.get_device_name(0), torch=torch.__version__, sglang=sgl.__version__,
                      scope='independent_engine_startup_seed_and_single_request_latent_capture')
        write_json(run_dir / 'result.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        monitor.stop()  # stop subprocess measurements before Engine kills its child tree
        if engine is not None:
            engine.shutdown()
        if loop is not None:
            loop.close()
            asyncio.set_event_loop(None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir')
    parser.add_argument('--run', choices=('same_a', 'same_b', 'different'))
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.run:
        run_once(Path(args.output_dir), args.run, args.seed)
        return
    root = Path(args.output_dir) if args.output_dir else Path('/kaggle/working/artifacts') / ('k0_smoke_' + time.strftime('%Y%m%d_%H%M%S'))
    if root.exists():
        raise FileExistsError(f'refusing to overwrite smoke output: {root}')
    if not Path(MODEL_PATH).is_dir():
        raise FileNotFoundError(MODEL_PATH)
    root.mkdir(parents=True)
    write_json(root / 'smoke_test.json', dict(status='incomplete', scope='k0'))
    try:
        question = load_question()
        overlay, scoring, source_hashes = prepare_overlay(root)
        model_dir = Path(MODEL_PATH)
        names = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')
        model_files = [model_dir / name for name in names if (model_dir / name).is_file()]
        weights = sorted(model_dir.glob('*.safetensors'))
        if not weights or not (model_dir / 'config.json').is_file():
            raise FileNotFoundError('GRPO checkpoint requires config.json and safetensors weights')
        hashes = {p.name: file_hash(p) for p in model_files + weights}
        hidden_size = json.loads((model_dir / 'config.json').read_text())['hidden_size']
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError('model config requires a positive hidden_size')
        model_identity = sha256_bytes(json.dumps(hashes, sort_keys=True).encode())
        write_json(root / 'input.json', dict(question=question, problem_id=PROBLEM_ID, split='train',
                                            model_path=MODEL_PATH, model_identity=model_identity,
                                            expected_hidden_size=hidden_size,
                                            model_file_sha256=hashes, data_sha256=SOURCE_SHA256,
                                            pinned_reference_commit=PINNED_COMMIT,
                                            original_runtime_package_sha256=EXPECTED_PACKAGE_SHA256,
                                            overlay_sampler_sha256=file_hash(overlay / 'sglang/srt/layers/sampler.py'),
                                            capture_hook_sha256=sha256_bytes(CAPTURE_SOURCE.encode()),
                                            runtime_source_sha256=source_hashes,
                                            scoring_path=str(scoring), scoring_sha256=file_hash(scoring)))
        env = os.environ.copy()
        env['PYTHONPATH'] = str(overlay) + os.pathsep + env.get('PYTHONPATH', '')
        runs = [('same_a', rollout_seed(0)), ('same_b', rollout_seed(0)), ('different', rollout_seed(1))]
        for run_id, seed in runs:
            print(f'K0 run={run_id}, seed={seed}, output={root}', flush=True)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--output-dir', str(root),
                            '--run', run_id, '--seed', str(seed)], env=env, check=True)
        results = [json.loads((root / name / 'result.json').read_text()) for name, _ in runs]
        a, b, c = results
        same = all(a[key] == b[key] for key in ('output_ids', 'trace_hash', 'hidden_hash'))
        different = a['trace_hash'] != c['trace_hash']
        if not same:
            raise ValueError('same startup seed failed token/mixture/hidden reproducibility')
        # Same text across different seeds is allowed when the latent mixture differs.
        status = 'passed' if different else 'inconclusive'
        write_json(root / 'smoke_test.json', dict(status=status, problem_id=PROBLEM_ID,
                    num_problems=1, num_runs=3, same_seed_passed=same,
                    different_seed_trace_changed=different, model_identity=model_identity,
                    pending=['persistent_engine_per_request_seed', 'formal_probing_throughput_budget'],
                    runs=[dict(run_id=name, **result) for (name, _), result in zip(runs, results)]))
        print(f'K0 {status.upper()}: {root / "smoke_test.json"}', flush=True)
        if not different:
            raise ValueError('different startup seed did not change latent trace; inspection required')
    except Exception as error:
        write_json(root / 'failure.json', dict(status='failed', error_type=type(error).__name__, error=str(error)))
        raise  # STOP: never retry, install, change settings or start formal probing


if __name__ == '__main__':
    main()
