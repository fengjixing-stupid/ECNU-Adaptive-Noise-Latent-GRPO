"""Single-GPU, persistent Engine, sequential per-request seed diagnostic."""
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
DATA_PATH = '/kaggle/input/datasets/fengjixing/datasets-for-latent-grpo/data/GSM8k-Aug-oss-dup-all.parquet'
SOURCE_SHA256 = '3766e3a83cd82ddd686392d8bc6ef6f262821490a09b694b2caed44f1a482501'
CANDIDATE_ROW = 7335
PROBLEM_ID = 'gsm8k_aug:3766e3a83cd82ddd686392d8bc6ef6f262821490a09b694b2caed44f1a482501:7335'
GENERATION_CONFIG = {'temperature': 0.6, 'top_p': 0.95, 'max_new_tokens': 128, 'gumbel_softmax_temperature': 1.0, 'noise_scale': 1.0, 'add_noise_gumbel_softmax': True, 'use_one_sided_gumbel_noise': False}
EFFECTIVE_GENERATION_CONFIG = {'temperature': 0.6, 'top_p': 0.95, 'max_new_tokens': 128, 'gumbel_softmax_temperature': 1.0, 'noise_scale': 1.0, 'add_noise_gumbel_softmax': True, 'use_one_sided_gumbel_noise': False, 'max_topk': 10}
ENGINE_STARTUP_SEED = 12345
SETUP_SOURCE_SHA256 = {'1': 'd61c544c62cb5c88925ff781b7044e92e36af4cd968ec9016e93a6dc45f90f52', '2': '3ebdc13202223903b48d68ea04669dceb1ca699043be75013981b5e561852d1f', '3': '2e54626cfa084cb4c367d9a56ab24b8f009c70e3e4000329dc21eb03768aff2d', '4': '4d8ce5ec73780379557e5434a011829cf89772054e7f61bf90672de941171b30', '5': '0a77d759020a3d2cfe78e92f53274b4e17aeb77d5f5970a76aa95a3eb69dfdb4', '6': 'f393104aa30d9ec98e995f8f6ceadb412c50f6604b3d55fb5654ace59a9d4363', '7': 'a2329ed7e98bc8b459634eea0794a14b69380727e8c0eae34ab0e76b6cc96eca', '8': 'b7076b5cff4956a1da2bb1d9ebd6c24dd2abbe5e9fe7201ededd4830fc865011'}

BATCH_SOURCE = "def install_batch(batch_class):\n    original_from = batch_class.from_schedule_batch.__func__\n    original_filter = batch_class.filter_batch\n    original_merge = batch_class.merge_batch\n\n    @classmethod\n    def from_batch(cls, batch, vocab_size):\n        result = original_from(cls, batch, vocab_size)\n        metadata = []\n        for req in batch.reqs:\n            custom = req.sampling_params.custom_params or {}\n            smoke = custom.get('adanoise_smoke')\n            metadata.append(dict(smoke, request_id=req.rid) if smoke is not None else None)\n        result.adanoise_requests = metadata\n        return result\n\n    def filtered(self, keep_indices, keep_indices_device):\n        metadata = getattr(self, 'adanoise_requests', None)\n        result = original_filter(self, keep_indices, keep_indices_device)\n        if metadata is not None:\n            self.adanoise_requests = [metadata[i] for i in keep_indices]\n        return result\n\n    def merged(self, other):\n        lhs = getattr(self, 'adanoise_requests', None)\n        rhs = getattr(other, 'adanoise_requests', None)\n        if lhs is None or rhs is None:\n            raise ValueError('request metadata missing during batch merge')\n        combined = list(lhs) + list(rhs)\n        result = original_merge(self, other)\n        self.adanoise_requests = combined\n        return result\n\n    batch_class.from_schedule_batch = from_batch\n    batch_class.filter_batch = filtered\n    batch_class.merge_batch = merged\n"

CAPTURE_SOURCE = "\nimport hashlib\nimport json\nimport os\nfrom pathlib import Path\nimport torch\n\ndef install(sampler_class):\n    original = sampler_class.forward\n    def captured(self, output, info, *args, enable_latent=False, **kwargs):\n        root = os.environ.get('ADANOISE_SMOKE_ROOT')\n        metadata = getattr(info, 'adanoise_requests', None)\n        if not root or not enable_latent or not metadata or all(m is None for m in metadata):\n            return original(self, output, info, *args, enable_latent=enable_latent, **kwargs)\n        if len(metadata) != 1 or metadata[0] is None:\n            raise ValueError('request seed smoke requires one active request')\n        request = metadata[0]\n        rid, run_id, seed = request['request_id'], request['run_id'], request['seed']\n        if run_id not in ('same_a', 'same_b', 'different') or not isinstance(rid, str) or not rid:\n            raise ValueError('invalid smoke request identity')\n        if type(seed) is not int or not 0 <= seed < 2**32:\n            raise ValueError('invalid request seed')\n        new_request = getattr(self, '_smoke_rid', None) != rid\n        if new_request:\n            seen = getattr(self, '_smoke_seen', set())\n            if rid in seen:\n                raise ValueError('interleaved requests are unsupported by this single-request smoke')\n            seen.add(rid)\n            self._smoke_seen = seen\n            self._smoke_rid = rid\n            self._smoke_metadata = (run_id, seed)\n            self._smoke_step = 0\n            # Initialize the actual worker RNG once, at the first sample of this request.\n            torch.random.default_generator.manual_seed(seed)\n            if output.next_token_logits.is_cuda:\n                with torch.cuda.device(output.next_token_logits.device):\n                    torch.cuda.manual_seed(seed)\n        elif self._smoke_metadata != (run_id, seed):\n            raise ValueError('metadata changed within an active request')\n        directory = str(Path(root) / run_id / 'events')\n        if output.next_token_logits.shape[0] != 1 or info.latent_modes.numel() != 1:\n            raise ValueError('K0 requires exactly one request row')\n        mode = bool(info.latent_modes.item())\n        record = dict(generation_idx=getattr(self, '_smoke_step', 0),\n                      latent_mode=mode, worker_pid=os.getpid(), seed=torch.initial_seed(),\n                      request_id=rid, run_id=run_id, request_seed_applied=new_request)\n        # Read only: no RNG calls, no full-vocabulary copy/history.\n        if mode:\n            hidden = output.hidden_states\n            if hidden is None or hidden.ndim != 2 or hidden.shape[0] != 1:\n                raise ValueError('LAST hidden missing or not aligned to single request')\n            scores, ids = torch.topk(output.next_token_logits.float(), k=10, dim=-1)\n            record.update(hidden_state=hidden[0].detach().float().cpu().tolist(),\n                          topk_probs=scores.softmax(-1)[0].detach().cpu().tolist(),\n                          clean_topk_ids=ids[0].detach().cpu().tolist())\n            del scores, ids, hidden\n        tokens = original(self, output, info, *args, enable_latent=enable_latent, **kwargs)\n        token = int(tokens[0].item())\n        record.update(next_token_id=token,\n                      is_mixture_step=mode and token != int(os.environ['ADANOISE_SMOKE_END_ID']))\n        if record['is_mixture_step']:\n            record.update(topk_ids=output.topk_indices[0].detach().cpu().tolist(),\n                          mixture_probs=output.topk_probs[0].detach().float().cpu().tolist())\n        if output.next_token_logits.is_cuda:\n            device = output.next_token_logits.device\n            with torch.cuda.device(device):\n                record['cuda_seed'] = torch.cuda.initial_seed()\n            record.update(worker_peak_allocated_mb=torch.cuda.max_memory_allocated(device)/2**20,\n                          worker_peak_reserved_mb=torch.cuda.max_memory_reserved(device)/2**20)\n        path = Path(directory) / f'{os.getpid()}.jsonl'\n        with path.open('a', encoding='utf-8') as stream:\n            stream.write(json.dumps(record, allow_nan=False) + '\\n')\n        self._smoke_step = record['generation_idx'] + 1\n        return tokens\n    sampler_class.forward = captured\n"

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


def prepare_overlay(root):
    spec = importlib.util.find_spec('sglang')
    if spec is None or not spec.origin:
        raise ModuleNotFoundError('author custom SGLang is required')
    package = Path(spec.origin).resolve().parent
    relative_paths = ('srt/layers/sampler.py', 'srt/sampling/sampling_batch_info.py')
    for relative in relative_paths:
        if not (package / relative).is_file():
            raise FileNotFoundError(package / relative)
    scoring = package.parents[2] / 'eval/eval_low_tasks_sglang.py'
    if not scoring.is_file():
        raise FileNotFoundError(scoring)
    # Record provenance only. No comparison against a pinned commit or source SHA.
    source = {'installed_package_path': str(package), 'package_sha256': package_hash(package),
              'scoring_sha256': file_hash(scoring)}
    overlay = root / 'runtime_overlay'
    overlay.mkdir()
    shutil.copytree(package, overlay / 'sglang', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (overlay / 'adanoise_smoke_capture.py').write_text(CAPTURE_SOURCE)
    (overlay / 'adanoise_request_metadata.py').write_text(BATCH_SOURCE)
    with (overlay / 'sglang/srt/layers/sampler.py').open('a') as stream:
        stream.write('\nfrom adanoise_smoke_capture import install as _smoke_install\n_smoke_install(Sampler)\n')
    with (overlay / 'sglang/srt/sampling/sampling_batch_info.py').open('a') as stream:
        stream.write('\nfrom adanoise_request_metadata import install_batch as _install_metadata\n_install_metadata(SamplingBatchInfo)\n')
    return overlay, scoring, source


def summarize_results(results):
    if len(results) != 3:
        raise ValueError('expected three sequential results')
    a, b, c = results
    if len({r['worker_pid'] for r in results}) != 1:
        raise ValueError('sampling worker changed; persistent Engine not verified')
    if len({r['request_id'] for r in results}) != 3:
        raise ValueError('expected three distinct scheduler request IDs')
    if a['seed'] != b['seed'] or a['seed'] == c['seed']:
        raise ValueError('request seed identities do not match the repeat/difference design')
    same = all(a[key] == b[key] for key in ('output_ids', 'trace_hash', 'hidden_hash'))
    if not same:
        raise ValueError('same request seed failed token/mixture/hidden reproducibility')
    different = a['trace_hash'] != c['trace_hash']
    return dict(status='passed' if different else 'inconclusive', same_seed_passed=same,
                different_seed_trace_changed=different, persistent_worker_verified=True,
                engine_start_count=1, num_problems=1, num_requests=3,
                scope='single_gpu_single_active_request_persistent_engine_seed')


def collect_result(root, run_id, seed, output, tokenizer, context, end_id, generation_sec, sampled_memory):
    run_dir = root / run_id
    paths = list((run_dir / 'events').glob('*.jsonl'))
    if len(paths) != 1:
        raise ValueError('expected one sampling worker event stream')
    records = [json.loads(line) for line in paths[0].read_text().splitlines()]
    if not records or any('cuda_seed' not in r for r in records):
        raise ValueError('CUDA sampling worker seed observation missing')
    if (not records[0]['request_seed_applied']
            or any(r['request_seed_applied'] for r in records[1:])
            or len({r['request_id'] for r in records}) != 1
            or any(r['run_id'] != run_id for r in records)):
        raise ValueError('request identity or first-step-only seed initialization mismatch')
    request_id = records[0]['request_id']
    returned_id = output.get('meta_info', {}).get('id')
    if returned_id is not None and returned_id != request_id:
        raise ValueError('returned scheduler request ID differs from sampler request')
    ids = output['output_ids']
    checks = validate_events(records, ids, seed, end_id, context['expected_hidden_size'])
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    score = grade_output(decoded, context['question']['reward_model']['ground_truth'], context['scoring_path'])
    finish = output.get('meta_info', {}).get('finish_reason')
    reason = finish.get('type') if isinstance(finish, dict) else finish
    identity = dict(schema_version=1, problem_id=PROBLEM_ID, scale=1.0,
                    rollout_id=0 if run_id != 'different' else 1, seed=seed,
                    model_identity=context['model_identity'], generation_config=EFFECTIVE_GENERATION_CONFIG)
    mixed = [r for r in records if r['is_mixture_step']]
    features = dict(identity, feature_schema='raw-latent-v1', feature_source='pre_gumbel',
                    steps=[dict(step_idx=i, hidden_state=r['hidden_state'], topk_probs=r['topk_probs'])
                           for i, r in enumerate(mixed)])
    trajectory = dict(identity, output_text=decoded, all_output_ids=ids, reward=score['reward'],
                      explicit_token_ids=[r['next_token_id'] for r in records if not r['latent_mode']],
                      steps=[dict(step_idx=i, topk_ids=r['topk_ids'], mixture_probs=r['mixture_probs'])
                             for i, r in enumerate(mixed)])
    write_json(run_dir / 'features.json', features)
    write_json(run_dir / 'trajectory.json', trajectory)
    stochastic = [{k: r[k] for k in ('next_token_id', 'latent_mode', 'is_mixture_step',
                                   'topk_ids', 'mixture_probs') if k in r} for r in records]
    result = dict(identity, status='passed', **checks, **score, request_id=request_id,
                  worker_pid=records[0]['worker_pid'], output_text=decoded, output_ids=ids,
                  output_hash=sha256_bytes(decoded.encode()),
                  trace_hash=sha256_bytes(json.dumps(stochastic, sort_keys=True).encode()),
                  hidden_hash=sha256_bytes(json.dumps(features['steps'], sort_keys=True).encode()),
                  finish_reason=finish, truncated=reason in ('length', 'max_tokens') if reason else None,
                  budget_hit=len(ids) >= GENERATION_CONFIG['max_new_tokens'],
                  instrumented_generation_sec=generation_sec, sampled_device_memory_max_mb=sampled_memory,
                  worker_peak_allocated_mb=max(r.get('worker_peak_allocated_mb', 0) for r in records),
                  worker_peak_reserved_mb=max(r.get('worker_peak_reserved_mb', 0) for r in records),
                  memory_scope='peaks_since_engine_start_and_device_samples',
                  request_seed_initialized_once=True, prefix_cache_enabled=False)
    write_json(run_dir / 'result.json', result)
    return result


def run_persistent(root):
    import asyncio
    import torch
    import sglang as sgl
    from transformers import AutoTokenizer
    context = json.loads((root / 'input.json').read_text())
    os.environ['ADANOISE_SMOKE_ROOT'] = str(root)
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'Tesla T4':
        raise RuntimeError('confirmed T4 GPU0 is not available')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    text = tokenizer.apply_chat_template(context['question']['prompt'], tokenize=False, add_generation_prompt=True)
    if not text.rstrip().endswith('<think>'):
        text += '<think>'
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    end_ids = tokenizer.encode('</think>', add_special_tokens=False)
    if not end_ids:
        raise ValueError('empty latent end marker tokenization')
    end_id = end_ids[0]
    os.environ['ADANOISE_SMOKE_END_ID'] = str(end_id)
    write_json(root / 'prompt.json', dict(text=text, input_ids=input_ids, end_marker_ids=end_ids,
                                        latent_end_token_id=end_id))
    runs = [('same_a', rollout_seed(0)), ('same_b', rollout_seed(0)), ('different', rollout_seed(1))]
    for run_id, _ in runs:
        (root / run_id / 'events').mkdir(parents=True)
    monitor = MemoryMonitor()
    monitor.thread.start()
    engine = None
    loop = None
    start = time.perf_counter()
    try:
        engine = sgl.Engine(model_path=MODEL_PATH, trust_remote_code=True, dtype='float16',
                            kv_cache_dtype='auto', tp_size=1, base_gpu_id=0,
                            random_seed=ENGINE_STARTUP_SEED, enable_latent=True,
                            latent_end_token_id=end_id, disable_cuda_graph=True,
                            disable_overlap_schedule=True, disable_radix_cache=True,
                            mem_fraction_static=0.90, sampling_backend='flashinfer',
                            max_running_requests=1, log_level='info', skip_tokenizer_init=True, max_topk=10)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        startup_sec = time.perf_counter() - start
        results = []
        for run_id, seed in runs:
            print(f'Persistent request={run_id}, seed={seed}, engine_start_count=1', flush=True)
            sample_start = len(monitor.samples)
            begin = time.perf_counter()
            params = dict(GENERATION_CONFIG, custom_params={
                'adanoise_smoke': dict(run_id=run_id, seed=seed)})
            output = engine.generate(input_ids=input_ids, sampling_params=params)
            generation_sec = time.perf_counter() - begin
            if monitor.error or not monitor.samples:
                raise RuntimeError(f'GPU memory measurement failed: {monitor.error}')
            values = monitor.samples[sample_start:]
            # A request can finish between two device samples; do not invent a per-request peak.
            sampled_memory = max(values) if values else None
            result = collect_result(root, run_id, seed, output, tokenizer, context, end_id,
                                    generation_sec, sampled_memory)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        checks = summarize_results(results)
        monitor.stop()
        if monitor.error:
            raise RuntimeError(f'GPU memory measurement failed: {monitor.error}')
        summary = dict(checks, problem_id=PROBLEM_ID, model_identity=context['model_identity'],
                       startup_sec=startup_sec, engine_startup_seed=ENGINE_STARTUP_SEED,
                       prefix_cache_enabled=False, max_running_requests=1,
                       sampled_device_memory_max_mb=max(monitor.samples), memory_sample_interval_sec=0.5,
                       gpu_name=torch.cuda.get_device_name(0), torch=torch.__version__, sglang=sgl.__version__,
                       runs=[dict(run_id=name, **r) for (name, _), r in zip(runs, results)],
                       pending=['multi_request_concurrent_rng_isolation', 'formal_probing_throughput_budget'])
        write_json(root / 'smoke_test.json', summary)
        print(f'REQUEST SEED {checks["status"].upper()}: {root / "smoke_test.json"}', flush=True)
        if checks['status'] != 'passed':
            raise ValueError('different request seed did not change trace; inspection required')
    finally:
        monitor.stop()
        if engine is not None:
            engine.shutdown()
        if loop is not None:
            loop.close()
            asyncio.set_event_loop(None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        run_persistent(Path(args.output_dir))
        return
    root = Path(args.output_dir) if args.output_dir else Path('/kaggle/working/artifacts') / ('request_seed_smoke_' + time.strftime('%Y%m%d_%H%M%S'))
    if root.exists():
        raise FileExistsError(f'refusing to overwrite output: {root}')
    if not Path(MODEL_PATH).is_dir():
        raise FileNotFoundError(MODEL_PATH)
    root.mkdir(parents=True)
    write_json(root / 'smoke_test.json', dict(status='incomplete', scope='persistent_request_seed'))
    try:
        question = load_question()
        overlay, scoring, provenance = prepare_overlay(root)
        model = Path(MODEL_PATH)
        weights = sorted(model.glob('*.safetensors'))
        if not weights or not (model / 'config.json').is_file():
            raise FileNotFoundError('GRPO checkpoint config/weights missing')
        files = [model / name for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json',
                                          'special_tokens_map.json') if (model / name).is_file()] + weights
        model_hashes = {p.name: file_hash(p) for p in files}
        hidden_size = json.loads((model / 'config.json').read_text())['hidden_size']
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError('model config requires a positive hidden_size')
        write_json(root / 'input.json', dict(question=question, problem_id=PROBLEM_ID, split='train',
                   model_path=MODEL_PATH, model_identity=sha256_bytes(json.dumps(model_hashes, sort_keys=True).encode()),
                   model_file_sha256=model_hashes, data_sha256=SOURCE_SHA256,
                   expected_hidden_size=hidden_size, scoring_path=str(scoring), runtime_provenance=provenance,
                   capture_hook_sha256=sha256_bytes(CAPTURE_SOURCE.encode()),
                   metadata_hook_sha256=sha256_bytes(BATCH_SOURCE.encode())))
        env = os.environ.copy()
        env['PYTHONPATH'] = str(overlay) + os.pathsep + env.get('PYTHONPATH', '')
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', '--output-dir', str(root)],
                       env=env, check=True)
    except Exception as error:
        write_json(root / 'failure.json', dict(status='failed', error_type=type(error).__name__, error=str(error)))
        raise  # STOP, no environment changes, retries or automatic formal probing


if __name__ == '__main__':
    main()
