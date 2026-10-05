import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import pytest
import torch


def load_script():
    path = Path('scripts/kaggle_request_seed_smoke.py')
    spec = importlib.util.spec_from_file_location('request_smoke', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def batch_fixture(module):
    class Info:
        @classmethod
        def from_schedule_batch(cls, batch, vocab_size):
            result = cls()
            result.custom_params = None  # author's default without custom processor
            result.rows = list(batch.reqs)
            return result

        def filter_batch(self, keep, device):
            self.rows = [self.rows[i] for i in keep]

        def merge_batch(self, other):
            self.rows.extend(other.rows)

    ns = {}
    exec(module.BATCH_SOURCE, ns)
    ns['install_batch'](Info)
    return Info


def req(rid, run_id, seed):
    return SimpleNamespace(rid=rid, sampling_params=SimpleNamespace(custom_params={
        'adanoise_smoke': dict(run_id=run_id, seed=seed)}))


def test_request_metadata_survives_no_custom_processor_and_filter_merge():
    m = load_script()
    Info = batch_fixture(m)
    info = Info.from_schedule_batch(SimpleNamespace(reqs=[req('a', 'same_a', 42), req('b', 'same_b', 42)]), 12)
    assert info.custom_params is None
    assert [r['request_id'] for r in info.adanoise_requests] == ['a', 'b']
    info.filter_batch([1], None)
    assert info.adanoise_requests[0]['run_id'] == 'same_b'
    other = Info.from_schedule_batch(SimpleNamespace(reqs=[req('c', 'different', 43)]), 12)
    info.merge_batch(other)
    assert [r['seed'] for r in info.adanoise_requests] == [42, 43]


def test_same_request_advances_rng_new_request_resets_it(monkeypatch, tmp_path):
    m = load_script()
    monkeypatch.setenv('ADANOISE_SMOKE_ROOT', str(tmp_path))
    monkeypatch.setenv('ADANOISE_SMOKE_END_ID', '524')
    for name in ('same_a', 'same_b', 'different'):
        (tmp_path / name / 'events').mkdir(parents=True)
    ns = {}
    exec(m.CAPTURE_SOURCE, ns)
    draws = []

    class Sampler:
        def forward(self, output, info, *args, enable_latent=False, **kwargs):
            draws.append(torch.rand(4))
            output.topk_indices = torch.arange(10).reshape(1, 10)
            output.topk_probs = torch.ones(1, 10) / 10
            return torch.tensor([0])

    ns['install'](Sampler)
    sampler = Sampler()

    def call(rid, run, seed):
        output = SimpleNamespace(next_token_logits=torch.arange(12.).reshape(1, 12), hidden_states=torch.ones(1, 2))
        info = SimpleNamespace(latent_modes=torch.tensor([True]), adanoise_requests=[dict(request_id=rid, run_id=run, seed=seed)])
        return sampler.forward(output, info, enable_latent=True)

    call('a', 'same_a', 42)
    call('a', 'same_a', 42)
    call('b', 'same_b', 42)
    call('c', 'different', 43)
    assert torch.equal(draws[0], draws[2])
    assert not torch.equal(draws[0], draws[1])
    assert not torch.equal(draws[0], draws[3])
    records = [json.loads(line) for p in (tmp_path / 'same_a/events').glob('*.jsonl') for line in p.read_text().splitlines()]
    assert [r['generation_idx'] for r in records] == [0, 1]
    assert [r['request_seed_applied'] for r in records] == [True, False]
    with pytest.raises(ValueError, match='interleav'):
        call('a', 'same_a', 42)


def test_latest_source_is_copied_without_pinned_commit_or_sha_gate(monkeypatch, tmp_path):
    m = load_script()
    package = tmp_path / 'author/pkg/python/sglang'
    for relative in ('__init__.py', 'srt/layers/sampler.py', 'srt/sampling/sampling_batch_info.py'):
        p = package / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('# latest user source\n')
    scoring = package.parents[2] / 'eval/eval_low_tasks_sglang.py'
    scoring.parent.mkdir()
    scoring.write_text('# current scoring source\n')
    monkeypatch.setattr(m.importlib.util, 'find_spec', lambda name: SimpleNamespace(origin=str(package / '__init__.py')))
    root = tmp_path / 'run'
    root.mkdir()
    overlay, _, _ = m.prepare_overlay(root)
    assert 'install_batch' in (overlay / 'sglang/srt/sampling/sampling_batch_info.py').read_text()
    assert (package / 'srt/layers/sampler.py').read_text() == '# latest user source\n'


def test_notebook_embeds_script_and_has_no_checkout_or_source_hash_check():
    m = load_script()
    n = json.loads(Path('notebooks/ecnu-smoke-adaptive-latent-grpo.ipynb').read_text())
    cell = ''.join(n['cells'][9]['source'])
    script = cell.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    assert script == Path('scripts/kaggle_request_seed_smoke.py').read_text().rstrip('\n')
    all_source = '\n'.join(''.join(c['source']) for c in n['cells'])
    assert 'checkout --detach' not in all_source
    assert 'EXPECTED_PACKAGE_SHA256' not in all_source
    assert 'EXPECTED_SOURCE_SHA256' not in all_source
    assert m.DATA_PATH == '/kaggle/input/datasets/fengjixing/datasets-for-latent-grpo/data/GSM8k-Aug-oss-dup-all.parquet'
    for index, digest in m.SETUP_SOURCE_SHA256.items():
        assert m.sha256_bytes(''.join(n['cells'][int(index)]['source']).encode()) == digest


@pytest.mark.parametrize('generation_failure', [False, True])
def test_persistent_runner_starts_one_engine_and_closes_it_on_failure(monkeypatch, tmp_path, generation_failure):
    m = load_script()
    context = dict(question={'prompt': [{'role': 'user', 'content': 'CPU fixture'}]}, model_identity='cpu_fixture')
    (tmp_path / 'input.json').write_text(json.dumps(context))
    calls = []

    class Engine:
        def __init__(self, **kwargs):
            calls.append(('init', kwargs))

        def generate(self, **kwargs):
            calls.append(('generate', kwargs))
            if generation_failure:
                raise RuntimeError('model generation failed')
            return {}

        def shutdown(self):
            calls.append(('shutdown',))

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return 'fixture<think>'

        def encode(self, text, **kwargs):
            return [524] if text == '</think>' else [1, 2]

    class Monitor:
        def __init__(self):
            self.samples = [100.]
            self.error = None
            self.thread = SimpleNamespace(start=lambda: None)

        def stop(self):
            pass

    monkeypatch.setitem(sys.modules, 'sglang', SimpleNamespace(Engine=Engine, __version__='fixture'))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'Tesla T4')
    monkeypatch.setattr(m, 'MemoryMonitor', Monitor)
    monkeypatch.setattr(m, 'collect_result', lambda root, run, seed, *args: dict(worker_pid=123,
                         request_id=run, seed=seed, output_ids=[seed], trace_hash=str(seed), hidden_hash=str(seed)))
    if generation_failure:
        with pytest.raises(RuntimeError, match='model generation failed'):
            m.run_persistent(tmp_path)
    else:
        m.run_persistent(tmp_path)
        requests = [v[1]['sampling_params']['custom_params']['adanoise_smoke'] for v in calls if v[0] == 'generate']
        assert [r['run_id'] for r in requests] == ['same_a', 'same_b', 'different']
        assert requests[0]['seed'] == requests[1]['seed'] != requests[2]['seed']
        assert json.loads((tmp_path / 'smoke_test.json').read_text())['persistent_worker_verified']
    assert len([v for v in calls if v[0] == 'init']) == 1
    assert calls[0][1]['disable_radix_cache'] is True
    assert calls[-1] == ('shutdown',)


def test_summary_rejects_worker_restart_and_allows_same_answer_with_changed_trace():
    m = load_script()
    rs = [dict(worker_pid=123, request_id=name, seed=seed, output_ids=[3],
               trace_hash=str(seed), hidden_hash=str(seed))
          for name, seed in (('a', 42), ('b', 42), ('c', 43))]
    assert m.summarize_results(rs)['status'] == 'passed'
    rs[2]['worker_pid'] = 124
    with pytest.raises(ValueError, match='worker changed'):
        m.summarize_results(rs)


def test_metadata_and_seed_hooks_reach_a_fresh_process(monkeypatch, tmp_path):
    m = load_script()
    package = tmp_path / 'author/pkg/python/sglang'
    (package / 'srt/layers').mkdir(parents=True)
    (package / 'srt/sampling').mkdir()
    (package / '__init__.py').write_text('')
    (package / 'srt/layers/sampler.py').write_text('''import torch
class Sampler:
    def forward(self,o,i,*args,enable_latent=False,**kwargs):
        torch.rand(3)
        o.topk_indices=torch.arange(10).reshape(1,10)
        o.topk_probs=torch.ones(1,10)/10
        return torch.tensor([0])
''')
    (package / 'srt/sampling/sampling_batch_info.py').write_text('''class SamplingBatchInfo:
    @classmethod
    def from_schedule_batch(cls,batch,vocab_size): return cls()
    def filter_batch(self,*args): pass
    def merge_batch(self,*args): pass
''')
    scoring = package.parents[2] / 'eval/eval_low_tasks_sglang.py'
    scoring.parent.mkdir()
    scoring.write_text('# latest scoring fixture')
    monkeypatch.setattr(m.importlib.util, 'find_spec', lambda name: SimpleNamespace(origin=str(package / '__init__.py')))
    root = tmp_path / 'run'
    root.mkdir()
    overlay, _, _ = m.prepare_overlay(root)
    (root / 'same_a/events').mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(overlay), ADANOISE_SMOKE_ROOT=str(root), ADANOISE_SMOKE_END_ID='524')
    code = '''import torch
from types import SimpleNamespace as NS
from sglang.srt.layers.sampler import Sampler
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
r=NS(rid='real-rid',sampling_params=NS(custom_params={'adanoise_smoke':{'run_id':'same_a','seed':77}}))
i=SamplingBatchInfo.from_schedule_batch(NS(reqs=[r]),12)
i.latent_modes=torch.tensor([True])
o=NS(next_token_logits=torch.arange(12.).reshape(1,12),hidden_states=torch.ones(1,2))
Sampler().forward(o,i,enable_latent=True)
'''
    subprocess.run([sys.executable, '-c', code], env=env, check=True)
    path = next((root / 'same_a/events').glob('*.jsonl'))
    record = json.loads(path.read_text())
    assert record['request_id'] == 'real-rid'
    assert record['seed'] == 77
    assert record['request_seed_applied'] is True
