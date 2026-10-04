import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


def load_script():
    path = Path('scripts/kaggle_latent_smoke.py')
    spec = importlib.util.spec_from_file_location('k0_smoke', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def events():
    return [dict(generation_idx=0, next_token_id=4, latent_mode=True,
                 is_mixture_step=True, seed=42, worker_pid=123,
                 hidden_state=[1., 2.], topk_probs=[.1]*10,
                 clean_topk_ids=list(range(10)), topk_ids=[4, 0, 1, 2, 3, 5, 6, 7, 8, 9],
                 mixture_probs=[.1]*10),
            dict(generation_idx=1, next_token_id=524, latent_mode=True,
                 is_mixture_step=False, seed=42, worker_pid=123)]


def test_alignment_uses_only_mixture_steps_and_rejects_token_mismatch():
    m = load_script()
    report = m.validate_events(events(), [4, 524], 42, 524)
    assert report['num_latent_steps'] == 1
    assert report['hidden_size'] == 2
    with pytest.raises(ValueError, match='token'):
        m.validate_events(events(), [5, 524], 42, 524)
    with pytest.raises(ValueError, match='seed'):
        m.validate_events(events(), [4, 524], 43, 524)
    with pytest.raises(ValueError, match='hidden dimension'):
        m.validate_events(events(), [4, 524], 42, 524, expected_hidden_size=3)
    records = events()
    records[0]['cuda_seed'] = 43
    with pytest.raises(ValueError, match='seed'):
        m.validate_events(records, [4, 524], 42, 524)


def test_alignment_rejects_missing_hidden_and_invalid_probabilities():
    m = load_script()
    records = events()
    records[0]['hidden_state'] = []
    with pytest.raises(ValueError, match='hidden'):
        m.validate_events(records, [4, 524], 42, 524)
    records = events()
    records[0]['mixture_probs'][0] = float('nan')
    with pytest.raises(ValueError, match='probab'):
        m.validate_events(records, [4, 524], 42, 524)


def test_hook_records_pre_noise_features_without_changing_rng(monkeypatch, tmp_path):
    m = load_script()
    monkeypatch.setenv('ADANOISE_SMOKE_EVENTS', str(tmp_path))
    monkeypatch.setenv('ADANOISE_SMOKE_END_ID', '524')
    ns = {'__name__': 'capture_fixture'}
    exec(m.CAPTURE_SOURCE, ns)

    class Sampler:
        def forward(self, output, info, *args, enable_latent=False, **kwargs):
            torch.rand(3)  # the real sampler is allowed to advance RNG
            output.next_token_logits.zero_()  # features must precede this mutation
            output.topk_indices = torch.arange(10).reshape(1, 10)
            output.topk_probs = torch.ones(1, 10) / 10
            return torch.tensor([4])

    ns['install'](Sampler)
    info = SimpleNamespace(latent_modes=torch.tensor([True]))
    output = SimpleNamespace(next_token_logits=torch.arange(12.).reshape(1, 12),
                             hidden_states=torch.tensor([[1., 2.]]))
    torch.manual_seed(42)
    before = torch.get_rng_state()
    assert Sampler().forward(output, info, enable_latent=True).tolist() == [4]
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    torch.rand(3)
    assert torch.equal(after, torch.get_rng_state())
    records = [json.loads(line) for path in tmp_path.glob('*.jsonl') for line in path.read_text().splitlines()]
    assert records[0]['clean_topk_ids'] == list(range(11, 1, -1))
    assert records[0]['hidden_state'] == [1., 2.]
    assert records[0]['seed'] == 42


def test_notebook_script_matches_cli_and_preserves_user_setup():
    m = load_script()
    n = json.loads(Path('notebooks/kaggle_latent_smoke.ipynb').read_text())
    script_cell = ''.join(n['cells'][9]['source'])
    embedded = script_cell.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    assert embedded == Path('scripts/kaggle_latent_smoke.py').read_text().rstrip('\n')
    for i in range(1, 9):
        source = ''.join(n['cells'][i]['source'])
        assert m.SETUP_SOURCE_SHA256[str(i)] == m.sha256_bytes(source.encode())
    assert m.MODEL_PATH.endswith('LLaMA3.2-1B-Instruct-Latent-GRPO-Top10')


def test_artifact_generation_config_matches_existing_selector():
    from adaptive_noise.probing import generation_settings
    m = load_script()
    assert m.EFFECTIVE_GENERATION_CONFIG == generation_settings('gsm8k_aug', 1.0)


def test_overlay_hook_reaches_fresh_worker_and_preserves_installed_source(monkeypatch, tmp_path):
    m = load_script()
    package = tmp_path / 'author/pkg/python/sglang'
    sampler_path = package / 'srt/layers/sampler.py'
    sampler_path.parent.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    source = '''import torch
class Sampler:
    def forward(self, output, info, *args, enable_latent=False, **kwargs):
        output.topk_indices = torch.arange(10).reshape(1, 10)
        output.topk_probs = torch.ones(1, 10)/10
        return torch.tensor([0])
'''
    sampler_path.write_text(source)
    scoring = package.parents[2] / 'eval/eval_low_tasks_sglang.py'
    scoring.parent.mkdir()
    scoring.write_text('# fixture')
    monkeypatch.setattr(m.importlib.util, 'find_spec', lambda name: SimpleNamespace(origin=str(package / '__init__.py')))
    monkeypatch.setattr(m, 'EXPECTED_SOURCE_SHA256', {'srt/layers/sampler.py': m.file_hash(sampler_path)})
    monkeypatch.setattr(m, 'EXPECTED_SCORING_SHA256', m.file_hash(scoring))
    monkeypatch.setattr(m, 'EXPECTED_PACKAGE_SHA256', m.package_hash(package))
    root = tmp_path / 'run'
    root.mkdir()
    overlay, _, _ = m.prepare_overlay(root)
    assert sampler_path.read_text() == source
    event_dir = root / 'events'
    event_dir.mkdir()
    env = dict(os.environ, PYTHONPATH=str(overlay), ADANOISE_SMOKE_EVENTS=str(event_dir), ADANOISE_SMOKE_END_ID='524')
    code = '''import torch
from types import SimpleNamespace
from sglang.srt.layers.sampler import Sampler
torch.manual_seed(7)
o=SimpleNamespace(next_token_logits=torch.arange(12.).reshape(1,12), hidden_states=torch.ones(1,2))
i=SimpleNamespace(latent_modes=torch.tensor([True]))
Sampler().forward(o,i,enable_latent=True)
'''
    subprocess.run([sys.executable, '-c', code], env=env, check=True)
    paths = list(event_dir.glob('*.jsonl'))
    assert len(paths) == 1
    assert json.loads(paths[0].read_text())['seed'] == 7


def test_source_mismatch_stops_before_overlay(monkeypatch, tmp_path):
    m = load_script()
    init = tmp_path / 'sglang/__init__.py'
    init.parent.mkdir()
    init.write_text('')
    monkeypatch.setattr(m.importlib.util, 'find_spec', lambda name: SimpleNamespace(origin=str(init)))
    monkeypatch.setattr(m, 'EXPECTED_SOURCE_SHA256', {'srt/layers/sampler.py': 'missing'})
    with pytest.raises(ValueError, match='pinned reference'):
        m.prepare_overlay(tmp_path)
    assert not (tmp_path / 'runtime_overlay').exists()


def test_package_pin_covers_files_outside_the_short_key_file_list(monkeypatch, tmp_path):
    m = load_script()
    package = tmp_path / 'sglang'
    package.mkdir()
    (package / '__init__.py').write_text('')
    monkeypatch.setattr(m.importlib.util, 'find_spec', lambda name: SimpleNamespace(origin=str(package / '__init__.py')))
    monkeypatch.setattr(m, 'EXPECTED_SOURCE_SHA256', {})
    monkeypatch.setattr(m, 'EXPECTED_PACKAGE_SHA256', m.package_hash(package))
    (package / 'changed_embedding.py').write_text('raise RuntimeError("changed source")')
    with pytest.raises(ValueError, match='Python package differs'):
        m.prepare_overlay(tmp_path)
    assert not (tmp_path / 'runtime_overlay').exists()
