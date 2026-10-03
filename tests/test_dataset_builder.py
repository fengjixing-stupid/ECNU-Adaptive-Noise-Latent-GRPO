from pathlib import Path
import json
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from adaptive_noise.dataset_builder import build_dataset
from adaptive_noise.seeds import derive_dataset_seeds, derive_rollout_seed


def make_input(path, count, split='train'):
    rows = [{'prompt': [{'role': 'user', 'content': 'same question'}],
             'reward_model': {'ground_truth': '42', 'style': 'rule'},
             'extra_info': {'question': 'same question', 'index': i, 'split': split}}
            for i in range(count)]
    pq.write_table(pa.Table.from_pylist(rows), path)
    return rows


def fixture_config(tmp_path):
    names = ['gsm8k_aug', 'dapo_math', 'gsm8k_aug_test', 'math500_test']
    paths = {}
    for name in names:
        path = tmp_path / (name + '.parquet')
        make_input(path, 180 if name in names[:2] else 3)
        paths[name] = str(path)
    return {'master_seed': 42, 'sample_per_source': 160, 'train_per_source': 120,
            'validation_per_source': 40, 'inputs': paths,
            'output_dir': str(tmp_path / 'output')}


def test_approved_seeds_and_scale_normalization():
    assert derive_dataset_seeds(42) == {'gsm8k_aug_sample': 2028899853,
        'dapo_sample': 3785434924, 'train_validation_split': 262147858}
    assert derive_rollout_seed('p', 1, 0) == derive_rollout_seed('p', 1.0, 0)
    assert len({derive_rollout_seed('p', .5, i) for i in range(7)}) == 7
    for bad in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            derive_rollout_seed('p', bad, 0)


def test_build_reproducible_counts_roles_and_duplicate_preservation(tmp_path):
    config = fixture_config(tmp_path)
    first = build_dataset(config, tmp_path)
    out = Path(config['output_dir'])
    train = pq.read_table(out / 'train.parquet').to_pylist()
    val = pq.read_table(out / 'validation.parquet').to_pylist()
    assert len(train) == 240 and len(val) == 80
    assert {s: sum(r['source'] == s for r in train) for s in ['gsm8k_aug', 'dapo_math']} == {'gsm8k_aug': 120, 'dapo_math': 120}
    assert len({r['problem_id'] for r in train + val}) == 320
    assert len({r['question'] for r in train + val}) == 1  # no text dedup
    assert all(r['prompt'] == [{'role': 'user', 'content': 'same question'}] and r['ground_truth'] == '42' for r in train)
    test = pq.read_table(out / 'test_math500.parquet').to_pylist()
    assert all(r['split'] == 'test' and r['original_split'] == 'train' for r in test)
    assert first['counts']['train'] == 240
    assert json.loads((out / 'build_manifest.json').read_text())['status'] == 'complete'
    config['output_dir'] = str(tmp_path / 'second')
    build_dataset(config, tmp_path)
    assert pq.read_table(Path(config['output_dir']) / 'train.parquet').to_pylist() == train
    assert pq.read_table(Path(config['output_dir']) / 'validation.parquet').to_pylist() == val


def test_invalid_train_record_rejected_and_test_not_silently_removed(tmp_path):
    config = fixture_config(tmp_path)
    p = Path(config['inputs']['gsm8k_aug'])
    rows = pq.read_table(p).to_pylist()
    rows[0]['reward_model']['ground_truth'] = ''
    pq.write_table(pa.Table.from_pylist(rows), p)
    build_dataset(config, tmp_path)
    rejected = pq.read_table(Path(config['output_dir']) / 'rejected.parquet').to_pylist()
    assert len(rejected) == 1 and rejected[0]['source_row'] == 0
    config['output_dir'] = str(tmp_path / 'bad-test-output')
    p = Path(config['inputs']['math500_test'])
    rows = pq.read_table(p).to_pylist()
    rows[0]['extra_info']['question'] = ''
    pq.write_table(pa.Table.from_pylist(rows), p)
    with pytest.raises(ValueError, match='test'):
        build_dataset(config, tmp_path)
    assert not (Path(config['output_dir']) / 'build_manifest.json').exists()


def test_missing_input_preflight_and_existing_output(tmp_path):
    config = fixture_config(tmp_path)
    Path(config['inputs']['math500_test']).unlink()
    with pytest.raises(FileNotFoundError):
        build_dataset(config, tmp_path)
    assert not Path(config['output_dir']).exists()
    make_input(Path(config['inputs']['math500_test']), 3)
    build_dataset(config, tmp_path)
    with pytest.raises(FileExistsError):
        build_dataset(config, tmp_path)


def test_insufficient_candidates_do_not_auto_reduce_budget(tmp_path):
    config = fixture_config(tmp_path)
    make_input(Path(config['inputs']['dapo_math']), 159)
    with pytest.raises(ValueError, match='insufficient'):
        build_dataset(config, tmp_path)


def test_cli_missing_config_stops(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/build_dataset.py'
    result = subprocess.run([sys.executable, str(script), '--config', str(tmp_path / 'missing.yaml')], capture_output=True, text=True)
    assert result.returncode == 2 and 'STOP:' in result.stderr
