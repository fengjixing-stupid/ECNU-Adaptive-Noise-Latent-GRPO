"""Build a small, immutable candidate pool without loading a model."""
import hashlib
import json
from pathlib import Path
import platform
import random

import pyarrow as pa
import pyarrow.parquet as pq

from adaptive_noise.seeds import derive_dataset_seeds

TRAIN_SOURCES = ('gsm8k_aug', 'dapo_math')
TEST_FILES = {'gsm8k_aug_test': 'test_gsm8k_aug.parquet', 'math500_test': 'test_math500.parquet'}
ROW_SCHEMA = pa.schema([
    ('problem_id', pa.string()), ('source', pa.string()), ('source_file_sha256', pa.string()),
    ('source_row', pa.int64()), ('original_index', pa.int64()), ('original_split', pa.string()),
    ('split', pa.string()), ('question', pa.string()),
    ('prompt', pa.list_(pa.struct([('role', pa.string()), ('content', pa.string())]))),
    ('ground_truth', pa.string()), ('reward_style', pa.string())])
REJECT_SCHEMA = pa.schema([(name, ROW_SCHEMA.field(name).type) for name in
                          ('problem_id', 'source', 'source_file_sha256', 'source_row')] + [('reason', pa.string())])


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def iter_rows(path):
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            yield offset, row
            offset += 1


def _normalize(row, source, digest, index):
    info = row.get('extra_info') or {}
    reward = row.get('reward_model') or {}
    prompt = row.get('prompt')
    question = info.get('question')
    answer = reward.get('ground_truth')
    if not isinstance(question, str) or not question.strip():
        raise ValueError('empty_question')
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('empty_ground_truth')
    if not isinstance(prompt, list) or not prompt or not all(
        isinstance(m, dict) and m.get('role') in {'user', 'system', 'assistant'}
        and isinstance(m.get('content'), str) and m['content'].strip() for m in prompt):
        raise ValueError('invalid_prompt')
    if not any(m['role'] == 'user' for m in prompt):
        raise ValueError('missing_user_message')
    return {'problem_id': f'{source}:{digest}:{index}', 'source': source,
            'source_file_sha256': digest, 'source_row': index,
            'original_index': info.get('index'), 'original_split': info.get('split'),
            'split': '', 'question': question, 'prompt': prompt,
            'ground_truth': answer, 'reward_style': reward.get('style')}


def _write(path, rows, schema):
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def build_dataset(config, root):
    root = Path(root)
    sources = TRAIN_SOURCES + tuple(TEST_FILES)
    paths = {}
    for source in sources:
        path = Path(config['inputs'][source])
        path = path if path.is_absolute() else root / path
        if path.suffix != '.parquet' or not path.is_file():
            raise FileNotFoundError(f'missing required parquet input: {path}')
        paths[source] = path
    seeds = derive_dataset_seeds(config['master_seed'])
    sample, train, val = (config[k] for k in ('sample_per_source', 'train_per_source', 'validation_per_source'))
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (sample, train, val)) or train + val != sample:
        raise ValueError('positive train and validation counts must sum to sample_per_source')
    output = Path(config['output_dir'])
    output = output if output.is_absolute() else root / output
    if output.exists():
        raise FileExistsError(f'refusing to overwrite output: {output}')
    hashes = {source: file_sha256(path) for source, path in paths.items()}
    rejected, train_rows, val_rows = [], [], []
    input_stats = {}
    split_rng = random.Random(seeds['train_validation_split'])
    for source, role in zip(TRAIN_SOURCES, ('gsm8k_aug_sample', 'dapo_sample')):
        valid = []
        count = 0
        for index, row in iter_rows(paths[source]):
            count += 1
            try:
                _normalize(row, source, hashes[source], index)
                valid.append(index)
            except ValueError as error:
                rejected.append({'problem_id': f'{source}:{hashes[source]}:{index}', 'source': source,
                                 'source_file_sha256': hashes[source], 'source_row': index, 'reason': str(error)})
        if len(valid) < sample:
            raise ValueError(f'insufficient valid candidates for {source}: {len(valid)} < {sample}')
        selected = sorted(random.Random(seeds[role]).sample(valid, sample))
        split_rng.shuffle(selected)
        train_ids = set(selected[:train])
        selected_ids = set(selected)
        for index, row in iter_rows(paths[source]):
            if index not in selected_ids:
                continue
            record = _normalize(row, source, hashes[source], index)
            record['split'] = 'train' if index in train_ids else 'validation'
            (train_rows if index in train_ids else val_rows).append(record)
        input_stats[source] = {'raw': count, 'valid': len(valid), 'rejected': count - len(valid)}
    tests = {}
    for source in TEST_FILES:
        records = []
        for index, row in iter_rows(paths[source]):
            try:
                record = _normalize(row, source, hashes[source], index)
            except ValueError as error:
                raise ValueError(f'invalid test record {source}:{index}: {error}') from error
            record['split'] = 'test'
            records.append(record)
        tests[source] = records
        input_stats[source] = {'raw': len(records), 'valid': len(records), 'rejected': 0}
    # Recheck before publishing: all selected IDs refer to these input bytes.
    if any(file_sha256(path) != hashes[source] for source, path in paths.items()):
        raise ValueError('input changed during construction')
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'train.parquet', train_rows, ROW_SCHEMA)
    _write(output / 'validation.parquet', val_rows, ROW_SCHEMA)
    for source, name in TEST_FILES.items():
        _write(output / name, tests[source], ROW_SCHEMA)
    all_rows = train_rows + val_rows + [r for rows in tests.values() for r in rows]
    manifest_schema = pa.schema([(name, ROW_SCHEMA.field(name).type) for name in ('problem_id', 'source', 'split')])
    _write(output / 'split_manifest.parquet', all_rows, manifest_schema)
    _write(output / 'rejected.parquet', rejected, REJECT_SCHEMA)
    counts = {'train': len(train_rows), 'validation': len(val_rows),
              **{source: len(rows) for source, rows in tests.items()}, 'rejected': len(rejected)}
    manifest = {'schema_version': 1, 'status': 'complete', 'master_seed': config['master_seed'],
                'derived_seeds': seeds, 'seed_algorithm': 'sha256-v1',
                'rng_algorithm': 'python.random.Random', 'python': platform.python_version(),
                'pyarrow': pa.__version__, 'config': config,
                'inputs': {source: {'path': str(path), 'sha256': hashes[source], **input_stats[source]}
                           for source, path in paths.items()}, 'counts': counts,
                'deduplication': 'disabled_by_user',
                'output_sha256': {p.name: file_sha256(p) for p in sorted(output.glob('*.parquet'))}}
    temporary = output / 'build_manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.rename(output / 'build_manifest.json')
    return manifest
