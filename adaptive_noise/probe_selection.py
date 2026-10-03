"""Validate CPU probing artifacts and retain the best-scale trajectories."""
import hashlib
import json
import math
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

from adaptive_noise.dataset_builder import file_sha256, ROW_SCHEMA
from adaptive_noise.probing import decide_probe, plan_requests, generation_settings
from adaptive_noise.seeds import derive_rollout_seed


def _candidates(directory):
    directory = Path(directory)
    manifest_path = directory / 'build_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise ValueError('candidate build is not complete')
    for name, digest in manifest['output_sha256'].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('invalid candidate artifact path')
        if not path.is_file():
            raise FileNotFoundError(f'missing required candidate artifact: {path}')
        if file_sha256(path) != digest:
            raise ValueError(f'candidate artifact changed: {path}')
    records = []
    for split in ('train', 'validation'):
        for row in pq.read_table(directory / f'{split}.parquet').to_pylist():
            if row['split'] != split or row['source'] not in {'gsm8k_aug','dapo_math'}:
                raise ValueError('candidate source/split mismatch')
            records.append(row)
    if len({row['problem_id'] for row in records}) != len(records):
        raise ValueError('duplicate candidate IDs')
    mapping = {row['problem_id']: row for row in records}
    return manifest, mapping


def _group(mapping, rows):
    groups = {pid: [] for pid in mapping}
    for row in rows:
        pid = row['problem_id']
        if pid not in mapping:
            raise ValueError(f'unknown candidate problem_id: {pid}')
        candidate = mapping[pid]
        if row['source'] != candidate['source'] or row['split'] != candidate['split']:
            raise ValueError('rollout source/split differs from fixed candidate split')
        if row['seed'] != derive_rollout_seed(pid, row['scale'], row['rollout_id']):
            raise ValueError('rollout seed differs from approved identity')
        groups[pid].append(row)
    return groups


def plan_candidate_requests(candidate_dir, rows):
    _, mapping = _candidates(candidate_dir)
    groups = _group(mapping, rows)
    requests = []
    for pid, candidate in mapping.items():
        requests.extend(plan_requests(pid, candidate['split'], candidate['source'], groups[pid]))
    return requests


def _artifact(root, name):
    if not isinstance(name, str) or not name:
        raise FileNotFoundError('missing best-scale trajectory/features reference')
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('trajectory/features path must remain inside trajectory_root')
    if path.suffix != '.json' or not path.is_file():
        raise FileNotFoundError(f'missing required trajectory/features JSON: {path}')
    return path


def _vector(values, *, probabilities=False, size=None):
    if not isinstance(values, list) or not values or (size is not None and len(values) != size):
        raise ValueError('missing or incorrect feature/mixture vector size')
    if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in values):
        raise ValueError('nonfinite or nonnumeric trajectory/feature values')
    if probabilities and (any(v < 0 for v in values) or abs(sum(values)-1) > 1e-5):
        raise ValueError('probability vector must be nonnegative and sum to one')


def _validate_pair(root, row):
    paths = [_artifact(root, row.get(key)) for key in ('trajectory_path','features_path')]
    payloads = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    identities = []
    for payload in payloads:
        if payload.get('schema_version') != 1 or any(payload.get(k) != row[k] for k in ('problem_id','scale','rollout_id','seed')):
            raise ValueError('trajectory/features identity mismatch')
        if payload.get('generation_config') != generation_settings(row['source'],row['scale']):
            raise ValueError('trajectory/features generation config differs from approved request')
        model = payload.get('model_identity')
        if not isinstance(model,str) or not model:
            raise ValueError('missing model_identity')
        identities.append(model)
    if identities[0] != identities[1]:
        raise ValueError('trajectory and features model identity differs')
    trajectory, features = payloads
    if trajectory.get('reward') != row['reward'] or not isinstance(trajectory.get('output_text'),str):
        raise ValueError('trajectory reward/output mismatch')
    tokens = trajectory.get('explicit_token_ids')
    if not isinstance(tokens,list) or any(isinstance(n,bool) or not isinstance(n,int) or n < 0 for n in tokens):
        raise ValueError('missing/invalid explicit token sequence')
    steps, feature_steps = trajectory.get('steps'), features.get('steps')
    if not isinstance(steps,list) or not steps or not isinstance(feature_steps,list) or len(steps)!=len(feature_steps):
        raise ValueError('missing or misaligned latent trajectory/features')
    if features.get('feature_schema')!='raw-latent-v1' or features.get('feature_source')!='pre_gumbel':
        raise ValueError('unsupported or noisy feature schema')
    hidden_size = None
    for index,(step,feature) in enumerate(zip(steps,feature_steps)):
        if step.get('step_idx')!=index or feature.get('step_idx')!=index:
            raise ValueError('latent step alignment mismatch')
        ids = step.get('topk_ids')
        if not isinstance(ids,list) or len(ids)!=10 or any(isinstance(n,bool) or not isinstance(n,int) or n < 0 for n in ids):
            raise ValueError('missing sparse top10 mixture IDs')
        _vector(step.get('mixture_probs'), probabilities=True, size=10)
        _vector(feature.get('hidden_state'), size=hidden_size)
        hidden_size = len(feature['hidden_state'])
        _vector(feature.get('topk_probs'), probabilities=True, size=10)
    return paths, [file_sha256(path) for path in paths], identities[0], hidden_size


def _table_write(path, rows, schema=None):
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def select_probe_dataset(candidate_dir, rollouts_path, output_dir, trajectory_root):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite output: {output}')
    candidate_manifest, mapping = _candidates(candidate_dir)
    rollouts_path = Path(rollouts_path)
    if not rollouts_path.is_file():
        raise FileNotFoundError(f'missing required rollouts: {rollouts_path}')
    rows = pq.read_table(rollouts_path).to_pylist()
    groups = _group(mapping, rows)
    decisions, summaries, pairs = [], [], []
    selected, validation = [], []
    models, dimensions = set(), set()
    for pid, candidate in mapping.items():
        decision = decide_probe(groups[pid], candidate['split'])
        if not decision['terminal']:
            raise ValueError(f'incomplete probing: {pid}; target_m={decision["target_m"]}')
        record = {k: candidate[k] for k in ('problem_id','source','source_row','source_file_sha256','split')}
        record.update({k: decision[k] for k in ('reason','keep','delta','s_star','stage')})
        decisions.append(record)
        for summary in decision['summary']:
            scale_rows = [r for r in groups[pid] if r['scale']==summary['scale']]
            summaries.append(dict(summary, problem_id=pid,split=candidate['split'],source=candidate['source'],
                invalid_count=sum(r.get('invalid_reason') is not None for r in scale_rows),
                reason=decision['reason'],stage=decision['stage'],delta=decision['delta']))
        for row in groups[pid]:
            if row['scale'] != decision['s_star']:
                continue
            paths, hashes, model, hidden_size = _validate_pair(trajectory_root,row)
            models.add(model); dimensions.add(hidden_size)
            pairs.append((row,paths,hashes))
        if candidate['split']=='train' and decision['keep']:
            selected.append(dict(candidate, noise_target=decision['s_star']))
        elif candidate['split']=='validation':
            validation.append(dict(candidate, probe_reason=decision['reason'],noise_target=decision['s_star']))
    if len(models)!=1 or len(dimensions)!=1:
        raise ValueError('mixed/missing model identity or hidden feature size')
    output.mkdir(parents=True,exist_ok=False)
    (output/'trajectories').mkdir()
    retained = []
    for row, paths, hashes in pairs:
        key = hashlib.sha256(json.dumps([row['problem_id'],row['scale'],row['rollout_id']]).encode()).hexdigest()
        record = {k: row[k] for k in ('problem_id','source','split','scale','rollout_id','seed','reward')}
        for field,path,digest in zip(('trajectory_path','features_path'),paths,hashes):
            relative = f'trajectories/{key}_{field}.json'
            shutil.copyfile(path,output/relative)
            if file_sha256(output/relative)!=digest:
                raise ValueError('trajectory/features input changed while copying')
            record[field]=relative
        retained.append(record)
    train_schema = pa.schema(list(ROW_SCHEMA)+[pa.field('noise_target',pa.float64())])
    val_schema = pa.schema(list(train_schema)+[pa.field('probe_reason',pa.string())])
    _table_write(output/'train_selected.parquet',selected,train_schema)
    _table_write(output/'validation_marked.parquet',validation,val_schema)
    _table_write(output/'selection_decisions.parquet',decisions)
    _table_write(output/'probe_summary.parquet',summaries)
    light_fields = ('problem_id','source','split','scale','rollout_id','seed','status','reward',
                    'invalid_reason','truncated','finish_reason','extracted_answer')
    _table_write(output/'probe_rollouts.parquet',[{k:r.get(k) for k in light_fields} for r in rows])
    _table_write(output/'retained_trajectories.parquet',retained)
    manifest = {'schema_version':1,'status':'complete','candidate_manifest_sha256':file_sha256(Path(candidate_dir)/'build_manifest.json'),
                'rollouts_sha256':file_sha256(rollouts_path),'model_identity':next(iter(models)),
                'feature_schema':'raw-latent-v1','hidden_size':next(iter(dimensions)),
                'counts':{'train_selected':len(selected),'validation':len(validation),
                          'rollouts':len(rows),'retained_trajectories':len(retained)},
                'scales':[0,.25,.5,1,2],'root_seed':12345,'seed_algorithm':'probe-sha256-v1',
                'scope':'offline_selection_no_model_execution',
                'output_sha256':{p.name:file_sha256(p) for p in sorted(output.glob('*.parquet'))}}
    temp=output/'selection_manifest.json.tmp'
    temp.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    temp.rename(output/'selection_manifest.json')
    return manifest
