import json
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from adaptive_noise.probe_selection import select_probe_dataset, plan_candidate_requests
from adaptive_noise.seeds import derive_rollout_seed
from adaptive_noise.dataset_builder import build_dataset
from tests.test_dataset_builder import fixture_config
from tests.test_probing import records


def selection_fixture(tmp_path):
    config = fixture_config(tmp_path)
    config.update(sample_per_source=2, train_per_source=1, validation_per_source=1)
    build_dataset(config, tmp_path)
    candidates = Path(config['output_dir'])
    subjects = pq.read_table(candidates / 'train.parquet').to_pylist() + pq.read_table(candidates / 'validation.parquet').to_pylist()
    trajectory_root = tmp_path / 'trajectories'
    trajectory_root.mkdir()
    rollouts = []
    for n, subject in enumerate(subjects):
        pattern = ('sensitive' if subject['source'] == 'gsm8k_aug' else 'wrong') if subject['split'] == 'train' else ('correct' if subject['source'] == 'gsm8k_aug' else 'capped')
        counts = {'sensitive': [0,2,2,1,1], 'wrong': [0,0,0,0,0], 'correct': [1,3,3,3,3], 'capped': [0,2,2,2,2]}[pattern]
        rows = records(7 if pattern == 'capped' else 3, counts)
        for row in rows:
            if pattern == 'capped' and row['scale'] != 0:
                row['reward'] = int(row['rollout_id'] in (0,3))
            row.update(problem_id=subject['problem_id'], source=subject['source'], split=subject['split'],
                       seed=derive_rollout_seed(subject['problem_id'], row['scale'], row['rollout_id']),
                       invalid_reason=None, finish_reason='stop', final_answer=str(row['reward']),
                       extracted_answer=str(row['reward']))
            key = f'{n}_{row["scale"]}_{row["rollout_id"]}'
            generation = {'max_new_tokens':128 if subject['source']=='gsm8k_aug' else 4096,
                          'temperature':.6,'top_p':.95,'max_topk':10, 'gumbel_softmax_temperature':1.,
                          'noise_scale':row['scale'],'add_noise_gumbel_softmax':True,'use_one_sided_gumbel_noise':False}
            identity = {'schema_version':1, 'seed':row['seed'], 'generation_config':generation,
                        'problem_id':row['problem_id'], 'scale':row['scale'],
                        'rollout_id':row['rollout_id'], 'model_identity':'synthetic_cpu_fixture'}
            trajectory = dict(identity, output_text=str(row['reward']), explicit_token_ids=[1,2],
                              reward=row['reward'], steps=[{'step_idx':0,'topk_ids':list(range(10)), 'mixture_probs':[.1]*10}])
            features = dict(identity, feature_schema='raw-latent-v1', feature_source='pre_gumbel',
                            steps=[{'step_idx':0, 'hidden_state':[.1,.2], 'topk_probs':[.1]*10}])
            for field, payload in [('trajectory_path', trajectory), ('features_path', features)]:
                path = trajectory_root / f'{key}_{field}.json'
                path.write_text(json.dumps(payload))
                row[field] = path.name
            rollouts.append(row)
    path = tmp_path / 'rollouts.parquet'
    pq.write_table(pa.Table.from_pylist(rollouts), path)
    return candidates, path, trajectory_root, rollouts


def test_best_scale_all_trajectories_and_validation_marks(tmp_path):
    candidates, path, root, rows = selection_fixture(tmp_path)
    output = tmp_path / 'selected'
    manifest = select_probe_dataset(candidates, path, output, root)
    selected = pq.read_table(output / 'train_selected.parquet').to_pylist()
    assert len(selected) == 1
    decisions = pq.read_table(output / 'selection_decisions.parquet').to_pylist()
    assert sorted(d['reason'] for d in decisions if d['split']=='validation') == ['all_correct','not_sensitive_at_cap']
    assert all(d['keep'] for d in decisions if d['split']=='validation')
    trajectories = pq.read_table(output / 'retained_trajectories.parquet').to_pylist()
    best_train = [r for r in trajectories if r['problem_id']==selected[0]['problem_id']]
    assert len(best_train) == 3 and {r['reward'] for r in best_train} == {0,1}
    assert all(r['scale']==.25 for r in best_train)
    assert len(trajectories)==12
    assert len(pq.read_table(output / 'probe_rollouts.parquet')) == len(rows)
    assert 'trajectory_path' not in pq.read_table(output / 'probe_rollouts.parquet').column_names
    for row in trajectories:
        assert (output / row['trajectory_path']).is_file() and (output / row['features_path']).is_file()
    assert len(list(root.glob('*.json'))) == len(rows)*2  # source data untouched
    assert manifest['status']=='complete' and manifest['model_identity']=='synthetic_cpu_fixture'
    assert len(pq.read_table(output / 'validation_marked.parquet')) == 2
    rejected = [d for d in decisions if not d['keep']]
    assert rejected[0]['source_row'] >= 0 and rejected[0]['problem_id']


@pytest.mark.parametrize('mutation', ['missing_feature', 'wrong_seed', 'unknown_id', 'missing_rollout', 'duplicate', 'bad_feature_identity', 'nonfinite_feature'])
def test_selection_stops_before_publishing_on_incomplete_contract(tmp_path, mutation):
    candidates, path, root, rows = selection_fixture(tmp_path)
    # First subject is GSM train, best scale .25.
    best = next(r for r in rows if r['split']=='train' and r['source']=='gsm8k_aug' and r['scale']==.25)
    if mutation=='missing_feature': (root / best['features_path']).unlink()
    if mutation=='wrong_seed': rows[0]['seed'] += 1
    if mutation=='unknown_id': rows[0]['problem_id']='foreign'
    if mutation=='missing_rollout': rows.pop()
    if mutation=='duplicate': rows.append(rows[0].copy())
    if mutation in {'bad_feature_identity','nonfinite_feature'}:
        p=root/best['features_path']; payload=json.loads(p.read_text())
        if mutation=='bad_feature_identity': payload['problem_id']='foreign'
        else: payload['steps'][0]['hidden_state'][0]=float('nan')
        p.write_text(json.dumps(payload))
    pq.write_table(pa.Table.from_pylist(rows), path)
    out=tmp_path/'invalid-selection'
    with pytest.raises((ValueError, FileNotFoundError)):
        select_probe_dataset(candidates,path,out,root)
    assert not (out/'selection_manifest.json').exists()


def test_plan_all_candidate_requests_without_model_and_resume(tmp_path):
    config=fixture_config(tmp_path)
    build_dataset(config,tmp_path)
    planned=plan_candidate_requests(Path(config['output_dir']), [])
    assert len(planned)==4160
    assert all(r['split'] in {'train','validation'} for r in planned)
    first=planned[0]
    completed=dict(first,reward=0,status='completed')
    resumed=plan_candidate_requests(Path(config['output_dir']),[completed])
    assert len(resumed)==4159
    assert (first['problem_id'],first['scale'],first['rollout_id']) not in {(r['problem_id'],r['scale'],r['rollout_id']) for r in resumed}


def test_selection_cli_missing_input(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/select_probe_dataset.py'
    result=subprocess.run([sys.executable,str(script),'--candidate-dir',str(tmp_path/'missing'),
        '--rollouts',str(tmp_path/'missing.parquet'),'--trajectory-root',str(tmp_path),
        '--output-dir',str(tmp_path/'out')],capture_output=True,text=True)
    assert result.returncode==2 and 'STOP:' in result.stderr


def test_completed_selection_preserves_truncation_without_changing_reward(tmp_path):
    candidates,path,root,rows=selection_fixture(tmp_path)
    chosen=next(r for r in rows if r['scale']==.25 and r['reward']==1)
    for row in rows:
        row['truncated']=row is chosen
    chosen['finish_reason']='length'
    pq.write_table(pa.Table.from_pylist(rows),path)
    out=tmp_path/'truncated-output'
    select_probe_dataset(candidates,path,out,root)
    light=pq.read_table(out/'probe_rollouts.parquet').to_pylist()
    record=next(r for r in light if r['problem_id']==chosen['problem_id'] and r['scale']==chosen['scale'] and r['rollout_id']==chosen['rollout_id'])
    assert record['truncated'] is True and record['reward']==1


def test_successful_selection_and_planning_clis(tmp_path):
    candidates,path,root,rows=selection_fixture(tmp_path)
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    out=tmp_path/'cli-selection'
    selected=subprocess.run([sys.executable,str(scripts/'select_probe_dataset.py'),
        '--candidate-dir',str(candidates),'--rollouts',str(path),'--trajectory-root',str(root),
        '--output-dir',str(out)],capture_output=True,text=True)
    assert selected.returncode==0, selected.stderr
    assert json.loads(selected.stdout)['counts']['retained_trajectories']==12
    planned=subprocess.run([sys.executable,str(scripts/'plan_probe_requests.py'),
        '--candidate-dir',str(candidates),'--output-dir',str(tmp_path/'cli-requests')],capture_output=True,text=True)
    assert planned.returncode==0, planned.stderr
    assert json.loads(planned.stdout)['requests']==52


@pytest.mark.parametrize('field', ['trajectory_path','features_path'])
@pytest.mark.parametrize('mutation', ['missing_seed','wrong_seed','wrong_generation'])
def test_best_artifact_seed_and_generation_must_match_approved_row(tmp_path, field, mutation):
    candidates,path,root,rows=selection_fixture(tmp_path)
    best=next(r for r in rows if r['source']=='gsm8k_aug' and r['split']=='train' and r['scale']==.25)
    artifact=root/best[field]
    payload=json.loads(artifact.read_text())
    if mutation=='missing_seed': del payload['seed']
    elif mutation=='wrong_seed': payload['seed']+=1
    else: payload['generation_config']['max_new_tokens']=999
    artifact.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        select_probe_dataset(candidates,path,tmp_path/'bad-artifact',root)
