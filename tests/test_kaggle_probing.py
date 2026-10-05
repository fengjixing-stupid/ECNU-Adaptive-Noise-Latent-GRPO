import json
from pathlib import Path
import pyarrow.parquet as pq
import pytest
from tests.test_probe_selection import selection_fixture
from tests.test_dataset_builder import fixture_config
from adaptive_noise.dataset_builder import build_dataset, file_sha256


def module():
    from adaptive_noise import kaggle_probing
    return kaggle_probing


def test_locked_pool_rebuild_without_resampling(tmp_path):
    m = module()
    cfg = fixture_config(tmp_path)
    build_dataset(cfg, tmp_path)
    original = Path(cfg['output_dir'])
    locked = m.make_locked_pool(original)
    assert len(locked['rows']) == 320
    assert all(set(r) == {'source', 'source_row', 'split'} for r in locked['rows'])
    out = tmp_path / 'restored'
    m.restore_candidates(locked, {s: Path(p) for s,p in cfg['inputs'].items()}, out)
    for split in ('train','validation'):
        assert pq.read_table(out / f'{split}.parquet').to_pylist() == pq.read_table(original / f'{split}.parquet').to_pylist()
    assert len(m.load_pool(out)) == 320
    bad = json.loads(json.dumps(locked))
    bad['inputs']['dapo_math']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='identity'):
        m.restore_candidates(bad, {s: Path(p) for s,p in cfg['inputs'].items()}, tmp_path/'bad')


def test_collection_resume_atomic_and_best_scale_cleanup(tmp_path):
    m = module()
    candidates, _, root, rows = selection_fixture(tmp_path)
    store = m.ProbeStore(tmp_path/'collection', candidates, 'synthetic_cpu_fixture')
    mapping = m.load_pool(candidates)
    first = next(iter(mapping.values()))
    initial = store.pending(first)
    assert len(initial) == 13
    first_row = next(r for r in rows if r['problem_id'] == first['problem_id'] and r['scale']==0)
    stage = store.new_attempt()
    # An interrupted attempt has no committed result and contributes no reward.
    (stage/'features.json').write_text('{}')
    assert len(store.history) == 0 and len(store.pending(first)) == 13
    # Complete each 3/5/7 round before requesting the next one.
    for row in sorted(rows, key=lambda r: (r['problem_id'], 0 if r['rollout_id']<3 else (1 if r['rollout_id']<5 else 2), r['scale'], r['rollout_id'])):
        stage = store.new_attempt()
        for key,name in [('features_path','features.json'),('trajectory_path','trajectory.json')]:
            (stage/name).write_bytes((root/row[key]).read_bytes())
        store.commit(row, stage)
    restored = m.ProbeStore(tmp_path/'collection', candidates, 'synthetic_cpu_fixture')
    assert len(restored.history) == len(rows)
    assert all(not restored.pending(c) for c in mapping.values())
    assert not any(r['scale']==0 for r in restored.pending(first))
    decisions = restored.finish_questions()
    assert len(decisions) == 4
    for row in restored.history:
        d = decisions[row['problem_id']]
        assert (restored.root/row['features_path']).exists() == (row['scale']==d['s_star'])
    manifest = restored.publish()
    assert manifest['counts']['validation'] == 2
    assert manifest['counts']['train_selected'] == 1
    assert len(pq.read_table(restored.root/'probe_rollouts.parquet')) == len(rows)
    assert restored.publish() == manifest
    with pytest.raises(ValueError, match='model'):
        m.ProbeStore(tmp_path/'collection', candidates, 'different-model')
    with pytest.raises(ValueError, match='already'):
        store.commit(first_row, store.new_attempt())


def test_incomplete_cannot_publish_and_failure_does_not_commit(tmp_path):
    m = module()
    candidates, _, root, rows = selection_fixture(tmp_path)
    store = m.ProbeStore(tmp_path/'collection', candidates, 'synthetic_cpu_fixture')
    stage = store.new_attempt()
    (stage/'features.json').write_text('{}')
    (stage/'trajectory.json').write_text('{}')
    with pytest.raises(ValueError):
        store.commit(rows[0], stage)
    assert not store.history
    with pytest.raises(ValueError, match='incomplete'):
        store.publish()
    assert not (store.root/'selected/selection_manifest.json').exists()


def test_interrupted_export_can_resume_without_incomplete_selected(tmp_path,monkeypatch):
    m=module()
    candidates,_,root,rows=selection_fixture(tmp_path)
    store=m.ProbeStore(tmp_path/'collection',candidates,'synthetic_cpu_fixture')
    for row in sorted(rows,key=lambda r:(r['problem_id'],0 if r['rollout_id']<3 else (1 if r['rollout_id']<5 else 2),r['scale'],r['rollout_id'])):
        attempt=store.new_attempt()
        for key,name in [('features_path','features.json'),('trajectory_path','trajectory.json')]:
            (attempt/name).write_bytes((root/row[key]).read_bytes())
        store.commit(row,attempt)
    original=m.shutil.copyfile
    def interrupted(*args,**kwargs):
        raise OSError('interrupted export')
    monkeypatch.setattr(m.shutil,'copyfile',interrupted)
    with pytest.raises(OSError,match='interrupted'):
        store.publish()
    assert not (store.root/'selected').exists()
    monkeypatch.setattr(m.shutil,'copyfile',original)
    store=m.ProbeStore(store.root,candidates,'synthetic_cpu_fixture')
    assert store.publish()['status']=='complete'


def test_interrupted_candidate_write_not_published(tmp_path,monkeypatch):
    m=module()
    cfg=fixture_config(tmp_path)
    build_dataset(cfg,tmp_path)
    locked=m.make_locked_pool(cfg['output_dir'])
    paths={s:Path(p) for s,p in cfg['inputs'].items()}
    out=tmp_path/'restored'
    original=m.pq.write_table
    def interrupted(*args,**kwargs):
        raise OSError('interrupted candidate write')
    monkeypatch.setattr(m.pq,'write_table',interrupted)
    with pytest.raises(OSError,match='interrupted'):
        m.restore_candidates(locked,paths,out)
    assert not out.exists()
    monkeypatch.setattr(m.pq,'write_table',original)
    m.restore_candidates(locked,paths,out)
    assert len(m.load_pool(out))==320


def test_resume_cleans_events_if_stopped_after_completion_rename(tmp_path,monkeypatch):
    m=module()
    candidates,_,root,rows=selection_fixture(tmp_path)
    store=m.ProbeStore(tmp_path/'collection',candidates,'synthetic_cpu_fixture')
    row=rows[0]
    attempt=store.new_attempt()
    (attempt/'events/worker.jsonl').write_text('large temporary hidden record')
    for key,name in [('features_path','features.json'),('trajectory_path','trajectory.json')]:
        (attempt/name).write_bytes((root/row[key]).read_bytes())
    original=m.shutil.rmtree
    def stop(*args,**kwargs):raise OSError('interrupted after rename')
    monkeypatch.setattr(m.shutil,'rmtree',stop)
    with pytest.raises(OSError):store.commit(row,attempt)
    completed=store.root/'rollouts'/m.request_key(row)
    assert (completed/'completion.json').exists() and (completed/'events').exists()
    monkeypatch.setattr(m.shutil,'rmtree',original)
    resumed=m.ProbeStore(store.root,candidates,'synthetic_cpu_fixture')
    assert len(resumed.history)==1
    assert not (completed/'events').exists()
    candidate=resumed.mapping[row['problem_id']]
    assert all(r['scale']!=0 for r in resumed.pending(candidate))
