"""Durable collection of approved fixed-pool probing; no model imports."""
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
from adaptive_noise.dataset_builder import ROW_SCHEMA, _normalize, file_sha256, iter_rows
from adaptive_noise.probe_selection import _candidates, _group, _validate_pair, select_probe_dataset
from adaptive_noise.probing import decide_probe, plan_requests


def atomic_json(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def request_key(row):
    return hashlib.sha256(json.dumps([row['problem_id'],float(row['scale']),row['rollout_id']],
                                    separators=(',',':')).encode()).hexdigest()


def load_pool(directory):
    return _candidates(directory)[1]


def make_locked_pool(directory):
    manifest, mapping = _candidates(directory)
    return {'schema_version':1, 'master_seed':42,
            'original_manifest_sha256':file_sha256(Path(directory)/'build_manifest.json'),
            'inputs':{s:{'sha256':manifest['inputs'][s]['sha256']} for s in ('gsm8k_aug','dapo_math')},
            'rows':[{k:r[k] for k in ('source','source_row','split')} for r in mapping.values()]}


def restore_candidates(locked, paths, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if locked['schema_version'] != 1 or locked['master_seed'] != 42:
        raise ValueError('unknown locked pool protocol')
    digests = {}
    for source in ('gsm8k_aug','dapo_math'):
        digests[source] = file_sha256(paths[source])
        if digests[source] != locked['inputs'][source]['sha256']:
            raise ValueError(f'input identity mismatch: {source}')
    selected = {}
    for r in locked['rows']:
        key = (r['source'],r['source_row'])
        if key in selected or r['split'] not in ('train','validation') or r['source'] not in digests:
            raise ValueError('invalid locked row/split')
        if type(r['source_row']) is not int or r['source_row'] < 0:
            raise ValueError('invalid locked source_row')
        selected[key] = r['split']
    if len(selected)!=320 or any(sum(s==source and split==part for (s,_),split in selected.items())!=count
        for source in digests for part,count in [('train',120),('validation',40)]):
        raise ValueError('locked pool must retain approved 120/40 per source')
    found = {}
    for source in digests:
        for index,row in iter_rows(paths[source]):
            key=(source,index)
            if key in selected:
                normalized=_normalize(row,source,digests[source],index)
                normalized['split']=selected[key]
                found[key]=normalized
    if found.keys()!=selected.keys():
        raise ValueError('missing locked source row')
    # Preserve the local fixed-pool order, independent of Python/Kaggle RNG version.
    records=[found[(r['source'],r['source_row'])] for r in locked['rows']]
    if any(file_sha256(paths[s])!=digests[s] for s in digests):
        raise ValueError('input changed while restoring pool')
    final=output
    output=final.with_name(final.name+'.pending-'+uuid.uuid4().hex)
    output.mkdir(parents=True)
    for split in ('train','validation'):
        pq.write_table(pa.Table.from_pylist([r for r in records if r['split']==split],schema=ROW_SCHEMA),output/f'{split}.parquet')
    atomic_json(output/'build_manifest.json',dict(status='complete',scope='locked_train_validation_restore_no_test_read',
        original_manifest_sha256=locked['original_manifest_sha256'],master_seed=42,
        inputs=locked['inputs'],counts={'train':240,'validation':80},
        output_sha256={p.name:file_sha256(p) for p in output.glob('*.parquet')}))
    output.rename(final)


class ProbeStore:
    def __init__(self, root, candidates, model_identity):
        self.root, self.candidates = Path(root), Path(candidates)
        self.mapping=load_pool(candidates)
        self.model_identity=model_identity
        self.root.mkdir(parents=True,exist_ok=True)
        identity={'candidate_manifest_sha256':file_sha256(self.candidates/'build_manifest.json'),
                  'model_identity':model_identity,'protocol':'adanoise.fixed-probing.v1',
                  'scales':[0,.25,.5,1,2],'prefix_cache_enabled':False,'max_running_requests':1}
        path=self.root/'collection_identity.json'
        if path.exists():
            if json.loads(path.read_text())!=identity:
                raise ValueError('collection candidate/model identity mismatch')
        else:
            if (self.root/'rollouts').exists():
                raise ValueError('collection identity missing for existing rollouts')
            atomic_json(path,identity)
        for name in ('rollouts','attempts'):
            (self.root/name).mkdir(exist_ok=True)
        self.history=[]
        for path in sorted((self.root/'rollouts').glob('*/completion.json')):
            row=json.loads(path.read_text())
            if request_key(row)!=path.parent.name or row.get('model_identity')!=model_identity:
                raise ValueError('completed rollout identity/model mismatch')
            self.history.append(row)
            if (path.parent/'events').exists():
                shutil.rmtree(path.parent/'events')  # finish cleanup of own atomically committed request
        self.groups=_group(self.mapping,self.history)
        self.decisions={}
        for candidate in self.mapping.values():
            decide_probe(self.rows(candidate),candidate['split'])
        self.finish_questions()

    def rows(self,candidate):
        return self.groups[candidate['problem_id']]

    def pending(self,candidate):
        return plan_requests(candidate['problem_id'],candidate['split'],candidate['source'],self.rows(candidate))

    def new_attempt(self):
        directory=self.root/'attempts'/uuid.uuid4().hex
        (directory/'events').mkdir(parents=True)
        return directory

    def commit(self,row,attempt):
        attempt=Path(attempt).resolve()
        if attempt.parent != (self.root/'attempts').resolve():
            raise ValueError('attempt outside this collection')
        key=request_key(row)
        target=self.root/'rollouts'/key
        if target.exists():
            raise ValueError('rollout already completed')
        candidate=self.mapping[row['problem_id']]
        if not any(request_key(req)==key for req in self.pending(candidate)):
            raise ValueError('unexpected/terminal rollout request')
        completed=dict(row,status='completed',model_identity=self.model_identity,
                       trajectory_path='trajectory.json',features_path='features.json')
        _group(self.mapping,[completed])
        _,_,model,_=_validate_pair(attempt,completed)
        if model!=self.model_identity:
            raise ValueError('collected model identity mismatch')
        completed.update(trajectory_path=f'rollouts/{key}/trajectory.json',features_path=f'rollouts/{key}/features.json')
        atomic_json(attempt/'completion.json',completed)
        # Atomic directory rename publishes both features and result together.
        attempt.rename(target)
        self.history.append(completed)
        self.groups[completed['problem_id']].append(completed)
        shutil.rmtree(target/'events')  # own transient per-token stream; never source data
        return completed

    def finish_questions(self):
        decisions=self.decisions
        for candidate in self.mapping.values():
            if candidate['problem_id'] in decisions:
                continue
            rows=self.rows(candidate)
            decision=decide_probe(rows,candidate['split'])
            if not decision['terminal']:
                continue
            decisions[candidate['problem_id']]=decision
            for row in rows:
                if row['scale']==decision['s_star']:
                    _validate_pair(self.root,row)
                else:
                    for field in ('trajectory_path','features_path'):
                        path=(self.root/row[field]).resolve()
                        expected=(self.root/'rollouts'/request_key(row)/Path(row[field]).name).resolve()
                        if path!=expected or Path(row[field]).name not in ('features.json','trajectory.json'):
                            raise ValueError('invalid generated artifact path')
                        path.unlink(missing_ok=True)
                    (self.root/'rollouts'/request_key(row)/'prompt.json').unlink(missing_ok=True)
        atomic_json(self.root/'decisions.json',decisions)
        return decisions

    def snapshot(self,status):
        path=self.root/'probe_rollouts.parquet'
        # Nulls remain typed until subsequent records arrive; write only nonempty history.
        if self.history:
            temp=path.with_suffix('.parquet.tmp')
            pq.write_table(pa.Table.from_pylist(self.history),temp)
            temp.replace(path)
        groups={}
        for row in self.history:
            group=groups.setdefault((row['source'],row['scale']),{'count':0,'success':0,'invalid':0,'generation_sec':0.})
            group['count']+=1; group['success']+=row['reward']
            group['invalid']+=row.get('invalid_reason') is not None
            group['generation_sec']+=row.get('instrumented_generation_sec',0.)
        report={'status':status,'completed_rollouts':len(self.history),
                'terminal_questions':len(self.finish_questions()),'total_questions':len(self.mapping),
                'by_source_scale':[dict(source=s,scale=scale,**v,q_hat=v['success']/v['count'],
                    invalid_rate=v['invalid']/v['count'],mean_generation_sec=v['generation_sec']/v['count'])
                    for (s,scale),v in sorted(groups.items())]}
        atomic_json(self.root/'collection_status.json',report)
        return report

    def publish(self):
        if any(not decide_probe(self.rows(c),c['split'])['terminal'] for c in self.mapping.values()):
            raise ValueError('incomplete probing; no final selection published')
        self.snapshot('complete')
        out=self.root/'selected'
        if (out/'selection_manifest.json').is_file():
            return json.loads((out/'selection_manifest.json').read_text())
        staging=self.root/'exports'
        staging.mkdir(exist_ok=True)
        pending=staging/uuid.uuid4().hex
        manifest=select_probe_dataset(self.candidates,self.root/'probe_rollouts.parquet',pending,self.root)
        pending.rename(out)
        return manifest
