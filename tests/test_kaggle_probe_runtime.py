import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import pytest
from tests.test_probe_selection import selection_fixture

ROOT=Path(__file__).resolve().parents[1]


def script():
    spec=importlib.util.spec_from_file_location('probe_runtime',ROOT/'scripts/kaggle_probe.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_uses_controller_and_stops_before_budget_then_resumes(tmp_path):
    m=script()
    from adaptive_noise.kaggle_probing import ProbeStore
    candidates,_,artifacts,rows=selection_fixture(tmp_path)
    bykey={m.request_key(r):r for r in rows}
    store=ProbeStore(tmp_path/'collection',candidates,'synthetic_cpu_fixture')
    clock=[0.]
    calls=[]
    def sample(candidate,request,attempt):
        clock[0]+=1
        calls.append(request)
        row=bykey[m.request_key(request)]
        for key,name in [('features_path','features.json'),('trajectory_path','trajectory.json')]:
            (attempt/name).write_bytes((artifacts/row[key]).read_bytes())
        return row
    status=m.collect_pending(store,sample,deadline=2.,clock=lambda:clock[0])
    assert status=='budget_stopped' and len(store.history)==2
    completed={m.request_key(r) for r in calls}
    store=ProbeStore(tmp_path/'collection',candidates,'synthetic_cpu_fixture')
    assert m.collect_pending(store,sample,deadline=1000.,clock=lambda:clock[0])=='complete'
    assert len(store.history)==len(rows)
    assert len({m.request_key(r) for r in calls})==len(calls)
    assert all(sum(r['scale']==0 and r['problem_id']==p for r in calls)==1 for p in store.mapping)
    assert (store.root/'selected/selection_manifest.json').exists()
    assert completed.issubset({m.request_key(r) for r in store.history})


def test_runtime_error_does_not_become_reward_or_retry(tmp_path):
    m=script()
    from adaptive_noise.kaggle_probing import ProbeStore
    candidates,_,_,_=selection_fixture(tmp_path)
    store=ProbeStore(tmp_path/'collection',candidates,'synthetic_cpu_fixture')
    calls=[]
    def fail(*args):
        calls.append(1)
        raise RuntimeError('simulated OOM')
    with pytest.raises(RuntimeError,match='OOM'):
        m.collect_pending(store,fail,deadline=10,clock=lambda:0)
    assert calls==[1] and not store.history
    assert json.loads((store.root/'collection_status.json').read_text())['status']=='failed'


def test_watchdog_stops_process_at_deadline():
    m=script()
    start=time.monotonic()
    assert m.supervise([sys.executable,'-c','import time; time.sleep(30)'],{},timeout=.2)=='budget_stopped'
    assert time.monotonic()-start<3
    with pytest.raises(subprocess.CalledProcessError):
        m.supervise([sys.executable,'-c','raise SystemExit(3)'],{},timeout=5)


def test_high_and_low_scoring_use_actual_author_helpers():
    m=script()
    base=ROOT/'latent_grpo_minfix/Latent-GRPO-final/Latent-GRPO/eval'
    assert m.grade('gsm8k_aug',r'\boxed{3}','3',base/'eval_low_tasks_sglang.py')['reward']==1
    assert m.grade('dapo_math',r'\boxed{3}','3',base/'eval_high_tasks_sglang.py')['reward']==1
    assert m.grade('dapo_math','no final answer','3',base/'eval_high_tasks_sglang.py')['invalid_reason']=='answer_unextractable'
    assert m.grade('dapo_math','', '3',base/'eval_high_tasks_sglang.py')['reward']==0
    assert m.MAX_SECONDS==8*3600
    assert m.SHUTDOWN_MARGIN_SECONDS>=60


def test_formal_notebook_bundle_matches_sources_and_preserves_setup():
    import ast
    m=script()
    notebook=json.loads((ROOT/'notebooks/ecnu-smoke-adaptive-latent-grpo.ipynb').read_text())
    cell=''.join(notebook['cells'][9]['source'])
    assert 'ADANOISE_PROJECT_ROOT=' in cell
    embedded=cell.split("cat > /kaggle/working/test_sglang_seed.py <<'PY'\n",1)[1].split('\nPY\n',1)[0]
    assert embedded==(ROOT/'scripts/kaggle_probe.py').read_text().rstrip('\n')
    bootstrap=cell.split("<<'BUNDLE'\n",1)[1].split('\nBUNDLE\n',1)[0]
    tree=ast.parse(bootstrap)
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='files' for t in n.targets))
    files=ast.literal_eval(assignment.value)
    for relative,content in files.items():
        assert content==(ROOT/relative).read_text()
    assert 'adaptive_noise/kaggle_probing.py' in files
    assert 'configs/kaggle_candidate_rows.json' in files
    archived=json.loads((ROOT/'notebooks/kaggle_request_seed_smoke.ipynb').read_text())
    assert notebook['cells'][1:9]==archived['cells'][1:9]
    assert '--hours 8' in cell
    assert 'checkout --detach' not in cell


def test_parent_watchdog_covers_preparation_before_any_model_input(monkeypatch,tmp_path):
    m=script()
    calls=[]
    def supervise(command,env,timeout):
        calls.append((command,timeout))
        assert '--prepare' in command
        assert timeout<=8*3600-30
        return 'budget_stopped'
    monkeypatch.setattr(m,'supervise',supervise)
    monkeypatch.setattr(sys,'argv',['kaggle_probe.py','--output-dir',str(tmp_path/'collection')])
    assert m.main()==0
    assert len(calls)==1
    assert json.loads((tmp_path/'collection/watchdog.json').read_text())['status']=='budget_stopped'


@pytest.mark.parametrize('source,scale',[('gsm8k_aug',0.),('dapo_math',2.)])
def test_collect_output_dynamic_identity_and_sparse_features(tmp_path,source,scale):
    from adaptive_noise.seeds import derive_rollout_seed
    from adaptive_noise.probing import generation_settings
    m=script()
    attempt=tmp_path/('a'*32)
    (attempt/'events').mkdir(parents=True)
    req=dict(problem_id='fixture',source=source,split='train',scale=scale,rollout_id=0,
             seed=derive_rollout_seed('fixture',scale,0),**generation_settings(source,scale))
    records=[]
    for i,(token,latent,mixed) in enumerate([(1,True,True),(524,True,False),(2,False,False)]):
        record=dict(generation_idx=i,next_token_id=token,latent_mode=latent,is_mixture_step=mixed,
            worker_pid=42,seed=req['seed'],cuda_seed=req['seed'],request_id='request',run_id=attempt.name,
            request_seed_applied=i==0)
        if mixed:
            record.update(hidden_state=[.1,.2],topk_probs=[.1]*10,clean_topk_ids=list(range(1,11)),
                          topk_ids=list(range(1,11)),mixture_probs=[.1]*10)
        records.append(record)
    (attempt/'events/42.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    from types import SimpleNamespace
    tokenizer=SimpleNamespace(decode=lambda *a,**k:r'\boxed{42}')
    scoring=ROOT/'latent_grpo_minfix/Latent-GRPO-final/Latent-GRPO/eval'
    row=m.collect_output(attempt,dict(source=source,split='train',ground_truth='42'),req,
        dict(output_ids=[1,524,2],meta_info={'id':'request','finish_reason':{'type':'stop'}}),tokenizer,
        dict(hidden_size=2,model_identity='cpu',scoring_paths={source:str(scoring/('eval_low_tasks_sglang.py' if source=='gsm8k_aug' else 'eval_high_tasks_sglang.py'))}),524,.5)
    assert row['reward']==1 and row['scale']==scale and row['num_latent_steps']==1
    features=json.loads((attempt/'features.json').read_text())
    assert features['generation_config']==generation_settings(source,scale)
    assert len(features['steps'])==1 and len(features['steps'][0]['topk_probs'])==10


@pytest.mark.parametrize('fail',[False,True])
def test_worker_one_engine_dynamic_requests_and_shutdown(monkeypatch,tmp_path,fail):
    from types import SimpleNamespace
    import shutil
    from adaptive_noise.kaggle_probing import ProbeStore
    m=script()
    candidates,_,artifacts,rows=selection_fixture(tmp_path)
    root=tmp_path/'collection'
    root.mkdir();shutil.copytree(candidates,root/'candidates')
    session=root/'sessions/cpu';session.mkdir(parents=True)
    (session/'input.json').write_text(json.dumps({'model_identity':'synthetic_cpu_fixture'}))
    byseed={r['seed']:r for r in rows}
    calls=[]
    class Engine:
        def __init__(self,**kwargs):calls.append(('init',kwargs))
        def generate(self,**kwargs):
            calls.append(('generate',kwargs))
            if fail:raise RuntimeError('simulated OOM')
            return kwargs['sampling_params']
        def shutdown(self):calls.append(('shutdown',None))
    tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**k:'question<think>',encode=lambda text,**k:[524] if text=='</think>' else [1])
    class Monitor:
        error=None;samples=[14000.]
        thread=SimpleNamespace(start=lambda:None)
        def stop(self):pass
    def collect(attempt,candidate,request,output,*args):
        assert output['custom_params']['adanoise_smoke']=={'run_id':attempt.name,'seed':request['seed']}
        expected=m.generation_settings(candidate['source'],request['scale'])
        assert all(output[k]==v for k,v in expected.items() if k!='max_topk')
        row=byseed[request['seed']]
        for key,name in [('features_path','features.json'),('trajectory_path','trajectory.json')]:
            (attempt/name).write_bytes((artifacts/row[key]).read_bytes())
        return dict(row,worker_pid=42,request_id=attempt.name)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(__version__='cpu_mock',cuda=SimpleNamespace(is_available=lambda:True,get_device_name=lambda i:'Tesla T4')))
    monkeypatch.setitem(sys.modules,'sglang',SimpleNamespace(Engine=Engine,__version__='cpu_mock'))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:tokenizer)))
    monkeypatch.setattr(m.smoke,'MemoryMonitor',Monitor)
    monkeypatch.setattr(m,'collect_output',collect)
    if fail:
        with pytest.raises(RuntimeError,match='OOM'):m.run_worker(root,session,time.monotonic()+100)
        assert not ProbeStore(root,root/'candidates','synthetic_cpu_fixture').history
    else:
        m.run_worker(root,session,time.monotonic()+100)
        assert len([c for c in calls if c[0]=='generate'])==len(rows)
        assert json.loads((session/'result.json').read_text())['status']=='complete'
    assert len([c for c in calls if c[0]=='init'])==1
    assert calls[0][1]['disable_radix_cache'] is True
    assert calls[-1][0]=='shutdown'


def test_fresh_process_imports_exported_notebook_bundle(tmp_path):
    import ast
    import os
    n=json.loads((ROOT/'notebooks/ecnu-smoke-adaptive-latent-grpo.ipynb').read_text())
    cell=''.join(n['cells'][9]['source'])
    bootstrap=cell.split("<<'BUNDLE'\n",1)[1].split('\nBUNDLE\n',1)[0]
    bundle=tmp_path/'bundle'
    bootstrap=bootstrap.replace("Path('/kaggle/working/adanoise_probe_bundle')",repr(bundle))
    # Path repr is PosixPath(...); import it only for this local fixture bootstrap.
    bootstrap='from pathlib import PosixPath\n'+bootstrap
    subprocess.run([sys.executable,'-c',bootstrap],check=True,cwd=tmp_path,capture_output=True,text=True)
    embedded=cell.split("cat > /kaggle/working/test_sglang_seed.py <<'PY'\n",1)[1].split('\nPY\n',1)[0]
    entry=tmp_path/'test_sglang_seed.py';entry.write_text(embedded)
    env=os.environ.copy();env['ADANOISE_PROJECT_ROOT']=str(bundle)
    completed=subprocess.run([sys.executable,str(entry),'--help'],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert completed.returncode==0 and '--hours' in completed.stdout


def test_watchdog_kills_child_even_if_leader_exits_on_term(tmp_path):
    m=script()
    heartbeat=tmp_path/'child.heartbeat'
    child=("import signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
           "stream=open(sys.argv[1],'a',buffering=1)\n"
           "for i in range(3000):\n    stream.write(str(i)+'\\n'); time.sleep(.01)\n")
    parent=("import subprocess,sys,time; "
            "subprocess.Popen([sys.executable,'-c',sys.argv[2],sys.argv[1]]); time.sleep(30)")
    assert m.supervise([sys.executable,'-c',parent,str(heartbeat),child],{},timeout=.7)=='budget_stopped'
    assert heartbeat.is_file() and heartbeat.stat().st_size>0  # proves child ran with SIGTERM ignored
    time.sleep(.05)  # let the delivered SIGKILL settle, without system process inspection
    size=heartbeat.stat().st_size
    time.sleep(.15)
    assert heartbeat.stat().st_size==size  # the formerly active child stopped writing
