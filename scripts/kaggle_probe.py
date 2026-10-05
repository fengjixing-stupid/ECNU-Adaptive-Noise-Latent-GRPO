"""Approved Kaggle frozen-checkpoint probing, single persistent Engine, <=8h/run."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ENTRY_STARTED=time.monotonic()

ROOT=Path(os.environ.get('ADANOISE_PROJECT_ROOT',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from adaptive_noise.kaggle_probing import ProbeStore, atomic_json, request_key, restore_candidates
from adaptive_noise.probing import generation_settings
from adaptive_noise.answer_verifier import load_author_helpers
import kaggle_request_seed_smoke as smoke

MAX_SECONDS=8*3600
SHUTDOWN_MARGIN_SECONDS=300
MODEL_PATH=smoke.MODEL_PATH
DATA_ROOT=Path('/kaggle/input/datasets/fengjixing/datasets-for-latent-grpo/data')
FILES={'gsm8k_aug':'GSM8k-Aug-oss-dup-all.parquet','dapo_math':'DAPO-Math-17k-en-train.parquet'}
# Same validated hook, generalized only from three diagnostic names to private attempt IDs.
CAPTURE_SOURCE=smoke.CAPTURE_SOURCE.replace(
    "run_id not in ('same_a', 'same_b', 'different')",
    "not isinstance(run_id, str) or len(run_id) != 32 or any(c not in '0123456789abcdef' for c in run_id)")


def grade(source,text,truth,path):
    profile='low' if source=='gsm8k_aug' else 'high'
    helpers=load_author_helpers(profile,path)
    answer=''
    if text.strip():
        if profile=='low':
            answer=helpers['extract_answer'](text)
        else:
            boxed=helpers['last_boxed_only_string'](text)
            if boxed is not None:
                try:
                    answer=helpers['remove_boxed'](boxed)
                except AssertionError:
                    answer=''
    invalid='empty_output' if not text.strip() else ('answer_unextractable' if not answer.strip() else None)
    return dict(reward=int(helpers['check_is_correct'](answer,truth) if profile=='low' else helpers['is_equiv'](answer,truth)) if invalid is None else 0,
                invalid_reason=invalid,extracted_answer=answer)


def collect_pending(store,sample,deadline,clock=time.monotonic):
    try:
        for candidate in store.mapping.values():
            while True:
                pending=store.pending(candidate)
                if not pending:
                    store.finish_questions()
                    break
                for request in pending:
                    if clock()>=deadline:
                        store.snapshot('budget_stopped')
                        return 'budget_stopped'
                    attempt=store.new_attempt()
                    row=sample(candidate,request,attempt)
                    saved=store.commit(row,attempt)
                    print(json.dumps({k:saved.get(k) for k in ('problem_id','split','scale','rollout_id','seed','reward','invalid_reason','instrumented_generation_sec')})+f' completed={len(store.history)}',flush=True)
            store.snapshot('running')
        store.publish()
        return 'complete'
    except Exception:
        store.snapshot('failed')
        raise  # no automatic retries or conversion of engine errors to reward=0


def stop_group(process):
    try:
        os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # Leader exit alone does not guarantee all Engine workers exited.
    try:
        os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def supervise(command,env,timeout):
    if timeout<=0:
        return 'budget_stopped'
    process=subprocess.Popen(command,env=env,start_new_session=True)
    try:
        code=process.wait(timeout=timeout)
        if code:
            stop_group(process)
            raise subprocess.CalledProcessError(code,command)
        return 'finished'
    except subprocess.TimeoutExpired:
        stop_group(process)
        return 'budget_stopped'
    except BaseException:
        stop_group(process)
        raise


def collect_output(attempt,candidate,request,output,tokenizer,context,end_id,elapsed):
    paths=list((attempt/'events').glob('*.jsonl'))
    if len(paths)!=1:
        raise ValueError('expected one CUDA worker event stream')
    records=[json.loads(line) for line in paths[0].read_text().splitlines()]
    if not records or any(r.get('cuda_seed')!=request['seed'] for r in records):
        raise ValueError('missing/mismatched CUDA request seed')
    if not records[0]['request_seed_applied'] or any(r['request_seed_applied'] for r in records[1:]):
        raise ValueError('request seed must be initialized once')
    if len({r['request_id'] for r in records})!=1 or any(r['run_id']!=attempt.name for r in records):
        raise ValueError('sampler request metadata mismatch')
    rid=records[0]['request_id']
    returned=output.get('meta_info',{}).get('id')
    if returned is not None and returned!=rid:
        raise ValueError('returned request ID differs from sampler')
    ids=output['output_ids']
    checks=smoke.validate_events(records,ids,request['seed'],end_id,context['hidden_size'])
    text=tokenizer.decode(ids,skip_special_tokens=False)
    score=grade(candidate['source'],text,candidate['ground_truth'],context['scoring_paths'][candidate['source']])
    finish=output.get('meta_info',{}).get('finish_reason')
    reason=finish.get('type') if isinstance(finish,dict) else finish
    identity=dict(schema_version=1,**{k:request[k] for k in ('problem_id','scale','rollout_id','seed')},
                  model_identity=context['model_identity'],generation_config=generation_settings(candidate['source'],request['scale']))
    mixed=[r for r in records if r['is_mixture_step']]
    features=dict(identity,feature_schema='raw-latent-v1',feature_source='pre_gumbel',
        steps=[dict(step_idx=i,hidden_state=r['hidden_state'],topk_probs=r['topk_probs']) for i,r in enumerate(mixed)])
    trajectory=dict(identity,output_text=text,reward=score['reward'],all_output_ids=ids,
        explicit_token_ids=[r['next_token_id'] for r in records if not r['latent_mode']],
        steps=[dict(step_idx=i,topk_ids=r['topk_ids'],mixture_probs=r['mixture_probs']) for i,r in enumerate(mixed)])
    atomic_json(attempt/'features.json',features)
    atomic_json(attempt/'trajectory.json',trajectory)
    return dict(identity,source=candidate['source'],split=candidate['split'],**score,**checks,
        request_id=rid,worker_pid=records[0]['worker_pid'],request_seed_initialized_once=True,
        finish_reason=reason,truncated=reason in ('length','max_tokens') if reason else None,
        budget_hit=len(ids)>=request['max_new_tokens'],instrumented_generation_sec=elapsed,
        worker_peak_allocated_mb=max(r.get('worker_peak_allocated_mb',0) for r in records),
        worker_peak_reserved_mb=max(r.get('worker_peak_reserved_mb',0) for r in records),
        memory_scope='peaks_since_engine_start')


def run_worker(root,session,deadline):
    import asyncio
    import torch
    import sglang as sgl
    from transformers import AutoTokenizer
    context=json.loads((session/'input.json').read_text())
    store=ProbeStore(root,root/'candidates',context['model_identity'])
    if time.monotonic()>=deadline:
        store.snapshot('budget_stopped')
        return
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0)!='Tesla T4':
        raise RuntimeError('confirmed T4 GPU0 unavailable')
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True,trust_remote_code=True)
    end_ids=tokenizer.encode('</think>',add_special_tokens=False)
    if not end_ids:
        raise ValueError('missing latent end marker')
    end_id=end_ids[0]
    os.environ['ADANOISE_SMOKE_ROOT']=str(root/'attempts')
    os.environ['ADANOISE_SMOKE_END_ID']=str(end_id)
    engine=None
    loop=None
    monitor=smoke.MemoryMonitor()
    monitor.thread.start()
    worker_pid=None
    seen_rids=set()
    start=time.monotonic()
    try:
        engine=sgl.Engine(model_path=MODEL_PATH,trust_remote_code=True,dtype='float16',kv_cache_dtype='auto',
            tp_size=1,base_gpu_id=0,random_seed=12345,enable_latent=True,latent_end_token_id=end_id,
            disable_cuda_graph=True,disable_overlap_schedule=True,disable_radix_cache=True,
            mem_fraction_static=.90,sampling_backend='flashinfer',max_running_requests=1,
            log_level='info',skip_tokenizer_init=True,max_topk=10)
        loop=asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        startup=time.monotonic()-start
        def sample(candidate,request,attempt):
            nonlocal worker_pid
            text=tokenizer.apply_chat_template(candidate['prompt'],tokenize=False,add_generation_prompt=True)
            if not text.rstrip().endswith('<think>'):
                text+='<think>'
            ids=tokenizer.encode(text,add_special_tokens=False)
            atomic_json(attempt/'prompt.json',dict(text=text,input_ids=ids,end_marker_ids=end_ids))
            params={k:v for k,v in generation_settings(candidate['source'],request['scale']).items() if k!='max_topk'}
            params['custom_params']={'adanoise_smoke':dict(run_id=attempt.name,seed=request['seed'])}
            begin=time.monotonic()
            output=engine.generate(input_ids=ids,sampling_params=params)
            elapsed=time.monotonic()-begin
            if monitor.error:
                raise RuntimeError(f'GPU measurement failed: {monitor.error}')
            row=collect_output(attempt,candidate,request,output,tokenizer,context,end_id,elapsed)
            if worker_pid is not None and worker_pid!=row['worker_pid']:
                raise ValueError('persistent Engine sampling worker changed')
            if row['request_id'] in seen_rids:
                raise ValueError('scheduler reused request ID')
            worker_pid=row['worker_pid']; seen_rids.add(row['request_id'])
            return row
        status=collect_pending(store,sample,deadline)
        monitor.stop()
        if monitor.error or not monitor.samples:
            raise RuntimeError(f'GPU measurement failed: {monitor.error}')
        atomic_json(session/'result.json',dict(status=status,engine_start_count=1,worker_pid=worker_pid,
            startup_sec=startup,completed_rollouts=len(store.history),elapsed_worker_sec=time.monotonic()-start,
            sampled_device_memory_max_mb=max(monitor.samples),memory_sample_interval_sec=.5,
            prefix_cache_enabled=False,max_running_requests=1,gpu=torch.cuda.get_device_name(0),
            torch=torch.__version__,sglang=sgl.__version__))
        print(f'PROBING {status.upper()}: {root}/collection_status.json',flush=True)
    finally:
        monitor.stop()
        if engine is not None:
            engine.shutdown()
        if loop is not None:
            loop.close()
            asyncio.set_event_loop(None)


def main():
    began=ENTRY_STARTED
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('/kaggle/working/artifacts/probing_v1'))
    parser.add_argument('--pool',type=Path,default=ROOT/'configs/kaggle_candidate_rows.json')
    parser.add_argument('--data-root',type=Path,default=DATA_ROOT)
    parser.add_argument('--hours',type=float,default=8.)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--began',type=float)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--session',type=Path)
    parser.add_argument('--deadline',type=float)
    args=parser.parse_args()
    if args.worker:
        run_worker(args.output_dir,args.session,args.deadline)
        return 0
    if not 0<args.hours<=8 or args.hours*3600<=SHUTDOWN_MARGIN_SECONDS+30:
        parser.error('--hours must be >330 seconds and <=8 hours')
    root=args.output_dir.resolve()
    budget=args.hours*3600
    if not args.prepare:
        command=[sys.executable,str(Path(__file__).resolve()),'--prepare','--began',str(began),
            '--output-dir',str(root),'--pool',str(args.pool.resolve()),'--data-root',str(args.data_root),
            '--hours',str(args.hours)]
        try:
            status=supervise(command,os.environ.copy(),timeout=began+budget-30-time.monotonic())
            if status=='budget_stopped':
                root.mkdir(parents=True,exist_ok=True)
                atomic_json(root/'watchdog.json',dict(status=status,total_elapsed_sec=time.monotonic()-began,
                    note='Preparation and generation supervised together; committed rollouts survive.'))
                print(f'PROBING BUDGET_STOPPED: resume with the same --output-dir {root}',flush=True)
            return 0
        except subprocess.CalledProcessError as error:
            print(f'STOP: preparation/Engine process exited {error.returncode}; see failure.json',file=sys.stderr)
            return 2
    if args.began is None:
        parser.error('--prepare requires --began from the supervising parent')
    began=args.began
    session=root/'sessions'/uuid.uuid4().hex
    store=None
    try:
        if not Path(MODEL_PATH).is_dir():
            raise FileNotFoundError(MODEL_PATH)
        locked=json.loads(args.pool.read_text())
        root.mkdir(parents=True,exist_ok=True)
        pool_path=root/'locked_pool.json'
        if pool_path.exists() and json.loads(pool_path.read_text())!=locked:
            raise ValueError('locked pool changed; cannot resume this collection')
        atomic_json(pool_path,locked)
        if not (root/'candidates').exists():
            restore_candidates(locked,{s:args.data_root/name for s,name in FILES.items()},root/'candidates')
        else:
            # Resume still checks mounted source bytes, without rerunning sampling/splitting.
            for source,name in FILES.items():
                if smoke.file_hash(args.data_root/name)!=locked['inputs'][source]['sha256']:
                    raise ValueError(f'input identity mismatch: {source}')
        session.mkdir(parents=True)
        overlay,low,provenance=smoke.prepare_overlay(session)
        (overlay/'adanoise_smoke_capture.py').write_text(CAPTURE_SOURCE)
        high=low.with_name('eval_high_tasks_sglang.py')
        for profile,path in [('low',low),('high',high)]:
            load_author_helpers(profile,path)  # required scorer preflight, no fixed commit checks
        model=Path(MODEL_PATH)
        weights=sorted(model.glob('*.safetensors'))
        if not weights or not (model/'config.json').is_file():
            raise FileNotFoundError('checkpoint config/weights missing')
        files=[model/name for name in ('config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json') if (model/name).is_file()]+weights
        hashes={p.name:smoke.file_hash(p) for p in files}
        hidden_size=json.loads((model/'config.json').read_text())['hidden_size']
        if type(hidden_size) is not int or hidden_size<=0:
            raise ValueError('positive hidden_size required')
        identity=smoke.sha256_bytes(json.dumps(hashes,sort_keys=True).encode())
        context=dict(model_identity=identity,model_file_sha256=hashes,hidden_size=hidden_size,
            scoring_paths={'gsm8k_aug':str(low),'dapo_math':str(high)},runtime_provenance=provenance,
            high_scoring_sha256=smoke.file_hash(high),capture_hook_sha256=smoke.sha256_bytes(CAPTURE_SOURCE.encode()),
            total_budget_seconds=budget,shutdown_margin_seconds=SHUTDOWN_MARGIN_SECONDS,
            started_unix_time=time.time(),input_data_root=str(args.data_root))
        atomic_json(session/'input.json',context)
        store=ProbeStore(root,root/'candidates',identity)
        if all(not store.pending(c) for c in store.mapping.values()):
            store.publish()
            print(f'PROBING COMPLETE (already collected): {root}/selected',flush=True)
            return 0
        # Preparation and all Engine workers stay in this parent's supervised process group.
        sys.path.insert(0,str(overlay))
        os.environ['PYTHONPATH']=str(overlay)+os.pathsep+str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')
        run_worker(root,session,began+budget-SHUTDOWN_MARGIN_SECONDS)
        return 0
    except Exception as error:
        if store is not None:
            try:
                store=ProbeStore(root,root/'candidates',store.model_identity)
                store.snapshot('failed')
            except Exception:
                pass  # original runtime error remains the STOP reason
        root.mkdir(parents=True,exist_ok=True)
        atomic_json(root/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error)))
        print(f'STOP: {type(error).__name__}: {error}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
