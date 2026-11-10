"""Bound global inference concurrency; grade completed trajectories during inference."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from report import load,publish_scores,write_report
from suite import ROOT,sha


def execute(args,plans,blocked,names,counts):
    from run import prepare_score
    root=args.output_root/args.run_id
    states={name:{'assigned':counts[name],'state':'pending'} for name in names}
    for r in blocked:states[r['benchmark']].update(state='blocked',reason=r['reason'])
    from trajectory_scoring import TrajectoryScorer
    schedule = getattr(args, 'score_schedule', 'trajectory')
    incremental = TrajectoryScorer(args, {name:env for name,_,env in plans}) if schedule == 'trajectory' else None
    queue=list(plans);running={};judging={};slots=args.workers;last=0
    affinity = None
    if getattr(args, 'affinity_endpoints', None):
        from affinity import AffinityScheduler
        affinity = AffinityScheduler(args.affinity_endpoints.split(','), plans, counts, root, args.workers)
        queue = []
    def score_one(name,env):
        start=time.monotonic();state=states[name];packet=root/'score-inputs'/name
        pred=args.output_root/'predictions'/(args.run_id+'_'+name)/('suite_'+name)
        try:
            if packet.exists():
                m=load(packet/'manifest.json',{})
                if not m or any(not Path(p).exists() or sha(p)!=h for p,h in m.get('prediction_inputs',{}).items()):
                    packet.rename(packet.with_name(name+'.previous-'+str(time.time_ns())))
            if not packet.exists():prepare_score(args.suite,packet,pred,name,args.corpus_config)
            score_parent=root/'scores'/name;attempt=len(list(score_parent.glob('attempt-*')))+1
            out=score_parent/f'attempt-{attempt:03d}';state.update(state='scoring',score_dir=str(out))
            cmd=[str(args.score_python) if args.score_python else os.environ.get('SUITE_SCORE_PYTHON',str(args.suite.parent/'score-venv/bin/python')),
                str(Path(__file__).with_name('score.py')),'--suite',str(args.suite),'--packet',str(packet),'--out',str(out)]
            if args.score_config:cmd+=['--config',str(args.score_config)]
            with (root/(name+'.score.log')).open('a') as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
            receipt=load(out/'receipt.json',{})
            state['state']='scored' if result.returncode==0 else 'score_error'
            if result.returncode==0 and str(receipt.get('status','')).startswith('native_evaluation_partial'):
                state['state']='partially_scored'
            state['score_status']=receipt.get('status')
            if result.returncode:state['reason']=receipt.get('error_type','native scorer failed; see score log')
            if args.viewer:publish_scores(args,name,out,env)
        except Exception as exc:state.update(state='score_error',reason=type(exc).__name__+': '+str(exc)[:250])
        finally:state['score_seconds']=time.monotonic()-start
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        try:
            while queue or running or judging or (affinity and affinity.pending):
                if incremental:incremental.tick()
                if affinity:
                    queue.extend(affinity.ready())
                while queue and (affinity or slots>0):
                    name,cmd,env=queue.pop(0);budget=args.workers if affinity else min(counts[name],slots,args.workers)
                    if not affinity:slots-=budget
                    cmd=list(cmd);cmd[cmd.index('--workers-per-endpoint')+1]=str(budget)
                    for flag in ('--num-workers','--ramp-start','--ramp-max'):
                        if flag in cmd:cmd[cmd.index(flag)+1]=str(budget)
                    log=(root/(name+'.infer.log')).open('a')
                    proc=subprocess.Popen(cmd,cwd=ROOT/'Evaluation',env=env,stdout=log,stderr=subprocess.STDOUT)
                    running[name]=(proc,budget,time.monotonic(),log,env);states[name].update(state='inferring',pid=proc.pid,inference_workers=budget)
                    print(json.dumps({'benchmark':name,'stage':'infer','workers':budget,'pid':proc.pid}),flush=True)
                for name,(proc,budget,start,log,env) in list(running.items()):
                    code=proc.poll()
                    if code is None:continue
                    log.close()
                    if affinity:affinity.finished(name)
                    else:slots+=budget
                    del running[name];states[name].update(infer_seconds=time.monotonic()-start,infer_exit=code)
                    if args.phase=='all' and schedule == 'dataset':
                        pred=args.output_root/'predictions'/(args.run_id+'_'+name)/('suite_'+name)
                        if any(load(p,{}).get('status')=='success' for p in pred.glob('conv/*/*.json')):
                            states[name]['state']='awaiting_score';judging[pool.submit(score_one,name,env)]=name
                        else:states[name].update(state='inference_error',reason='no completed predictions; no judge request made')
                    else:states[name]['state']='inferred' if code==0 else 'inference_error'
                for future,name in list(judging.items()):
                    if future.done():future.result();del judging[future];print(json.dumps({'benchmark':name,'stage':states[name]['state']}),flush=True)
                if time.monotonic()-last>=15:write_report(args,states);last=time.monotonic()
                if queue or running or judging or (affinity and affinity.pending):time.sleep(1)
            if incremental:incremental.drain()
        finally:
            if incremental:incremental.close()
            for proc,_,_,log,_ in running.values():
                if proc.poll() is None:proc.terminate()
                log.close()
            write_report(args,states)
    print('Report: '+str(root/'report.json'),flush=True)
