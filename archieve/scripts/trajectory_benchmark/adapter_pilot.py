"""Reproducible offline adapter pilot against a completed, content-bound baseline.

Sampling is by source/file. A sampled SearchSwarm bundle retains all children.
The baseline's global exact-duplicate decisions are fixed in both arms, so a
smaller dedup search space cannot masquerade as an adapter improvement.
"""
import argparse
import collections
import copy
import concurrent.futures
import csv
import datetime
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'Shared'),str(ROOT/'scripts/trajectory_preprocess')]


def read(path):return json.loads(path.read_text())
def save(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def rows(path):
    with path.open() as f:
        for line in f:yield json.loads(line)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def search_batch_schema_upgrade(previous,current):
    """Allow only the reviewed single-query -> bounded query-array extension."""
    old=previous['web_search']['parameters']['properties']['query']
    new=current['web_search']['parameters']['properties']['query']
    expected_query={'oneOf':[{'type':'string','minLength':1,'pattern':'\\S'},
        {'type':'array','items':{'type':'string','minLength':1,'pattern':'\\S'},'minItems':1,'maxItems':5}]}
    if old!={'type':'string'} or new!=expected_query:return False
    expected=copy.deepcopy(previous)
    expected['web_search']['parameters']['properties']['query']=expected_query
    return expected==current
def csvout(path,data,fields=None):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(data[0]));writer.writeheader();writer.writerows(data)


def select_sample(baseline,fraction,seed):
    import pyarrow.parquet as pq
    manifest=read(baseline/'intake/manifest.json');files=manifest['inputs']
    swarm=read(baseline/'searchswarm/manifest.json')
    specs=[dict(s) for s in files]+[{'source':'SearchSwarm','path':swarm['source']}]
    bysource=collections.defaultdict(list)
    for spec in specs:
        p=Path(spec['path']);assert spec['source']!='OpenResearcher'
        if p.suffix=='.parquet':n=pq.ParquetFile(p).metadata.num_rows
        else:
            with p.open('rb') as f:n=sum(b.count(b'\n') for b in iter(lambda:f.read(8*1024**2),b''))
        bysource[spec['source']].append((str(p),n))
    indices={};strata=[]
    for source,items in bysource.items():
        total=sum(n for _,n in items);target=math.ceil(total*fraction)
        quotas={p:int(n*target/total) for p,n in items}
        order=sorted(items,key=lambda x:(-(x[1]*target/total-quotas[x[0]]),x[0]))
        for p,_ in order[:target-sum(quotas.values())]:quotas[p]+=1
        for p,n in items:
            relative=str(Path(p).relative_to(ROOT))
            rank=lambda i:hashlib.sha256(f'{seed}|{relative}|{i}'.encode()).digest()
            selected=sorted(heapq.nsmallest(quotas[p],range(n),key=rank));indices[p]=selected
            strata.append({'source':source,'path':p,'full_rows':n,'sample_rows':len(selected)})
    return {'version':'source-file-hash-sample-v1','fraction':fraction,'seed':seed,
            'searchswarm_unit':'bundle including all archived children','strata':strata,'indices_by_path':indices}


from reporting.pilot import report, baseline_decisions


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--fraction',type=float,default=.05);ap.add_argument('--seed',type=int,default=20261108);ap.add_argument('--workers',type=int,default=16)
    ap.add_argument('--allow-search-batch-schema-upgrade',action='store_true')
    args=ap.parse_args();out=args.output.resolve();baseline=args.baseline.resolve();assert 0<args.fraction<=1
    if out.exists():raise RuntimeError('Use a new output directory')
    os.umask(0o077);out.mkdir(parents=True);started=time.monotonic()
    identities=read(baseline/'input_file_hashes.json')
    def check_input(entry):
        path=Path(entry['path']);stat=path.stat()
        assert (stat.st_size,stat.st_mtime_ns)==(entry['bytes'],entry['mtime_ns'])
        assert sha(path)==entry['sha256'],'Baseline input changed: '+str(path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(check_input,identities['files']))
    save(out/'input_file_hashes.json',identities)
    sample=select_sample(baseline,args.fraction,args.seed);save(out/'sample_manifest.json',sample)
    if (baseline/'sample_manifest.json').exists():
        assert sample==read(baseline/'sample_manifest.json'),'Pilot baseline requires the identical sampled cohort'
    save(out/'inputs.json',[s for s in read(baseline/'inputs.json') if s['source']!='OpenResearcher'])
    from glasseek_contract import build_contract
    contract=build_contract();previous=read(baseline/'unified/tools.json')
    for key in ('code_policy','delegation','toolsets'):assert contract[key]==previous[key], 'Target capability changed: '+key
    upgrade=contract['schemas']!=previous['schemas']
    if upgrade:
        assert args.allow_search_batch_schema_upgrade and search_batch_schema_upgrade(previous['schemas'],contract['schemas']), 'Unapproved target schema change'
    versions={}
    for prefix in ('scripts/trajectory_preprocess','Shared/glasseek_shared'):
        for p in (ROOT/prefix).rglob('*.py'):versions[str(p.relative_to(ROOT))]=sha(p)
    for name in ('unify_trajectory_candidates.py','audit_full_prepare.py','adapter_pilot.py','sample_xml_audit.py'):
        p=ROOT/'scripts/trajectory_benchmark'/name;versions[str(p.relative_to(ROOT))]=sha(p)
    for name in ('Evaluation_backend/tools/web_tools.py','RL_training/glasseek_training/web_tools.py'):
        versions[name]=sha(ROOT/name)
    name='Shared/glasseek_shared/hermes/qwen_xml_protocol.py';versions[name]=sha(ROOT/name)
    versions['scripts/trajectory_preprocess/safety_policy.json']=sha(ROOT/'scripts/trajectory_preprocess/safety_policy.json')
    save(out/'source_hashes.json',versions)
    snapshot=out/'source_snapshot'
    for rel in versions:
        target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
    save(out/'manifest.json',{'status':'running','fraction':args.fraction,'seed':args.seed,'workers':args.workers,'baseline':str(baseline),'excluded_sources':['OpenResearcher'],'llm_calls':0,'live_tool_calls':0,'contract_sha256':contract['sha256'],'search_batch_schema_upgrade':upgrade,'source_hashes':versions})
    guard=out/'offline_guard';guard.mkdir();(guard/'sitecustomize.py').write_text('import sys,os\ndef audit(event,args):\n if event=="socket.getaddrinfo" or (event=="socket.connect" and getattr(args[0],"family",None) in (2,10)):\n  with open(os.environ["TRAJECTORY_NETWORK_AUDIT"],"a") as f:f.write(event+"\\n")\n  raise RuntimeError("Network disabled for offline curation")\nsys.addaudithook(audit)\n')
    env=dict(os.environ,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',ARROW_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env['PYTHONPATH']=os.pathsep.join([str(guard),str(ROOT/'Shared'),str(ROOT/'scripts/trajectory_preprocess'),str(ROOT/'scripts/trajectory_benchmark')])
    env['TRAJECTORY_NETWORK_AUDIT']=str(out/'network_attempts.log')
    for key in ('OPENAI_API_KEY','DEEPSEEK_API_KEY','ANTHROPIC_API_KEY','JUDGE_API_KEY'):env.pop(key,None)
    keyfile=Path('/root/.config/glasseek/safety/trajectory-glasseek-v7-safety-rules-v1.key');env['TRAJECTORY_SAFETY_HMAC_KEY']=keyfile.read_text().strip()
    def phase(name,command):
        folder=out/name;folder.mkdir(exist_ok=True);t=time.monotonic()
        save(out/f'phase-{name}.json',{'status':'running','command':command})
        with (folder/'run.log').open('w') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                if shutil.disk_usage(out).free<5*1024**3:proc.terminate();raise RuntimeError('Disk reserve reached')
                time.sleep(1)
        save(out/f'phase-{name}.json',{'status':'completed' if proc.returncode==0 else 'failed','returncode':proc.returncode,'elapsed_seconds':time.monotonic()-t,'command':command})
        if proc.returncode:raise RuntimeError('Stage failed: '+name)
        event=json.dumps({'phase':name,'elapsed_seconds':round(time.monotonic()-t,2)})
        with (out/'run.log').open('a') as log:log.write(event+'\n')
        print(event,flush=True)
    py=sys.executable;selection=['--sample-manifest',str(out/'sample_manifest.json')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(phase,'intake',[py,'-u','scripts/trajectory_preprocess/full_prepare.py','--output',str(out/'intake'),'--inputs',str(out/'inputs.json'),'--workers',str(args.workers),'--checkpoint-every','500',*selection]),
                 pool.submit(phase,'searchswarm',[py,'-u','scripts/trajectory_preprocess/migrate_searchswarm.py','--output',str(out/'searchswarm'),*selection])]
        for future in futures:future.result()
    phase('intake-audit',[py,'scripts/trajectory_benchmark/audit_full_prepare.py','--input',str(out/'intake')])
    phase('union-job',[py,'-u','scripts/trajectory_benchmark/unify_trajectory_candidates.py','--intake',str(out/'intake'),'--searchswarm',str(out/'searchswarm'),'--output',str(out/'unified')])
    phase('safety',[py,'-u','scripts/trajectory_preprocess/safety_pipeline.py','scan','--input',str(out/'unified/candidates.jsonl'),'--output',str(out/'safety'),'--workers',str(args.workers)])
    phase('safety-report',[py,'scripts/trajectory_preprocess/report_safety_scan.py','--run',str(out/'safety'),'--key-file',str(keyfile)])
    for rel,expected in versions.items():assert sha(ROOT/rel)==expected,'Source changed during pilot: '+rel
    attempts=out/'network_attempts.log';assert not attempts.exists() or not attempts.stat().st_size
    report(out,baseline)
    manifest=read(out/'manifest.json');manifest.update(status='completed',elapsed_seconds=time.monotonic()-started,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save(out/'manifest.json',manifest)


if __name__=='__main__':main()
