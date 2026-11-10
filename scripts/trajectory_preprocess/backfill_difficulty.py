"""Supplement existing numeric judgments; never overwrite the active source batch."""
import argparse,concurrent.futures as futures,fcntl,json,sqlite3,time
from pathlib import Path
from client import Client,digest,public_config
from rubric_batch import unpack,fingerprint,validate_config
from trajectory_rubric import difficulty_request,merge_difficulty
import policy


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--per-gpu',type=int,default=8)
    ap.add_argument('--limit',type=int,default=0,help='Cumulative selected rows; 0 means all eligible rows')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if args.source.resolve()==args.output.resolve():ap.error('Use a separate output directory')
    if not 1<=args.per_gpu<=64 or args.limit<0:ap.error('Invalid concurrency or limit')
    config=json.loads((args.source/'config.json').read_text())
    if config.get('scoring_mode')!='numeric':ap.error('Numeric source required')
    for p in config['providers'].values():p['max_concurrent_requests']=args.per_gpu
    config['execution']['workers']=args.per_gpu*len(config['providers'])
    validate_config(config)
    if config['generation'].get('temperature')!=0 or config['generation'].get('seed')!=42:ap.error('Require seed=42 and temperature=0')
    source=sqlite3.connect('file:'+str(args.source.resolve()/'batch.sqlite')+'?mode=ro',uri=True)
    candidates=[];counts={}
    for ident,blob in source.execute("SELECT id,result FROM results WHERE state='completed' ORDER BY id"):
        r=json.loads(blob)
        if r.get('difficulty') is not None:key='already_has_difficulty'
        elif not r.get('judgment') or 'numeric' not in r.get('rubric_version',''):key='no_numeric_judgment'
        elif r.get('eligibility',{}).get('status')!='eligible':key='not_eligible'
        else:key='eligible';candidates.append((ident,r))
        counts[key]=counts.get(key,0)+1
    if args.limit:candidates=candidates[:args.limit]
    print(json.dumps({'source_counts':counts,'selected':len(candidates),'per_gpu':args.per_gpu},ensure_ascii=False),flush=True)
    if args.dry_run:return
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        db=sqlite3.connect(args.output/'batch.sqlite')
        db.execute('CREATE TABLE IF NOT EXISTS results (id TEXT PRIMARY KEY,state TEXT,payload TEXT,source_hash TEXT)')
        binding=digest([fingerprint(),public_config(config),str(args.source.resolve())])
        manifest=args.output/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())['binding']!=binding:raise ValueError('Code/config/source changed; use another output')
        manifest.write_text(json.dumps({'binding':binding,'source':str(args.source.resolve()),'config':public_config(config)},indent=2))
        db.execute("UPDATE results SET state='error',payload=? WHERE state='inflight'",(json.dumps({'error':'interrupted_not_retried'}),));db.commit()
        clients=[Client({**config,'providers':{'default':p}},args.output/name,binding) for name,p in config['providers'].items()]
        def work(index,item,original):
            start=time.monotonic()
            try:
                prompt,schema,validator=difficulty_request(item)
                annotation=clients[index%len(clients)].call('whole_trajectory',item['id'],prompt,policy.SYSTEM,validator,schema=schema)
                result=merge_difficulty(item,original,annotation)
                result['difficulty_policy_hash']=binding
                return 'completed',result
            except Exception as exc:return 'error',{'id':item['id'],'error':str(exc),'seconds':time.monotonic()-start}
        with futures.ThreadPoolExecutor(max_workers=config['execution']['workers']) as pool:
            pending={}
            last_progress=time.monotonic()
            def flush():
                nonlocal last_progress
                done,_=futures.wait(pending,return_when=futures.FIRST_COMPLETED)
                for f in done:
                    ident=pending.pop(f);state,payload=f.result()
                    db.execute('UPDATE results SET state=?,payload=? WHERE id=?',(state,json.dumps(payload,ensure_ascii=False),ident));db.commit()
                if time.monotonic()-last_progress>=30:
                    print(json.dumps(dict(db.execute('SELECT state,count(*) FROM results GROUP BY state'))),flush=True)
                    last_progress=time.monotonic()
            for index,(ident,r) in enumerate(candidates):
                prior=db.execute('SELECT source_hash FROM results WHERE id=?',(ident,)).fetchone()
                if prior:
                    if prior[0]!=digest(r):raise ValueError('Source result changed: '+ident)
                    continue
                blob=source.execute('SELECT sample FROM inputs WHERE id=?',(ident,)).fetchone()
                if not blob:raise ValueError('Missing source input: '+ident)
                item=unpack(blob[0])
                if digest(item)!=r['sample_hash']:raise ValueError('Source sample mismatch: '+ident)
                db.execute('INSERT INTO results VALUES (?,?,?,?)',(ident,'inflight',None,digest(r)));db.commit()
                pending[pool.submit(work,index,item,r)]=ident
                if len(pending)>=config['execution']['workers']:flush()
            while pending:flush()
        # Rebuild export from durable records so restarting cannot duplicate lines.
        temp=args.output/'results.jsonl.tmp'
        with temp.open('w') as f:
            for payload, in db.execute("SELECT payload FROM results WHERE state='completed' ORDER BY id"):f.write(payload+'\n')
        temp.replace(args.output/'results.jsonl')
        print(dict(db.execute('SELECT state,count(*) FROM results GROUP BY state')),flush=True)

if __name__=='__main__':main()
