"""Bounded, resumable single-call screening. SQLite keeps full-corpus memory bounded."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import time
import zlib

from client import Client, digest, save, public_config
from artifact_io import open_jsonl


def records(path):
    with open_jsonl(path) as file:
        for line in file:
            if line.strip():
                yield json.loads(line)
from glasseek_contract import build_contract
from safety_rules import Scanner
import trajectory_rubric as rubric

HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def pack(value):
    return zlib.compress(json.dumps(value, ensure_ascii=False).encode(), 1)


def unpack(value):
    return json.loads(zlib.decompress(value))


def fingerprint():
    # Include prompts and all rule/protocol dependencies, not only the rubric module.
    paths = sorted(HERE.glob('*.py')) + sorted((HERE/'prompts').glob('*')) + [HERE/'safety_policy.json']
    paths += sorted((HERE.parents[1]/'Shared/glasseek_shared').rglob('*.py'))
    return digest([(str(p.relative_to(HERE.parents[1])), hashlib.sha256(p.read_bytes()).hexdigest())
                   for p in paths if not p.name.startswith('test_')])


def connect(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS inputs (seq INTEGER PRIMARY KEY, id TEXT UNIQUE, sample BLOB, canonical BLOB NOT NULL, artifact TEXT, line INTEGER)')
    db.execute('CREATE TABLE IF NOT EXISTS results (id TEXT PRIMARY KEY, state TEXT NOT NULL, result TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
    return db


def prepare(db, canonical_path, input_path=None):
    if db.execute("SELECT value FROM meta WHERE key='prepared'").fetchone():
        return
    if db.execute('SELECT count(*) FROM inputs').fetchone()[0]:
        raise ValueError('Incomplete preparation; use a fresh output directory (no model calls have been made)')
    seen = Counter()
    for i,c in enumerate(records(canonical_path)):
        if input_path is None and c.get('safety',{}).get('state') != 'pass_rules':
            raise ValueError('Full-corpus input must be the existing safety-passed export')
        db.execute('INSERT INTO inputs VALUES (?,?,?,?,?,?)',
                   (i,c['id'],None,pack(c),c.get('source_artifact'),c.get('source_line_zero_based')))
        seen[c['source']] += 1
        if i % 1000 == 0:
            db.commit()
            print('prepare canonical',i+1,flush=True)
    db.commit()
    if input_path:
        for item in records(input_path):
            cursor=db.execute('UPDATE inputs SET sample=? WHERE id=? AND sample IS NULL',(pack(item),item['id']))
            if cursor.rowcount != 1:
                raise ValueError('Duplicate or unmatched pilot input')
    else:
        # Canonical exports retain exact source-artifact/line links. Read each compressed
        # source once, rather than loading the 126k full trajectories into RAM.
        for (artifact,) in db.execute('SELECT DISTINCT artifact FROM inputs').fetchall():
            if not artifact:
                raise ValueError('Missing normalized source artifact')
            wanted = {line:ident for ident,line in db.execute('SELECT id,line FROM inputs WHERE artifact=?',(artifact,))}
            for index,item in enumerate(records(artifact)):
                ident=wanted.get(index)
                if ident is None:
                    continue
                if item['id'] != ident:
                    raise ValueError('Source artifact linkage mismatch')
                db.execute('UPDATE inputs SET sample=? WHERE id=?',(pack(item),ident))
                if index % 1000 == 0:
                    db.commit()
            db.commit()
    if db.execute('SELECT count(*) FROM inputs WHERE sample IS NULL').fetchone()[0]:
        raise ValueError('Missing normalized samples')
    meta={'canonical_path':str(Path(canonical_path).resolve()),'input_path':str(Path(input_path).resolve()) if input_path else None,
          'counts':dict(seen),'records':sum(seen.values()),'prepared_at':now()}
    db.execute("INSERT INTO meta VALUES ('prepared',?)",(json.dumps(meta),))
    db.commit()


def validate_config(config):
    if config.get('local_only') is not True:
        raise ValueError('This batch only permits local Qwen 9B')
    from urllib.parse import urlparse
    for p in config['providers'].values():
        if (urlparse(p['url']).hostname not in ('127.0.0.1','localhost','::1')
                or p.get('trust_env') or p.get('model') != 'qwen3.5-9b-local'):
            raise ValueError('Only explicit local qwen3.5-9b-local endpoints are allowed')
        concurrency=p.get('max_concurrent_requests')
        if p.get('transport_attempts') != 1 or type(concurrency) is not int or not 1<=concurrency<=64:
            raise ValueError('Require one transport attempt and 1–64 requests per GPU')
    if len({p['max_concurrent_requests'] for p in config['providers'].values()})!=1:
        raise ValueError('Round-robin scheduling requires equal per-GPU concurrency')
    if config['generation'].get('attempts') != 1:
        raise ValueError('Each trajectory permits exactly one generation attempt')
    if config.get('stages') != {'whole_trajectory':{'provider':'default'}}:
        raise ValueError('Only the whole_trajectory stage is allowed')


def percentile(values, q):
    if not values:
        return None
    a=sorted(values); pos=(len(a)-1)*q; lo=int(pos); hi=min(lo+1,len(a)-1)
    return round(a[lo]+(a[hi]-a[lo])*(pos-lo),3)


def summarize(db, out, started, state):
    counts=Counter(); sources=defaultdict(Counter); ranked=[]; times=[]; gate_counts=Counter()
    for (blob,) in db.execute("SELECT result FROM results WHERE state='completed'"):
        r=json.loads(blob); key=r.get('decision') if r['status']=='routed' else r['status']
        counts[key]+=1; sources[r['source']][key]+=1
        if r.get('llm_eligible'):
            times.append(r['seconds'])
        gate_counts.update(r.get('failed_gates',[]))
        if r['status']=='rankable':
            entry={k:r[k] for k in ('id','source','score','points')}
            if 'difficulty' in r:entry['difficulty']=r['difficulty']
            ranked.append(entry)
    ranked.sort(key=lambda r:(-r['score'],r['id']))
    ranks=Counter()
    with (out/'ranking.jsonl').open('w') as f:
        for n,r in enumerate(ranked,1):
            ranks[r['source']]+=1
            f.write(json.dumps({**r,'rank':n,'dataset_rank':ranks[r['source']],'training_ready':False},ensure_ascii=False)+'\n')
    summary={'state':state,'records':db.execute('SELECT count(*) FROM inputs').fetchone()[0],
             'completed':sum(counts.values()),'outcomes':dict(counts),'by_dataset':dict(sources),
             'failed_gates':dict(gate_counts),'wall_seconds':round(time.monotonic()-started,3),
             'eligible_latency_seconds':{'p50':percentile(times,.5),'p95':percentile(times,.95),
                                        'mean':sum(times)/len(times) if times else None},
             'training_ready':0,'updated_at':now()}
    save(out/'summary.json',summary)
    return summary


def run(db, config, out):
    validate_config(config)
    policy_hash=fingerprint()
    binding=digest([policy_hash,public_config(config)])
    old=db.execute("SELECT value FROM meta WHERE key='run_binding'").fetchone()
    if old and old[0]!=binding:
        raise ValueError('Policy/config changed: use a new output directory')
    db.execute("INSERT OR IGNORE INTO meta VALUES ('run_binding',?)",(binding,))
    db.commit()
    # Never automatically resubmit a request whose outcome is unknown after a crash.
    for ident, in db.execute("SELECT id FROM results WHERE state='inflight'").fetchall():
        sample,canonical=db.execute('SELECT sample,canonical FROM inputs WHERE id=?',(ident,)).fetchone()
        item=unpack(sample)
        r={'id':ident,'source':item['source'],'status':'annotation_error',
           'error':'interrupted_outcome_unknown_not_retried','score':None,'training_ready':False,
           'seconds':0,'llm_eligible':False,'sample_hash':digest(item)}
        db.execute("UPDATE results SET state='completed',result=? WHERE id=?",(json.dumps(r),ident))
    db.commit()
    clients=[]
    for name,p in config['providers'].items():
        local={**config,'providers':{'default':p}}
        client=Client(local,out/name,policy_hash)
        # Confirm advertised identity before any generation; no discovery/fallback.
        models=client.request(p,'GET','/models')
        if p['model'] not in [r['id'] for r in models.get('data',[])]:
            raise ValueError('Local 9B model unavailable')
        clients.append(client)
    scanner=Scanner(json.loads((HERE/'safety_policy.json').read_text()),build_contract(),secrets.token_bytes(32))
    per_gpu=next(iter(config['providers'].values()))['max_concurrent_requests']
    workers=per_gpu*len(clients)
    started=time.monotonic(); pending=set(); last=0
    save(out/'run.json',{'started_at':now(),'policy_hash':policy_hash,'config':public_config(config),
                         'producer_per_gpu':per_gpu,'producer_total':workers,'single_call':True})
    def work(seq,item,canonical):
        start=time.monotonic()
        try:
            result=rubric.screen(item,canonical,scanner,clients[seq%len(clients)],numeric=config.get('scoring_mode')=='numeric')
        except Exception as exc:
            result={'id':item['id'],'source':item['source'],'status':'annotation_error',
                    'score':None,'error':type(exc).__name__+': '+str(exc),'training_ready':False}
        result.update(seconds=round(time.monotonic()-start,3),policy_hash=policy_hash,
                      sample_hash=digest(item),canonical_hash=digest(canonical),
                      llm_eligible=result.get('eligibility',{}).get('status')=='eligible')
        return result
    cursor=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        exhausted=False
        while pending or not exhausted:
            while not exhausted and len(pending)<2*workers:
                batch=db.execute('SELECT seq,id,sample,canonical FROM inputs WHERE seq>=? ORDER BY seq LIMIT 1',(cursor,)).fetchone()
                if batch is None:
                    exhausted=True; break
                seq,ident,sample,canonical=batch; cursor=seq+1
                if db.execute('SELECT 1 FROM results WHERE id=?',(ident,)).fetchone():
                    continue
                db.execute("INSERT INTO results VALUES (?,'inflight',NULL)",(ident,)); db.commit()
                pending.add(pool.submit(work,seq,unpack(sample),unpack(canonical)))
            if not pending:
                break
            done,pending=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:
                r=future.result()
                db.execute("UPDATE results SET state='completed',result=? WHERE id=?",(json.dumps(r,ensure_ascii=False),r['id']))
                db.commit()
                with (out/'results.jsonl').open('a') as f:
                    f.write(json.dumps(r,ensure_ascii=False)+'\n')
            if time.monotonic()-last>=30:
                print(json.dumps(summarize(db,out,started,'running'),ensure_ascii=False),flush=True)
                last=time.monotonic()
    summary=summarize(db,out,started,'completed')
    save(out/'completion.json',{'finished_at':now(),'policy_hash':policy_hash,**summary})
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--canonical',type=Path)
    ap.add_argument('--input',type=Path)
    ap.add_argument('--config',type=Path)
    ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        db=connect(args.output/'batch.sqlite')
        if args.canonical:
            prepare(db,args.canonical,args.input)
        elif not db.execute("SELECT 1 FROM meta WHERE key='prepared'").fetchone():
            raise ValueError('Prepare inputs first')
        if not args.prepare_only:
            if not args.config:
                raise ValueError('--config required')
            print(json.dumps(run(db,json.loads(args.config.read_text()),args.output),ensure_ascii=False))
        db.close()


if __name__=='__main__':
    main()
