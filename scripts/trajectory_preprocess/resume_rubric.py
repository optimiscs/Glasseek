"""Dataset-ordered continuation of a paused batch, without replaying anchors."""
import argparse,json,sqlite3,fcntl,os
from client import digest,public_config
from collections import Counter, defaultdict
import heapq
import sys,time,tempfile
from contextlib import nullcontext
from pathlib import Path
from rubric_batch import connect,unpack,run,now,fingerprint

def pool_signature(path):
    """Persistent logical revision: checkpoints and filesystem metadata are irrelevant."""
    import uuid
    from contextlib import closing
    with closing(sqlite3.connect(str(path),timeout=30)) as db:
        ready=db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='_sample_index_revision'").fetchone()
        if not ready:
            with db:
                db.execute('CREATE TABLE IF NOT EXISTS _sample_index_revision (identity TEXT PRIMARY KEY, revision INTEGER NOT NULL)')
                if not db.execute('SELECT 1 FROM _sample_index_revision').fetchone():
                    db.execute('INSERT INTO _sample_index_revision VALUES (?,0)',(str(uuid.uuid4()),))
                for event,suffix in [('INSERT','insert'),('DELETE','delete'),('UPDATE OF id,seq,sample','update')]:
                    db.execute(f'CREATE TRIGGER IF NOT EXISTS _sample_index_{suffix} AFTER {event} ON inputs BEGIN UPDATE _sample_index_revision SET revision=revision+1; END')
        identity,revision=db.execute('SELECT identity,revision FROM _sample_index_revision').fetchone()
        return json.dumps(['logical-inputs-v1',identity,revision])


def pool_index(path):
    """Atomic, disposable metadata cache; source trajectories remain untouched."""
    cache=path.with_name('sample-index.sqlite')
    print('Checking sample index: '+str(cache),file=sys.stderr,flush=True)
    with cache.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        signature=pool_signature(path)
        if cache.exists():
            try:
                with sqlite3.connect('file:'+str(cache)+'?mode=ro',uri=True) as db:
                    if db.execute('SELECT signature FROM metadata').fetchone()==(signature,):
                        rows=db.execute('SELECT dataset,seq,id FROM samples ORDER BY dataset,seq').fetchall()
                        print(f'Index hit: {len(rows):,} records; no trajectory decompression.',file=sys.stderr,flush=True)
                        return rows
            except sqlite3.DatabaseError:pass
        fd,tmp=tempfile.mkstemp(prefix='sample-index-',suffix='.tmp',dir=path.parent)
        os.close(fd)
        try:
            with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as source, sqlite3.connect(tmp) as index:
                version=source.execute('PRAGMA data_version').fetchone()
                source.execute('BEGIN')
                total=source.execute('SELECT count(*) FROM inputs').fetchone()[0]
                index.execute('CREATE TABLE samples (id TEXT PRIMARY KEY,dataset TEXT NOT NULL,seq INTEGER UNIQUE)')
                index.execute('CREATE TABLE metadata (signature TEXT NOT NULL)')
                started=last=time.monotonic()
                print(f'Building index: 0/{total:,}; index missing or input revision changed; scanning compressed trajectories.',file=sys.stderr,flush=True)
                rows=[]
                for done,(seq,ident,blob) in enumerate(source.execute('SELECT seq,id,sample FROM inputs ORDER BY seq'),1):
                    dataset=unpack(blob)['source']
                    rows.append((dataset,seq,ident))
                    index.execute('INSERT INTO samples VALUES (?,?,?)',(ident,dataset,seq))
                    current=time.monotonic()
                    if current-last>=5 or done==total:
                        elapsed=current-started
                        print(f'Building index: {done:,}/{total:,} ({done/max(total,1):.1%}), {elapsed:.1f}s',file=sys.stderr,flush=True)
                        last=current
                source.rollback()
                signature=pool_signature(path)
                if source.execute('PRAGMA data_version').fetchone()!=version:
                    raise RuntimeError('Source pool changed during indexing; retry')
                index.execute('INSERT INTO metadata VALUES (?)',(signature,))
                index.commit()
            os.replace(tmp,cache)
            return sorted(rows)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)


def balanced_selection(candidates, completed, limit):
    """Water-fill cumulative dataset counts; scarce pools redistribute their quota."""
    groups=defaultdict(list)
    for row in sorted(candidates):groups[row[0]].append(row)
    heap=[(completed.get(name,0),name) for name in groups]
    heapq.heapify(heap)
    taken=Counter()
    selected=[]
    while heap and len(selected)<limit:
        count,name=heapq.heappop(heap)
        selected.append(groups[name][taken[name]])
        taken[name]+=1
        if taken[name]<len(groups[name]):heapq.heappush(heap,(count+1,name))
    return sorted(selected)

def extend_pool(args):
    import signal,time
    output=args.output.resolve()
    if not (output/'batch.sqlite').exists():raise ValueError('Existing output database required')
    # Explicitly discard the paused producer and its uncertain requests.
    procpath=output/'process.json'
    if procpath.exists():
        pid=json.loads(procpath.read_text())['pid']
        proc=Path('/proc')/str(pid)
        if proc.exists():
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if 'resume_rubric.py' in cmd and output.name in cmd:
                if 'T (stopped)' not in (proc/'status').read_text():
                    raise ValueError('Producer is running; pause it before appending')
                if not args.dry_run:
                    os.kill(pid,signal.SIGKILL)
                    time.sleep(.2)
    with (nullcontext() if args.dry_run else (output/'run.lock').open('w')) as lock:
        if lock is not None:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        db=(sqlite3.connect('file:'+str(output/'batch.sqlite')+'?mode=ro',uri=True)
            if args.dry_run else connect(output/'batch.sqlite'))
        config=json.loads((output/'config.json').read_text())
        if config.get('scoring_mode')!='numeric' or config['generation']['seed']!=42 or config['generation']['temperature']!=0:
            raise ValueError('Expected existing numeric seed=42 temperature=0 config')
        source=sqlite3.connect('file:'+str(args.full_pool.resolve()/'batch.sqlite')+'?mode=ro',uri=True)
        indexed=pool_index(args.full_pool.resolve()/'batch.sqlite')
        datasets={ident:dataset for dataset,seq,ident in indexed}
        excluded=set()
        completed=Counter()
        discarded=[]
        for ident,state,result_json in db.execute(
                'SELECT i.id,r.state,r.result FROM inputs i JOIN results r ON i.id=r.id'):
            excluded.add(ident)
            if state=='inflight':
                if args.dry_run:
                    discarded.append({'id':ident})
                    continue
                item=unpack(db.execute('SELECT sample FROM inputs WHERE id=?',(ident,)).fetchone()[0])
                discarded.append({'id':ident,'source':item['source'],'status':'routed',
                    'decision':'interrupted_discarded','score':None,'training_ready':False,
                    'seconds':0,'sample_hash':digest(item)})
            elif json.loads(result_json).get('decision')!='interrupted_discarded':
                completed[json.loads(result_json).get('source') or datasets[ident]]+=1
        reused=output/'reused-128.jsonl'
        if reused.exists():
            for line in reused.open():
                if not line.strip():continue
                row=json.loads(line)
                if row['id'] not in excluded:completed[row['source']]+=1
                excluded.add(row['id'])
        candidates=[row for row in indexed if row[2] not in excluded]
        existing_total=sum(completed.values())
        remaining=max(0,args.limit-existing_total)
        selected=balanced_selection(candidates,completed,remaining)
        allocation=Counter(row[0] for row in selected)
        plan={'limit':args.limit,'limit_mode':'cumulative','existing_total':existing_total,
              'remaining_to_target':remaining,'scheduled':len(selected),
              'projected_total':existing_total+len(selected),'discarded_inflight':len(discarded),
              'allocation':{name:{'existing':completed[name],'scheduled':allocation[name],
                                  'cumulative':completed[name]+allocation[name]}
                            for name in sorted(set(completed)|set(allocation))}}
        print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
        if args.dry_run:return
        # Only never-requested inputs may be reselected. Results are immutable.
        queued=db.execute('SELECT count(*) FROM inputs WHERE id NOT IN (SELECT id FROM results)').fetchone()[0]
        db.execute('DELETE FROM inputs WHERE id NOT IN (SELECT id FROM results)')
        for result in discarded:
            db.execute("UPDATE results SET state='completed',result=? WHERE id=?",
                       (json.dumps(result,ensure_ascii=False),result['id']))
        offset=db.execute('SELECT coalesce(max(seq),-1)+1 FROM inputs').fetchone()[0]
        for index,(_,seq,ident) in enumerate(selected,offset):
            row=source.execute('SELECT sample,canonical,artifact,line FROM inputs WHERE id=?',(ident,)).fetchone()
            db.execute('INSERT INTO inputs VALUES (?,?,?,?,?,?)',(index,ident,*row))
        with (output/'results.jsonl').open('a') as f:
            for row in discarded:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        (output/'allocation.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
        old=db.execute("SELECT value FROM meta WHERE key='run_binding'").fetchone()
        new=digest([fingerprint(),public_config(config)])
        # Scheduler changed; retain explicit lineage rather than rewriting old result hashes.
        with (output/'invocations.jsonl').open('a') as f:
            f.write(json.dumps({'at':now(),'operation':'extend_full_pool','pool':str(args.full_pool),
                'limit':args.limit,'limit_mode':'cumulative','existing_total':existing_total,
                'scheduled':len(selected),'reselected_queued':queued,'allocation':plan['allocation'],'discarded_inflight':len(discarded),
                'previous_binding':old[0] if old else None,'new_binding':new,
                'note':'Existing rows/hashes unchanged; only scheduler scope expanded.'})+'\n')
        db.execute("INSERT OR REPLACE INTO meta VALUES ('run_binding',?)",(new,))
        db.commit()
        procpath.write_text(json.dumps({'pid':os.getpid(),'scope':'full_filtered_pool','append':True}))
        if selected:run(db,config,output)
        else:print('No remaining records to schedule; cumulative target reached or pool exhausted.',flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path)
    ap.add_argument('--anchors',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--limit',type=int,required=True,help='Cumulative result target, including reused anchors and rule routes; excludes discarded inflight records')
    ap.add_argument('--numeric-reference',type=Path,help='Use decimal scoring and reuse this completed 128-record numeric batch')
    ap.add_argument('--full-pool',type=Path,help='Expand existing output to the full prepared pool; discard paused inflight records')
    ap.add_argument('--dry-run',action='store_true',help='Preview full-pool allocation without changing records or processes')
    args=ap.parse_args()
    if args.dry_run and not args.full_pool:ap.error('--dry-run requires --full-pool')
    if args.limit<1:raise ValueError('Positive limit required')
    if args.full_pool:return extend_pool(args)
    if args.source is None or args.anchors is None:ap.error('--source and --anchors required without --full-pool')
    if args.limit<1 or args.output.resolve()==args.source.resolve():
        raise ValueError('Use positive limit and a separate continuation directory')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proc=json.loads((args.source/'process.json').read_text())
        status=Path('/proc')/str(proc['pid'])/'status'
        if status.exists() and 'T (stopped)' not in status.read_text():
            raise ValueError('Original producer must remain paused or exited')
        source=sqlite3.connect('file:'+str(args.source.resolve()/'batch.sqlite')+'?mode=ro',uri=True)
        target=connect(args.output/'batch.sqlite')
        anchors={json.loads(l)['id'] for l in args.anchors.open() if l.strip()}
        # Inflight outcomes are uncertain: never automatically duplicate requests.
        if args.numeric_reference:
            for procdir in Path('/proc').glob('[0-9]*'):
                if int(procdir.name)==os.getpid():continue
                try:
                    cmd=(procdir/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                    state=(procdir/'status').read_text()
                except (OSError,UnicodeError):continue
                if ('scripts/trajectory_preprocess/resume_rubric.py' in cmd or 'scripts/trajectory_preprocess/rubric_batch.py' in cmd) and 'T (stopped)' not in state and 'python' in cmd.split(' ')[0]:
                    raise ValueError('Another producer is active: '+procdir.name+'; pause it before numerical scoring')
            reference=args.numeric_reference
            config=json.loads((reference/'config.json').read_text())
            assert config.get('scoring_mode')=='numeric'
            assert config['generation']['temperature']==0 and config['generation']['seed']==42
            assert all(p['extra_body_by_host']['127.0.0.1']['seed']==42 for p in config['providers'].values())
            config['record_first_prompts']=False
            refdb=sqlite3.connect('file:'+str(reference.resolve()/'batch.sqlite')+'?mode=ro',uri=True)
            reused={i:json.loads(r) for i,r in refdb.execute("SELECT id,result FROM results WHERE state='completed'")}
            if set(reused)!=anchors:raise ValueError('Numeric reference must contain exactly the original 128 IDs')
            for ident,r in reused.items():
                row=source.execute('SELECT sample FROM inputs WHERE id=?',(ident,)).fetchone()
                if row is None or digest(unpack(row[0]))!=r['sample_hash']:raise ValueError('Reference input mismatch')
            (args.output/'reused-128.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reused.values()))
            excluded=anchors
        else:
            config=json.loads((args.source/'config.json').read_text())
            excluded=anchors | {r[0] for r in source.execute('SELECT id FROM results')}
        excluded.update(r[0] for r in target.execute('SELECT id FROM inputs'))
        candidates=[]
        for seq,ident,blob in source.execute('SELECT seq,id,sample FROM inputs'):
            if ident not in excluded:candidates.append((unpack(blob)['source'],seq,ident))
        candidates.sort()
        # Existing inputs reserve quota, including queued rows from earlier invocations.
        counted=set(anchors)
        databases=[target] if args.numeric_reference else [source,target]
        for database in databases:
            counted.update(ident for ident,blob in database.execute(
                "SELECT id,result FROM results WHERE state='completed'")
                if json.loads(blob).get('decision')!='interrupted_discarded')
        queued_ids={ident for ident, in target.execute(
            'SELECT id FROM inputs WHERE id NOT IN (SELECT id FROM results)')}
        remaining=max(0,args.limit-len(counted))
        # Trim only never-requested rows when the target is reduced.
        keep_queued={ident for ident, in target.execute(
            'SELECT id FROM inputs WHERE id NOT IN (SELECT id FROM results) ORDER BY seq LIMIT ?',
            (remaining,))}
        for ident in queued_ids-keep_queued:target.execute('DELETE FROM inputs WHERE id=?',(ident,))
        selected=candidates[:max(0,remaining-len(keep_queued))]
        offset=target.execute('SELECT coalesce(max(seq),-1)+1 FROM inputs').fetchone()[0]
        for index,(_,seq,ident) in enumerate(selected,offset):
            row=source.execute('SELECT sample,canonical,artifact,line FROM inputs WHERE id=?',(ident,)).fetchone()
            target.execute('INSERT INTO inputs VALUES (?,?,?,?,?,?)',(index,ident,*row))
        target.execute("INSERT OR IGNORE INTO meta VALUES ('prepared',?)",(json.dumps({'source':str(args.source.resolve()),'anchors':str(args.anchors.resolve())}),))
        target.commit()
        (args.output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
        with (args.output/'invocations.jsonl').open('a') as f:
            f.write(json.dumps({'at':now(),'limit':args.limit,'new_records':len(selected),
                'excluded_anchors':len(anchors),'excluded_source_results':source.execute('SELECT count(*) FROM results').fetchone()[0],
                'dataset_order':sorted({x[0] for x in selected}),
                'scoring_mode':config.get('scoring_mode','categorical'),'note':'Numeric mode skips only matching numeric anchors and local results; legacy scores are not reused.'})+'\n')
        if selected or keep_queued:run(target,config,args.output)
        else:print('No new eligible records to schedule')
if __name__=='__main__':main()
