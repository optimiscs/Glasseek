"""Offline safety commands for a unified candidate release.

scan: rules only, resumable, no model/network calls.
review: explicitly enabled, bounded LLM review through the shared provider config.
apply-review: content/policy-bound human decisions; cannot override hard schema/code gates.
redact: sanitized preview only, never an automatically approved training sample.
export: safety-passed review candidates, including parent/child dependency checks.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import time

from client import digest, public_config, save
from safety_rules import Scanner
from safety_review import make_client, review_item

HERE=Path(__file__).resolve().parent
OUTPUTS=('decisions','findings','review_queue','quarantine')
PASS={'pass_rules','pass_reviewed','pass_human'}
WORKER=None


def setup_worker(policy,contract,key):
    global WORKER
    WORKER=Scanner(policy,contract,key)


def process_record(task):
    index,start,end,raw=task
    try:
        item=json.loads(raw)
        if not isinstance(item,dict): raise ValueError('record_not_object')
        result=WORKER.scan(item)
        from safety_context import build_context
        _,actions,_=build_context(item)
        result['delegation_calls']=sum(a['name']=='delegate_task' for a in actions)
        result['delegation_tasks']=sum(len(a['arguments'].get('tasks') or [a]) for a in actions if a['name']=='delegate_task')
        key=WORKER.fingerprint(str(item.get('id','row:'+str(index))))
    except Exception as exc:
        key=WORKER.fingerprint('row:'+str(index))
        result={'state':'review','policy_hash':WORKER.policy_hash,'content_hash':WORKER.fingerprint(raw),
                'coverage_complete':False,'training_ready':False,'error_kind':type(exc).__name__,
                'findings':[{'finding_id':'record-error','rule_id':'coverage.record_unparseable','category':'coverage','disposition':'review'}]}
    result.update(record_key=key,source_row=index,offset_start=start,offset_end=end)
    return result


@contextmanager
def locked(out):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'safety.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def load(args):
    policy=json.loads(args.policy.read_text())
    if args.command=='scan': contract_path=args.contract or args.input.parent/'tools.json'
    else: contract_path=args.output/'contract.json'
    contract=json.loads(contract_path.read_text())
    key=os.environ.get(policy['hmac_key_env'],'').encode()
    scanner=Scanner(policy,contract,key)
    return policy,contract,scanner


def identity(path):
    stat=path.stat()
    return {'path':str(path.resolve()),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def code_hashes():
    return {n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in
            ('safety_pipeline.py','safety_rules.py','safety_context.py','safety_review.py','client.py','tool_xml.py','glasseek_contract.py','answer_leakage.py')}


def connect(out):
    db=sqlite3.connect(out/'state.sqlite')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS checkpoint(id INTEGER PRIMARY KEY,state TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS decisions(row_index INTEGER PRIMARY KEY,record_key TEXT UNIQUE,result TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS review_history(sequence INTEGER PRIMARY KEY,event TEXT)')
    return db


def audit_review(db,result,kind,old_state):
    event={k:result[k] for k in ('record_key','source_row','content_hash','policy_hash','state')}
    event.update(kind=kind,previous_state=old_state,timestamp=time.time())
    for k in ('human_review','llm_reviews','llm_policy','review_config_hash','review_route','review_error_kind'):
        if k in result:event[k]=result[k]
    db.execute('INSERT INTO review_history(event) VALUES(?)',(json.dumps(event),))


def export_review_history(db,out):
    # SQLite is authoritative; regenerate a readable audit sidecar after a commit.
    path=out/'review_history.jsonl.tmp'
    with path.open('w') as f:
        for sequence,event in db.execute('SELECT sequence,event FROM review_history ORDER BY sequence'):
            f.write(json.dumps({'sequence':sequence,**json.loads(event)},ensure_ascii=False)+'\n')
    path.replace(out/'review_history.jsonl')


def scan(args,policy,contract,scanner):
    manifest={'version':'safety-run-v1','input':identity(args.input),'policy_hash':scanner.policy_hash,
              'key_id':scanner.fingerprint('key-id')[:16],'code_sha256':code_hashes(),'mode':'rules_only',
              'llm_calls':0,'record_byte_limit':args.max_record_bytes}
    out=args.output
    if scanner.scrub(manifest['input']['path'])!=manifest['input']['path']:
        raise ValueError('Use an input path without sensitive values')
    path=out/'manifest.json'
    if path.exists():
        if not args.resume: raise ValueError('Output exists; use --resume or a new directory')
        if json.loads(path.read_text())!=manifest: raise ValueError('Resume input, policy, key or code changed')
    else:
        save(path,manifest);save(out/'contract.json',contract);save(out/'policy.json',policy)
        frozen=out/'frozen_source';frozen.mkdir()
        for name in manifest['code_sha256']: (frozen/name).write_bytes((HERE/name).read_bytes())
    db=connect(out);prior=db.execute('SELECT state FROM checkpoint WHERE id=1').fetchone()
    state=json.loads(prior[0]) if prior else {'next_row':0,'offset':0,'counts':{},'offsets':{n:0 for n in OUTPUTS},'elapsed_seconds':0}
    started=time.monotonic();previous=state['elapsed_seconds'];handles={}
    for name in OUTPUTS:
        p=out/(name+'.jsonl');h=p.open('r+b' if p.exists() else 'w+b')
        if h.seek(0,2)<state['offsets'][name]: raise ValueError('Checkpoint output was truncated')
        h.truncate(state['offsets'][name]);h.seek(0,2);handles[name]=h
    def emit(name,row): handles[name].write((json.dumps(row,ensure_ascii=False)+'\n').encode())
    def checkpoint(status):
        for name,h in handles.items(): h.flush();os.fsync(h.fileno());state['offsets'][name]=h.tell()
        state['status']=status;state['elapsed_seconds']=previous+time.monotonic()-started
        db.execute('INSERT OR REPLACE INTO checkpoint VALUES(1,?)',(json.dumps(state),));db.commit()
        save(out/'progress.json',state)
    setup_worker(policy,contract,scanner.key)
    pool=ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                            initializer=setup_worker,initargs=(policy,contract,scanner.key)) if args.workers>1 else None
    try:
        with args.input.open('rb') as source:
            source.seek(state['offset']);next_row=state['next_row'];finished=False
            while not finished:
                batch=[];batch_bytes=0
                while len(batch)<max(1,args.workers*2) and batch_bytes<64*1024**2:
                    if args.limit is not None and next_row>=args.limit: break
                    start=source.tell();raw=source.readline(args.max_record_bytes+1)
                    if not raw: finished=True;break
                    if len(raw)>args.max_record_bytes:
                        # Consume the complete rejected row; never treat the suffix as another record.
                        while not raw.endswith(b'\n'):
                            raw=source.readline(args.max_record_bytes+1)
                            if not raw: break
                        text='[oversize record withheld]'
                    else:
                        try: text=raw.decode('utf-8')
                        except UnicodeDecodeError: text='[invalid UTF-8 record withheld]'
                    batch.append((next_row,start,source.tell(),text));batch_bytes+=len(raw);next_row+=1
                if not batch: break
                results=pool.map(process_record,batch) if pool else map(process_record,batch)
                for result in results:
                    db.execute('INSERT INTO decisions VALUES(?,?,?)',(result['source_row'],result['record_key'],json.dumps(result)))
                    emit('decisions',result)
                    for finding in result['findings']:
                        emit('findings',{'record_key':result['record_key'],'content_hash':result['content_hash'],'source_row':result['source_row'],**finding})
                    if result['state']!='pass_rules': emit('quarantine' if result['state']=='quarantine' else 'review_queue',result)
                    counts=state['counts'];counts['scanned']=counts.get('scanned',0)+1
                    counts[result['state']]=counts.get(result['state'],0)+1
                    for finding in result['findings']:
                        rule='finding:'+finding['rule_id'];counts[rule]=counts.get(rule,0)+1
                    state['next_row']=result['source_row']+1;state['offset']=result['offset_end']
                checkpoint('running')
                if state['next_row']%100==0: print(json.dumps({'rows':state['next_row'],'counts':state['counts']}),flush=True)
            if identity(args.input)!=manifest['input']: raise ValueError('Input changed during scan')
            checkpoint('completed' if state['offset']==manifest['input']['bytes'] else 'paused_limit')
            save(out/'summary.json',{**state,'llm_calls':0,'training_ready':False})
    finally:
        if pool: pool.shutdown(wait=True,cancel_futures=True)
        for h in handles.values():h.close()
        db.close()
    return state


def checked_run(args,scanner):
    manifest=json.loads((args.output/'manifest.json').read_text())
    if manifest['policy_hash']!=scanner.policy_hash or manifest['key_id']!=scanner.fingerprint('key-id')[:16]:
        raise ValueError('Policy or HMAC key differs from scan')
    if manifest['code_sha256']!=code_hashes(): raise ValueError('Code changed since scan; create a new safety run')
    source=Path(manifest['input']['path'])
    if identity(source)!=manifest['input']: raise ValueError('Input changed since scan')
    return source,connect(args.output)


def read_item(source,result):
    source.seek(result['offset_start']);raw=source.read(result['offset_end']-result['offset_start'])
    item=json.loads(raw)
    if digest(item)!=result['content_hash']: raise ValueError('Stale content fingerprint')
    return item


def review(args,policy,scanner):
    if not policy['llm']['enabled'] and not args.enable_llm: raise ValueError('Use --enable-llm to explicitly enable optional review')
    source,db=checked_run(args,scanner)
    config=json.loads(args.config.read_text());client=make_client(config,args.output/'llm',scanner)
    public=json.loads(scanner.scrub(json.dumps(public_config(config),ensure_ascii=False)))
    config_hash=digest(public)
    save(args.output/'llm/config.public.json',public)
    save(args.output/'llm/configs'/(config_hash+'.json'),public)
    reviewed=0;failed=0
    try:
        with source.open('rb') as handle:
            # Only IDs are loaded; text stays streaming.
            for (row_index,) in db.execute('SELECT row_index FROM decisions ORDER BY row_index').fetchall():
                result=json.loads(db.execute('SELECT result FROM decisions WHERE row_index=?',(row_index,)).fetchone()[0])
                if result['state']!='review' or result.get('llm_reviews'): continue
                if reviewed>=policy['llm']['max_records']: break
                if not result.get('coverage_complete'): continue
                if any(f['rule_id'] not in policy['release_labels'] for f in result['findings']): continue
                item=read_item(handle,result)
                result['review_config_hash']=config_hash
                old_state=result['state']
                try: result=review_item(item,result,scanner,client);reviewed+=1
                except Exception as exc:
                    result.update(review_route='model_failure',review_error_kind=type(exc).__name__);failed+=1
                db.execute('UPDATE decisions SET result=? WHERE row_index=?',(json.dumps(result),row_index))
                audit_review(db,result,'llm',old_state);db.commit()
                if failed: break  # Do not repeatedly spend attempts on unavailable providers.
        export_review_history(db,args.output)
    finally: db.close()
    save(args.output/'llm_review_summary.json',{'reviewed_records':reviewed,'failed':failed,'training_ready':False})


def apply_review(args,policy,scanner):
    _,db=checked_run(args,scanner);count=0
    review_input=args.review_file.open()
    try:
        for line in review_input:
            decision=json.loads(line)
            row=db.execute('SELECT row_index,result FROM decisions WHERE record_key=?',(decision['record_key'],)).fetchone()
            if row is None: raise ValueError('Unknown review record')
            index,text=row;result=json.loads(text)
            if any(decision.get(k)!=result[k] for k in ('content_hash','policy_hash')): raise ValueError('Stale human decision')
            if not decision.get('reviewer') or not decision.get('reason'): raise ValueError('Human review needs attribution and reason')
            action=decision['decision'];old_state=result['state']
            if action=='release':
                if not result.get('coverage_complete'): raise ValueError('Incomplete coverage cannot be approved')
                for f in result['findings']:
                    if f['category'] not in policy['human_overridable_categories'] or (f['category']=='url' and f['disposition']=='quarantine'):
                        raise ValueError('Hard schema/code/URL policy gate cannot be overridden')
                if set(decision.get('cleared_finding_ids',[]))!={f['finding_id'] for f in result['findings']}: raise ValueError('All findings need a decision')
                result['state']='pass_human'
            elif action in ('review','quarantine'): result['state']=action
            else: raise ValueError('Unknown human decision')
            result['human_review']={'reviewer':scanner.scrub(decision['reviewer']), 'reason':scanner.scrub(decision['reason']), 'decision':action}
            db.execute('UPDATE decisions SET result=? WHERE row_index=?',(json.dumps(result),index))
            audit_review(db,result,'human',old_state);count+=1
        db.commit()
        export_review_history(db,args.output)
    finally: review_input.close();db.close()
    save(args.output/'human_review_summary.json',{'applied':count})


def redact(args,scanner):
    source,db=checked_run(args,scanner);count=0
    target=args.output/'redacted_preview.jsonl'
    if target.exists(): raise ValueError('Preview exists; preserve it or use a separate safety run')
    try:
        with source.open('rb') as handle,target.open('w') as output:
            for (text,) in db.execute('SELECT result FROM decisions ORDER BY row_index'):
                result=json.loads(text)
                if not result.get('coverage_complete') or not any(f['category'] in ('credential','privacy') for f in result['findings']):continue
                if count>=args.limit:break
                item=read_item(handle,result);preview=scanner.redact_preview(item)
                output.write(json.dumps(preview,ensure_ascii=False)+'\n');count+=1
    finally:db.close()
    save(args.output/'redaction_summary.json',{'rows':count,'state':'redacted_pending_validation','training_ready':False})


def export(args,scanner):
    source,db=checked_run(args,scanner)
    state=json.loads(db.execute('SELECT state FROM checkpoint WHERE id=1').fetchone()[0])
    if state['status']!='completed':
        db.close();raise ValueError('Finish the scan before exporting')
    decisions={}
    for key,text in db.execute('SELECT record_key,result FROM decisions'):
        row=json.loads(text)
        decisions[key]={k:row.get(k) for k in ('state','delegation_calls','delegation_tasks')}
    dependencies=collections.defaultdict(set);linked=set();task_links=collections.defaultdict(set)
    links_digest=hashlib.sha256()
    links=args.links or source.parent/'delegation_links.jsonl'
    if links.exists():
        with links.open() as link_input:
            for line in link_input:
                links_digest.update(line.encode())
                edge=json.loads(line);parent=edge.get('canonical_parent_id');child=edge.get('canonical_child_id')
                if not parent:continue
                p=scanner.fingerprint(str(parent));c=scanner.fingerprint(str(child)) if child else None
                linked.add(p)
                if p in decisions:
                    dependencies[p].add(c)
                    task_links[p].add((edge.get('call_id',c),edge.get('task_index',0)))
    blocked={key for key,r in decisions.items() if r['state'] not in PASS}
    dependency_review=set()
    for key,r in decisions.items():
        if r.get('delegation_calls') and (key not in linked or len(task_links[key])<(r.get('delegation_tasks') or 1)):
            dependency_review.add(key)
    # Eliminate leaf nodes; the remainder contains cycles and their ancestors.
    remaining={p:len({c for c in cs if c in dependencies}) for p,cs in dependencies.items()}
    reverse=collections.defaultdict(set)
    for p,cs in dependencies.items():
        for c in cs:reverse[c].add(p)
    leaves=collections.deque(p for p,n in remaining.items() if n==0)
    while leaves:
        leaf=leaves.popleft();remaining.pop(leaf)
        for parent in reverse[leaf]:
            if parent in remaining:
                remaining[parent]-=1
                if remaining[parent]==0:leaves.append(parent)
    dependency_review.update(remaining)
    changed=True
    while changed:
        old=len(dependency_review)
        for parent,children in dependencies.items():
            if any(c is None or c not in decisions or c in blocked or c in dependency_review for c in children):dependency_review.add(parent)
        changed=len(dependency_review)!=old
    counts=collections.Counter();destination=args.output/'export'
    if destination.exists(): raise ValueError('Export already exists; do not overwrite an audited export')
    destination.mkdir()
    try:
        with source.open('rb') as handle,(destination/'candidates.jsonl.tmp').open('w') as output, (destination/'withheld.jsonl').open('w') as withheld:
            for key,text in db.execute('SELECT record_key,result FROM decisions ORDER BY row_index'):
                r=json.loads(text)
                outcome=r['state']
                if key in dependency_review and outcome in PASS:outcome='dependency_review'
                counts[outcome]+=1
                if outcome not in PASS:
                    withheld.write(json.dumps({'record_key':key,'source_row':r['source_row'],'state':outcome,'content_hash':r['content_hash']})+'\n');continue
                item=read_item(handle,r);item['training_ready']=False
                item['safety']={'state':outcome,'content_hash':r['content_hash'],'policy_hash':scanner.policy_hash,'record_key':key}
                output.write(json.dumps(item,ensure_ascii=False)+'\n')
        (destination/'candidates.jsonl.tmp').replace(destination/'candidates.jsonl')
        assert sum(counts.values())==state['next_row']
        save(destination/'summary.json',{'status':'completed','counts':dict(counts),'policy_hash':scanner.policy_hash,
             'input_records':state['next_row'],'exported':sum(counts[k] for k in PASS),'training_ready':False,
             'scope':'Safety-screened review candidates, not factual approval. Unavailable or withheld child archives cause parent dependency review.',
             'links_identity':identity(links) if links.exists() else None,'links_sha256':links_digest.hexdigest() if links.exists() else None})
    finally: db.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=ap.add_subparsers(dest='command',required=True)
    for name in ('scan','review','apply-review','redact','export'):
        p=sub.add_parser(name);p.add_argument('--output',type=Path,required=True);p.add_argument('--policy',type=Path,default=HERE/'safety_policy.json')
        if name=='scan':
            p.add_argument('--input',type=Path,required=True);p.add_argument('--contract',type=Path)
            p.add_argument('--workers',type=int,default=4);p.add_argument('--resume',action='store_true');p.add_argument('--limit',type=int)
            p.add_argument('--max-record-bytes',type=int,default=32*1024**2)
        if name=='review':
            p.add_argument('--config',type=Path,default=HERE/'config.example.json')
            p.add_argument('--enable-llm',action='store_true',help='Enable review without changing the frozen rule policy')
        if name=='apply-review':p.add_argument('--review-file',type=Path,required=True)
        if name=='redact':p.add_argument('--limit',type=int,default=36)
        if name=='export':p.add_argument('--links',type=Path)
    args=ap.parse_args()
    if hasattr(args,'workers') and args.workers<1:ap.error('workers must be positive')
    if hasattr(args,'limit') and args.limit is not None and args.limit<1:ap.error('limit must be positive')
    if hasattr(args,'max_record_bytes') and args.max_record_bytes<1:ap.error('max-record-bytes must be positive')
    policy,contract,scanner=load(args)
    with locked(args.output):
        if args.command=='scan':scan(args,policy,contract,scanner)
        elif args.command=='review':review(args,policy,scanner)
        elif args.command=='apply-review':apply_review(args,policy,scanner)
        elif args.command=='redact':redact(args,scanner)
        else:export(args,scanner)


if __name__=='__main__':main()
