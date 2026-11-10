#!/usr/bin/env python3
"""Inventory declared local training/candidate sources without trajectory/tool filtering."""
import argparse
import glob
import json
import sqlite3
import sys
from pathlib import Path

from suite import ROOT, canonical, dump, sha

sys.path[:0] = [str(ROOT/'Shared'), str(ROOT/'scripts/trajectory_preprocess')]
from ingest import records


def content(value):
    if isinstance(value, str): return value
    if isinstance(value, list): return '\n'.join(content(v.get('text','')) if isinstance(v,dict) else content(v) for v in value)
    return ''


def question(row):
    for k in ('question','task_question','query'):
        if isinstance(row.get(k),str) and row[k].strip(): return row[k].strip()
    meta=row.get('meta') or {}
    if isinstance(meta,dict) and isinstance(meta.get('question'),str) and meta['question'].strip(): return meta['question'].strip()
    for k in ('messages','conversations','prompt','trajectory'):
        msgs=row.get(k)
        if isinstance(msgs,str):
            try: msgs=json.loads(msgs)
            except ValueError:
                if k=='prompt' and msgs.strip(): return msgs.strip()
        if not isinstance(msgs,list): continue
        for m in msgs:
            if isinstance(m,dict) and m.get('role',m.get('from')) in ('user','human'):
                text=content(m.get('content',m.get('value',''))).strip()
                if text:return text
    return ''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--audit-non-tasks',action='store_true',help='Independently verify all questionless rows are system-only stubs; preserve original manifest')
    a=p.parse_args()
    if a.audit_non_tasks:
        audit_non_tasks(a.output);return
    cfg=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
    conn=sqlite3.connect(a.output/'questions.sqlite');conn.execute('CREATE TABLE q(norm TEXT PRIMARY KEY, question TEXT, path TEXT, row INTEGER)')
    entries={}; errors=[]; unextractable=[]
    for group in cfg['training_sources']:
        paths=sorted(glob.glob(group['glob'],recursive=True))
        if not paths and group.get('required',True): errors.append({'pattern':group['glob'],'error':'no_matching_files'})
        for name in paths:
            path=Path(name).resolve()
            if str(path) in entries: continue
            n=0;miss=0
            try:
                before=path.stat(); h=sha(path)
                # Project Parquet onto question columns when possible, avoiding observations.
                if path.suffix=='.parquet':
                    import pyarrow.parquet as pq
                    pf=pq.ParquetFile(path); names=pf.schema_arrow.names
                    cols=[k for k in ('question','task_question','query') if k in names]
                    if not cols: cols=[k for k in ('meta','messages','conversations','prompt','trajectory') if k in names]
                    iterator=(r for b in pf.iter_batches(batch_size=16,columns=cols,use_threads=False) for r in b.to_pylist())
                else: iterator=records(path,max_record_chars=128000000)
                for i,r in enumerate(iterator):
                    n+=1;q=question(r)
                    if not q:
                        miss+=1
                        if len(unextractable)<100:unextractable.append({'path':str(path),'row':i,'keys':list(r)})
                    else:conn.execute('INSERT OR IGNORE INTO q VALUES (?,?,?,?)',(canonical(q),q,str(path),i))
                after=path.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): raise ValueError('source changed while indexing')
                entries[str(path)]={'sha256':h,'bytes':before.st_size,'rows':n,'unextractable':miss,'scope':group['scope']}
                conn.commit()
                print(json.dumps({'file':str(path),'rows':n,'missing_questions':miss}),flush=True)
            except Exception as e: errors.append({'path':str(path),'error':type(e).__name__+':'+str(e)[:250]})
    out=a.output/'questions.jsonl'
    with out.open('w') as f:
        for q,path,i in conn.execute('SELECT question,path,row FROM q ORDER BY norm'):
            f.write(json.dumps({'question':q,'path':path,'row':i},ensure_ascii=False,separators=(',',':'))+'\n')
    manifest={'version':1,'scope':'all declared local SFT/RL train files plus conservative downloaded trajectory candidate pool',
        'files':entries,'errors':errors,'unextractable_rows':sum(e['unextractable'] for e in entries.values()),
        'unextractable_examples':unextractable,'unique_questions':conn.execute('SELECT count(*) FROM q').fetchone()[0],
        'questions_sha256':sha(out),'script_sha256':sha(__file__),'config_sha256':sha(a.config),
        'limits':'covers declared local question text, not unknown remote corpora or model pretraining; paraphrase/translation detection is not exhaustive'}
    dump(a.output/'manifest.json',manifest)
    if errors or manifest['unextractable_rows']:sys.exit(2)


def audit_non_tasks(folder):
    manifest=json.loads((folder/'manifest.json').read_text());verified=[];unresolved=[]
    for path,entry in manifest['files'].items():
        if not entry['unextractable']:continue
        if sha(path)!=entry['sha256']:raise ValueError('source drift during non-task audit')
        found=0
        for i,r in enumerate(records(path,max_record_chars=128000000)):
            if question(r):continue
            found+=1
            messages=r.get('messages')
            # An empty/system-only conversation contains no trainable task.
            system_only=isinstance(messages,list) and all(isinstance(m,dict) and m.get('role') in ('system','developer') for m in messages)
            target=verified if system_only else unresolved
            target.append({'path':path,'row':i,'reason':'no_user_task_system_only' if system_only else 'unresolved_question_schema'})
        if found!=entry['unextractable']:raise ValueError('question extraction drift')
    revised=dict(manifest,original_manifest_sha256=sha(folder/'manifest.json'),
                 non_task_rows=verified,unextractable_rows=len(unresolved),unextractable_examples=unresolved,
                 audit_script_sha256=sha(__file__))
    dump(folder/'verified-manifest.json',revised)
    print(json.dumps({'verified_non_tasks':len(verified),'unresolved':len(unresolved)}))
    if unresolved or manifest['errors']:sys.exit(2)


if __name__=='__main__': main()
