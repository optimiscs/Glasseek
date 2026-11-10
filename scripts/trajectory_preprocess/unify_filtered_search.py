"""Normalize every record in a filtered pool; audit all search calls offline."""
import argparse
import collections
import copy
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'Shared'),str(ROOT/'scripts/trajectory_preprocess'),str(ROOT/'scripts/trajectory_preprocess')]
from audit_xml import format_filtered_row,syntax_check,sha,save
from glasseek_contract import build_contract
from client import digest as content_digest
from glasseek_shared.extract_protocol import merged_response_slices,batch_response_slices
from glasseek_shared.web_return_audit import analyze_search_return
from glasseek_shared.trajectory_release import validate_messages

CONTRACT=None
INPUT=None
AUDIT_ONLY=False
def initialize(contract,path,audit_only):
    global CONTRACT,INPUT,AUDIT_ONLY
    CONTRACT=contract;INPUT=path;AUDIT_ONLY=audit_only

def update_audits(row):
    result=copy.deepcopy(row);messages=result['messages']
    by_id={link['call_id']:link for m in messages for link in m.get('call_links',[])}
    before=[(m['role'],m['content']) for m in row['messages']]
    for message in messages:
        link=by_id.get(message.get('response_to'))
        if not link or link['name']!='web_search':continue
        text=message['content'];spans=[]
        if 'merged_response_parts' in message:spans=merged_response_slices(text,message['merged_response_parts'])
        elif 'batch_response_parts' in message:spans=batch_response_slices(text,message['batch_response_parts'])
        audit=analyze_search_return(link['arguments']['query'],text,link,[text[a:b] for a,b in spans])
        audit['rendered_response_sha256']=hashlib.sha256(text.encode()).hexdigest()
        message['search_return_audit']=audit
    allowed={k:v for k,v in CONTRACT['schemas'].items() if row.get('agent_role')!='child' or k not in CONTRACT['delegation']['blocked_child_tools']}
    validate_messages(messages,allowed);result['format_checks']=syntax_check(messages)
    assert before==[(m['role'],m['content']) for m in messages]
    result['audit_update']={'input':INPUT,'version':'search-return-audit-v3','source_content_sha256':content_digest(row),'model_visible_content_unchanged':True}
    return result

def prepare(entry):
    index,row=entry
    try:return {'status':'passed','row':update_audits(row) if AUDIT_ONLY else format_filtered_row(row,INPUT,index,CONTRACT)}
    except (ValueError,TypeError,KeyError) as exc:
        return {'status':'held','id':row['id'],'source':row['source'],'input_filtered_row':index,
                'error':type(exc).__name__+': '+str(exc)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=16);ap.add_argument('--audit-only',action='store_true');args=ap.parse_args()
    out=args.output.resolve();source=args.input.resolve()
    if out.exists():raise ValueError('Use a new output directory')
    os.umask(0o077);out.mkdir(parents=True);(out/'by_dataset').mkdir();started=time.monotonic()
    contract=build_contract();versions={}
    for directory in ('scripts/trajectory_preprocess','Shared/glasseek_shared'):
        for p in (ROOT/directory).glob('*.py'):versions[str(p.relative_to(ROOT))]=sha(p)
    for relative in ('Shared/glasseek_shared/hermes/qwen_xml_protocol.py','scripts/trajectory_preprocess/audit_xml.py','scripts/trajectory_preprocess/unify_filtered_search.py','RL_training/glasseek_training/web_tools.py','Evaluation_backend/tools/web_tools.py'):
        versions[relative]=sha(ROOT/relative)
    for relative in versions:
        p=out/'source_snapshot'/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/relative).read_bytes())
    save(out/'manifest.json',{'status':'running','input':str(source),'workers':args.workers,'source_sha256':versions,'llm_calls':0,'live_tool_calls':0})
    counts=collections.defaultdict(collections.Counter);audit_counts=collections.Counter();handles={};total=0
    def consume(result,canonical,held,audits,inventory):
        nonlocal total
        total+=1
        if result['status']=='held':
            counts[result['source']]['held']+=1;held.write(json.dumps(result,ensure_ascii=False)+'\n');return
        row=result['row'];src=row['source'];c=counts[src];c['passed']+=1
        line=json.dumps(row,ensure_ascii=False)+'\n';canonical.write(line)
        inventory.write(json.dumps({'id':row['id'],'source':src,'content_sha256':content_digest(row)})+'\n')
        if src not in handles:handles[src]=(out/'by_dataset'/(src+'.jsonl')).open('w')
        handles[src].write(line)
        by_id={link['call_id']:link for m in row['messages'] for link in m.get('call_links',[])}
        for message in row['messages']:
            audit=message.get('search_return_audit')
            if audit is None:continue
            c['search_calls']+=1
            query=by_id[message['response_to']]['arguments']['query']
            c['array_calls' if isinstance(query,list) else 'string_calls']+=1
            c['queries']+=len(query) if isinstance(query,list) else 1
            c['association_verified']+=audit['association']['status']=='verified'
            c['per_query_association_verified']+=audit['coverage']['per_query_association_verified']
            c['coverage_review']+=audit['coverage']['needs_review']
            c['execution:'+audit['execution']['status']]+=1
            audit_counts[(src,'coverage',audit['coverage']['status'])]+=1
            audit_counts[(src,'execution',audit['execution']['status'])]+=1
            audit_counts[(src,'query_results',audit['query_results']['status'])]+=1
            audits.write(json.dumps({'id':row['id'],'source':src,'call_id':message['response_to'],'query':query,
                                     'response_to':message['response_to'],'audit':audit},ensure_ascii=False)+'\n')
    with (out/'canonical.jsonl').open('w') as canonical,(out/'held.jsonl').open('w') as held,(out/'search_calls.jsonl').open('w') as audits,(out/'content_hashes.jsonl').open('w') as inventory:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(contract,str(source),args.audit_only)) as pool:
            with source.open() as f:
                batch=[]
                for index,line in enumerate(f):
                    batch.append((index,json.loads(line)))
                    if len(batch)<64:continue
                    for result in pool.map(prepare,batch,chunksize=4):consume(result,canonical,held,audits,inventory)
                    batch=[]
                    if total%512==0:print(json.dumps({'rows':total,'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
                for result in pool.map(prepare,batch,chunksize=4):consume(result,canonical,held,audits,inventory)
    for handle in handles.values():handle.close()
    for name,expected in versions.items():assert sha(ROOT/name)==expected,'Source changed: '+name
    table=[{'dataset':src,'read':c['passed']+c['held'],**{key:c[key] for key in ('passed','held','search_calls','string_calls','array_calls','queries','association_verified','per_query_association_verified','coverage_review')}} for src,c in sorted(counts.items())]
    with (out/'by_dataset.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    summary={'status':'completed','input_scope':'All records in the existing filtered pool, not the full raw corpus.',
             'read':total,'passed':sum(c['passed'] for c in counts.values()),'held':sum(c['held'] for c in counts.values()),
             'by_dataset':table,'search_audits':[{'source':s,'category':k,'status':v,'calls':n} for (s,k,v),n in sorted(audit_counts.items())],
             'llm_calls':0,'live_tool_calls':0,'training_ready':False,'semantic_quality':'not_evaluated','runtime_replay':False}
    summary['pass_meaning']='XML/schema/association consistency only; query labels, result presence and reported execution remain separate fields.'
    save(out/'summary.json',summary);save(out/'tools.json',contract)
    files=[p for p in out.glob('*.jsonl')]+list((out/'by_dataset').glob('*.jsonl'))+[out/'by_dataset.csv',out/'summary.json',out/'tools.json']
    save(out/'manifest.json',{'status':'completed','version':'filtered-search-xml-unification-v1','input':str(source),'input_sha256':sha(source),
        'source_sha256':versions,'files':[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],
        'elapsed_seconds':time.monotonic()-started,'rows':summary['read'],'passed':summary['passed'],'held':summary['held'],
        'llm_calls':0,'live_tool_calls':0,'training_ready':False,'audit_only':args.audit_only,'scope':summary['input_scope']})
    print(json.dumps({'read':total,'passed':summary['passed'],'held':summary['held'],'by_dataset':table,'output':str(out)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
