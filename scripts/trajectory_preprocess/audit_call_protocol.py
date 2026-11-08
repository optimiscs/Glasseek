"""Bounded real-data, offline protocol smoke. No LLM or live tool execution."""
import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path

from ingest import records, normalize, harness_plan
from glasseek_contract import build_contract, materialize
from migrate_searchswarm import convert_parent, review_messages
from glasseek_shared.extract_protocol import historical_results
from glasseek_shared.trajectory_release import validate_messages, digest, payload, file_hash
from glasseek_shared.dataset_policy import excluded_source


def audit(manifest_path, output, count=32):
    manifest=json.loads(Path(manifest_path).read_text())
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    contract=build_contract()
    config={'search_name':'web_search','extract_name':'web_extract','extract_max_urls':5,'contract':contract}
    sources={}
    for spec in manifest['inputs']:
        if excluded_source(spec.get('source')):
            continue
        sources.setdefault(spec['source'],spec)
    report={'sampling':'first N rows of first shard per source; diagnostic, not a population pass-rate estimate',
            'rows_per_source':count,'llm_calls':0,'live_tool_calls':0,'training_ready_rows':0,'sources':{},'multi_url_examples':[]}
    with (output/'protocol_candidates.jsonl').open('w') as candidates, (output/'held.jsonl').open('w') as held:
        for source,spec in sources.items():
            counts=collections.Counter();reasons=collections.Counter();page_shapes=collections.Counter()
            for ri,row in enumerate(itertools.islice(records(spec['path']),count)):
                counts['rows']+=1
                try:
                    if source=='SearchSwarm':
                        _,messages,_,errors=convert_parent(row,ri,Path(spec['path']),contract)
                        raw={'question':row['question'],'messages':messages}
                    else:raw=row;errors=[]
                    item=normalize(raw,source,spec['path'],ri)
                    item['harness']=harness_plan(item,config)
                    errors=sorted(set(errors+item['program_errors']+item['harness']['errors']))
                    for step in item['steps']:
                        counts['source_calls']+=len(step['calls'])
                        counts['linked_responses']+=len(step.get('responses',[]))
                        for ci,call in enumerate(step['calls']):
                            if call['name'].lower() not in ('visit','visit_urls','web_extract','webextract'):continue
                            urls=call['arguments'].get('urls',call['arguments'].get('url'))
                            if not isinstance(urls,list) or len(urls)<2:continue
                            counts['multi_url_calls']+=1
                            links=step.get('call_links',[])
                            cid=links[ci]['call_id'] if ci<len(links) else None
                            matching=[r for r in step.get('responses',[]) if cid and r['response_to']==cid]
                            if len(matching)==1:
                                data,reason=historical_results(matching[0]['content'],urls)
                            else:data,reason=None,'unverified_call_response_association'
                            shape=reason or 'explicit_json_results'
                            page_shapes[shape]+=1
                            if not any(e['source']==source and e['shape']==shape for e in report['multi_url_examples']):
                                report['multi_url_examples'].append({'source':source,'path':spec['path'],'row_zero_based':ri,
                                    'source_message_index':step.get('source_message_index'),'source_call_index':ci,
                                    'url_count':len(urls),'shape':shape,
                                    'response_sha256':hashlib.sha256(matching[0]['content'].encode()).hexdigest() if len(matching)==1 else None,
                                    'result_fields':[sorted(p) for p in data['results']] if data else None})
                    if errors:
                        reasons.update(errors);counts['adapter_held']+=1
                        held.write(json.dumps({'source':source,'row':ri,'reasons':errors})+'\n')
                        continue
                    materialize(item)
                    messages=review_messages(item)
                    try:validate_messages(messages,contract['schemas'])
                    except ValueError as exc:
                        reason=str(exc);reasons[reason]+=1;counts['protocol_held']+=1
                        held.write(json.dumps({'source':source,'row':ri,'reasons':[reason]})+'\n')
                        continue
                    candidate={'id':item['id'],'source':source,'messages':messages,'tools':[
                        {'type':'function','function':s} for s in contract['schemas'].values()],
                        'training_ready':False,'harness_replayed':False,'review_flags':item['harness']['flags'],
                        'provenance':item['provenance']}
                    candidate['content_sha256']=digest(payload(candidate))
                    candidates.write(json.dumps(candidate,ensure_ascii=False)+'\n')
                    counts['protocol_candidates']+=1
                except (ValueError,KeyError,TypeError) as exc:
                    counts['normalization_held']+=1;reasons[type(exc).__name__]+=1
            report['sources'][source]={'counts':dict(counts),'held_reasons':dict(reasons),'multi_url_return_shapes':dict(page_shapes)}
            print(json.dumps({'source':source,**dict(counts)}),flush=True)
    report['candidate_sha256']=file_hash(output/'protocol_candidates.jsonl')
    report['code_sha256']={str(p.relative_to(Path(__file__).resolve().parents[2])):file_hash(p) for p in [
        Path(__file__),Path(__file__).with_name('message_adapter.py'),Path(__file__).with_name('call_protocol.py'),
        Path(__file__).resolve().parents[2]/'Shared/glasseek_shared/trajectory_release.py']}
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (output/'acceptance.empty.jsonl').write_text('')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rows-per-source',type=int,default=32);args=p.parse_args()
    audit(args.manifest,args.output,args.rows_per_source)
