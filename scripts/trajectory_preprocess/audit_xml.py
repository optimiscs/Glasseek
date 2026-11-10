"""Seeded format audit of filtered trajectories against local Qwen templates.

This creates review samples, not a training release. No models, providers or
dataset code run; original filtered inputs and raw source pointers are retained.
"""
import argparse
import collections
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'Shared'),str(ROOT/'scripts/trajectory_preprocess'),str(ROOT/'RL_training')]
from glasseek_shared.trajectory_release import validate_messages
from glasseek_contract import build_contract,materialize
from ingest import normalize,harness_plan
from migrate_searchswarm import review_messages
from tool_xml import CALL,THINK,normalize_text
from glasseek_training.web_tools import parse_tool_calls

SOURCES=['DeepForge','OpenSeeker','LiteResearcher','REDSearcher','QUEST-Objective','QUEST-Open-ended','SearchSwarm','Glasseek']
LEGACY=re.compile(r'<(?:\|)?tool_calls?_(?:begin|end)(?:\|)?>|<｜tool▁calls▁(?:begin|end)｜>')
FRAME=re.compile(r'<\|im_start\|>|<\|im_end\|>')

def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def select(path,count,seed):
    pools=collections.defaultdict(list);seen=collections.Counter();rngs={}
    with path.open() as f:
        for line_no,line in enumerate(f):
            row=json.loads(line);source=row['source']
            if source not in SOURCES:continue
            key=(source,row.get('agent_role','parent') if source=='SearchSwarm' else 'all')
            budget=(count//2 if key[1]=='parent' else count-count//2) if source=='SearchSwarm' else count
            if key not in rngs:rngs[key]=random.Random(digest([seed,*key]))
            seen[key]+=1
            if len(pools[key])<budget:pools[key].append((line_no,row))
            else:
                i=rngs[key].randrange(seen[key])
                if i<budget:pools[key][i]=(line_no,row)
    return sorted((entry for entries in pools.values() for entry in entries),key=lambda x:(SOURCES.index(x[1]['source']),x[0])),seen

def input_messages(row):
    messages=copy.deepcopy(row['messages'])
    # The filtered XML archive stores identity in call_links, while the generic
    # source adapter reads native tool_calls. These are metadata, not JSON text
    # in the assistant completion. The emitted preview removes native calls.
    for message in messages:
        if message['role']=='assistant' and message.get('call_links'):
            message['tool_calls']=[{'id':link['call_id'],'function':{'name':link['name'],'arguments':copy.deepcopy(link['arguments'])}} for link in message['call_links']]
    return messages

def template_projection(messages):
    """Separate a leading think block to avoid Qwen3.8 adding it twice."""
    projected=[]
    for original in messages:
        message={'role':original['role'],'content':original['content']}
        if message['role']=='assistant':
            blocks=list(THINK.finditer(message['content']))
            if blocks:
                if len(blocks)!=1 or message['content'][:blocks[0].start()].strip():
                    raise ValueError('reasoning_is_not_one_leading_block')
                block=blocks[0]
                message['reasoning_content']=block.group()[len('<think>'):-len('</think>')]
                message['content']=message['content'][block.end():].lstrip('\n')
            else:message['reasoning_content']=''
        projected.append(message)
    return projected

def syntax_check(messages):
    checks={'calls':0,'responses':0,'json_call_blocks':0,'legacy_control_markers':0,'role_control_markers':0,'web_runtime_calls_checked':0}
    for message in messages:
        text=message['content'];checks['role_control_markers']+=len(FRAME.findall(text))
        if message['role']=='assistant':
            expected=[{'name':link['name'],'arguments':link['arguments']} for link in message.get('call_links',[])]
            canonical,calls=normalize_text(text,expected)
            if canonical!=text or calls!=expected:raise ValueError('noncanonical_xml_or_metadata_mismatch')
            visible=THINK.sub('',text)
            checks['json_call_blocks']+=sum(m.group(1).lstrip().startswith(('{','[')) for m in CALL.finditer(visible))
            outside=CALL.sub('',visible)
            checks['legacy_control_markers']+=len(LEGACY.findall(outside))
            expected_web=[call for call in expected if call['name'] in ('web_search','web_extract')]
            actual_web=[call for call in parse_tool_calls(text) if call['name'] in ('web_search','web_extract')]
            if expected_web!=actual_web:raise ValueError('actual_web_parser_roundtrip_mismatch')
            checks['web_runtime_calls_checked']+=len(expected_web);checks['calls']+=len(calls)
        elif message.get('response_to'):
            if message['role']!='user' or text.count('<tool_response>')!=1 or text.count('</tool_response>')!=1:
                raise ValueError('noncanonical_or_double_tool_response')
            checks['responses']+=1
    if any(checks[k] for k in ('json_call_blocks','legacy_control_markers','role_control_markers')):
        raise ValueError('noncanonical_control_tokens')
    return checks


def format_filtered_row(row,source,source_line,contract):
    config={'search_name':'web_search','extract_name':'web_extract','extract_max_urls':5,'contract':contract}
    item=normalize({'messages':input_messages(row)},row['source'],str(source),source_line)
    item['id']=row['id'];item['harness']=harness_plan(item,config)
    errors=sorted(set(item['program_errors']+item['harness']['errors']))
    if errors:raise ValueError('format_pipeline:'+','.join(errors))
    materialize(item);messages=review_messages(item)
    if 'tool_prompt_protocol_review_required' in item['harness']['flags']:raise ValueError('source_tool_protocol_requires_review')
    allowed={k:v for k,v in contract['schemas'].items() if row.get('agent_role')!='child' or k not in contract['delegation']['blocked_child_tools']}
    validate_messages(messages,allowed);checks=syntax_check(messages)
    by_id={link['call_id']:link for m in messages for link in m.get('call_links',[])}
    for message in messages:
        if message.get('response_to'):
            link=by_id[message['response_to']]
            if link['name']=='web_search' and 'search_return_audit' not in message:raise ValueError('missing_search_return_audit')
    return {'id':row['id'],'source':row['source'],'agent_role':row.get('agent_role','parent'),'messages':messages,
            'tools':[{'type':'function','function':s} for s in allowed.values()],
            'source_provenance':row['provenance'],'input_filtered_row':source_line,
            'format_transform_provenance':{'tool_prompt':item.get('tool_prompt_provenance'),
                'assistant_frames':[{'step_id':s['step_id'],**s['tool_format_provenance']} for s in item['steps'] if s.get('tool_format_provenance')]},
            'training_ready':False,'harness_replayed':False,'format_checks':checks,
            'review_flags':sorted(set(row.get('review_flags',[])+item['harness']['flags'])),
            'note':'Format-audit preview. Input screening is not inherited as approval for this changed payload.'}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--per-dataset',type=int,default=8);ap.add_argument('--seed',type=int,default=20261108)
    ap.add_argument('--tokenizer',type=Path,action='append',required=True)
    args=ap.parse_args();out=args.output.resolve();source=args.input.resolve()
    if out.exists():raise ValueError('Use a new output directory')
    os.umask(0o077);out.mkdir(parents=True);selected,pool_counts=select(source,args.per_dataset,args.seed)
    contract=build_contract()
    from transformers import AutoTokenizer
    tokenizers=[AutoTokenizer.from_pretrained(str(p),local_files_only=True) for p in args.tokenizer]
    models=[{'name':p.name,'path':str(p),'template_sha256':hashlib.sha256(t.chat_template.encode()).hexdigest(),
             'uses_tool_calls_begin':'tool_calls_begin' in t.chat_template} for p,t in zip(args.tokenizer,tokenizers)]
    results=[];selection=[];summary=collections.defaultdict(collections.Counter)
    for source_line,row in selected:
        ident=row['id'];src=row['source'];entry={'id':ident,'source':src,'agent_role':row.get('agent_role','parent')}
        selection.append({**entry,'input_line_zero_based':source_line,'input_row_sha256':digest(row),'raw_source_pointer':row['provenance']})
        save(out/'input_snapshots'/src/(ident+'.json'),row)
        try:
            formatted=format_filtered_row(row,source,source_line,contract)
            messages=formatted['messages'];checks=formatted['format_checks']
            entry.update(status='passed',**checks)
            projected=template_projection(messages)
            save(out/'template_inputs'/src/(ident+'.json'),{'messages':projected,'tools':formatted['tools']})
            for model,tokenizer in zip(models,tokenizers):
                text=tokenizer.apply_chat_template(projected,tools=formatted['tools'],tokenize=False,add_generation_prompt=False,enable_thinking=True,preserve_thinking=True)
                rendered_assistants=re.findall(r'<\|im_start\|>assistant\n([\s\S]*?)<\|im_end\|>',text)
                originals=[m for m in messages if m['role']=='assistant']
                if len(rendered_assistants)!=len(originals):raise ValueError('rendered_assistant_count_mismatch')
                for rendered,original in zip(rendered_assistants,originals):
                    _,calls=normalize_text(rendered)
                    expected=[{'name':link['name'],'arguments':link['arguments']} for link in original.get('call_links',[])]
                    if calls!=expected:raise ValueError('rendered_xml_calls_changed')
                    if re.search(r'<think>\s*<think>',rendered):raise ValueError('double_thinking_wrapper')
                target=out/'rendered'/model['name']/src/(ident+'.txt');target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
                tokens=len(tokenizer.encode(text,add_special_tokens=False))
                entry[model['name']+'_tokens']=tokens;entry[model['name']+'_under_128k']=tokens<=131072
            save(out/'normalized'/src/(ident+'.json'),formatted)
            with (out/'normalized'/src/'samples.jsonl').open('a') as f:f.write(json.dumps(formatted,ensure_ascii=False)+'\n')
            summary[src]['passed']+=1;summary[src]['calls']+=checks['calls'];summary[src]['responses']+=checks['responses']
        except (ValueError,TypeError,KeyError) as exc:
            entry.update(status='review',error=type(exc).__name__+': '+str(exc));summary[src]['review']+=1
            save(out/'format_holds'/src/(ident+'.json'),entry)
        summary[src]['sampled']+=1;results.append(entry)
        print(json.dumps({'source':src,'id':ident,'status':entry['status']}),flush=True)
    save(out/'selection.json',{'seed':args.seed,'requested_per_dataset':args.per_dataset,'records':selection})
    save(out/'checks.json',results);save(out/'tools.json',contract)
    manifest={'status':'completed','version':'filtered-xml-sample-audit-v1','input':str(source),'input_sha256':sha(source),
              'seed':args.seed,'requested_per_dataset':args.per_dataset,'sample_rows':len(selected),
              'pool_counts':[{'source':s,'role':r,'rows':n} for (s,r),n in sorted(pool_counts.items())],
              'by_dataset':dict(summary),'tokenizers':models,'llm_calls':0,'live_tool_calls':0,'training_ready':False,
              'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'scripts/trajectory_preprocess/tool_xml.py',ROOT/'scripts/trajectory_preprocess/tool_prompt.py',ROOT/'scripts/trajectory_preprocess/web_batching.py',ROOT/'RL_training/glasseek_training/web_tools.py',ROOT/'Shared/glasseek_shared/hermes/qwen_xml_protocol.py',ROOT/'Shared/glasseek_shared/search_protocol.py',ROOT/'Shared/glasseek_shared/web_return_audit.py')},
              'notes':['Sampling pool is the previous offline-passed subset, not the full raw corpus. OpenResearcher is excluded.',
                       'Glasseek has only 3 offline-passed records; all are included without duplication.',
                       'SearchSwarm is stratified by parent/child role. No resampling replaces a failed format check.',
                       'Canonical XML messages retain inline thinking. Template input projections separate a leading think block into reasoning_content to avoid duplicate wrappers.',
                       'Original filtered input snapshots and raw source coordinates are preserved. No source records or previous decisions are overwritten.',
                       'No training publication, semantic-quality approval or live tool replay.']}
    save(out/'manifest.json',manifest)
    print(json.dumps({'by_dataset':dict(summary),'sample_rows':len(selected),'output':str(out)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
