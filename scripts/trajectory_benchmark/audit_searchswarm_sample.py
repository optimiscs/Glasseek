"""Inspect 36 SearchSwarm bundles and materialize a review-only call mapping.

No source data is changed, no dataset code is executed, and no child agent runs.
This is not a train-ready export or a full-dataset pass-rate estimate.
"""
import collections
import hashlib
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'trajectory_preprocess'))
from tool_xml import normalize_text,serialize
from glasseek_contract import build_contract,validate_args

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'data/SearchSwarm-SFT/train.parquet'
OUT=ROOT/'reports/searchswarm-glasseek-compatibility-20261107'

def calls(messages):
    output=[];errors=[]
    for i,m in enumerate(messages or []):
        if not isinstance(m,dict) or m.get('role')!='assistant':continue
        try:_,parsed=normalize_text(m.get('content') or '')
        except (ValueError,TypeError,KeyError) as exc:
            errors.append({'message_index':i,'error':type(exc).__name__});continue
        output.extend((i,c) for c in parsed)
    return output,errors

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    contract=build_contract();stats=collections.Counter();names=collections.Counter();sizes=collections.Counter();samples=[];example=None
    pf=pq.ParquetFile(SOURCE)
    for idx,batch in enumerate(pf.iter_batches(batch_size=1,columns=['source','question','messages','subagents','num_subagents'],use_threads=False)):
        if idx>=36:break
        r=batch.to_pylist()[0];msgs=r['messages'] or [];children=r['subagents'] or []
        stats['bundles']+=1;stats['children_declared']+=r['num_subagents'] or 0;stats['children_present']+=len(children)
        stats['declared_count_mismatch']+=int((r['num_subagents'] or 0)!=len(children))
        child_questions=collections.Counter(c.get('question') for c in children)
        seen=collections.Counter();parent_calls,errors=calls(msgs)
        stats['bundles_parent_parse_error']+=int(bool(errors));names.update(c['name'] for _,c in parent_calls)
        issues=[];subcalls=0;taskcount=0;overcap=0;not_unique=0;missing_labels=0
        for i,c in parent_calls:
            if c['name']!='call_sub_agent':continue
            subcalls+=1;prompts=c['arguments'].get('prompts')
            if not isinstance(prompts,list):issues.append('invalid_prompts');continue
            sizes[len(prompts)]+=1;taskcount+=len(prompts)
            if len(prompts)>contract['delegation']['max_concurrent_children']:overcap+=1
            items=[];allmatched=True;labels=[]
            for task in prompts:
                if not isinstance(task,dict) or not isinstance(task.get('prompt'),str) or not isinstance(task.get('goal'),str):
                    issues.append('invalid_task');allmatched=False;continue
                seen[task['prompt']]+=1
                matches=child_questions[task['prompt']]
                if matches!=1:not_unique+=1;allmatched=False
                labels.append(task['goal'])
                # Source goal is a response label, not a child instruction. Target goal
                # is the full prompt: preserving the actual child-visible instructions.
                items.append({'goal':task['prompt']})
            response=msgs[i+1].get('content','') if i+1<len(msgs) and msgs[i+1].get('role') in ('tool','user') else ''
            for label in labels:
                if label not in response:missing_labels+=1
            if example is None and len(prompts)<=3 and allmatched and not validate_args('delegate_task',{'tasks':items},contract):
                original={'name':'call_sub_agent','arguments':c['arguments']};mapped={'name':'delegate_task','arguments':{'tasks':items}}
                xml=serialize(mapped)
                assert normalize_text(xml,[mapped])[1]==[mapped]
                example={'source_row_index':idx,'source_message_index':i,'source_call':original,'mapped_call':mapped,'mapped_xml':xml,
                    'source_goal_labels':labels,'source_child_question_exact_matches':[child_questions[t['prompt']] for t in prompts],
                    'source_following_response':response,'output_status':'review_only',
                    'training_ready':False,'runtime_replayed':False,
                    'limitations':['One call only, not a converted whole trajectory.','Historical response unchanged; Hermes response shape differs.','Child tool availability and source system prompt require reconciliation.']}
        stats['parent_delegate_calls']+=subcalls;stats['delegated_tasks']+=taskcount
        stats['bundles_with_delegate_calls']+=int(subcalls>0);stats['calls_exceed_batch_cap']+=overcap
        stats['bundles_with_overcap_call']+=int(overcap>0);stats['tasks_without_exact_unique_child']+=not_unique
        stats['response_labels_not_found']+=missing_labels
        stats['children_not_referenced_by_exact_prompt']+=sum(n for q,n in child_questions.items() if seen[q]==0)
        stats['bundles_all_tasks_have_unique_child']+=int(subcalls>0 and not not_unique)
        stats['bundles_with_any_task_without_unique_child']+=int(not_unique>0)
        childnames=collections.Counter()
        for child in children:
            cc,ce=calls(child.get('messages'));childnames.update(c['name'] for _,c in cc)
            stats['children_with_parse_error']+=int(bool(ce))
        stats['children_using_python']+=sum(any(c['name']=='PythonInterpreter' for _,c in calls(ch.get('messages'))[0]) for ch in children)
        stats['children_using_scholar']+=sum(any(c['name']=='google_scholar' for _,c in calls(ch.get('messages'))[0]) for ch in children)
        samples.append({'row_index':idx,'source':r['source'],'declared_children':r['num_subagents'],'actual_children':len(children),
          'parent_call_count':subcalls,'delegated_tasks':taskcount,'calls_over_batch_limit':overcap,
          'tasks_without_unique_exact_child':not_unique,'response_labels_not_found':missing_labels,
          'parent_tools':dict(collections.Counter(c['name'] for _,c in parent_calls)), 'child_tools':dict(childnames),'parse_errors':errors,'issues':issues})
    summary={'status':'completed','sample_selection':'First 36 bundles in parquet order; exploratory, not a representative estimate.',
             'source':str(SOURCE),'scope':'No code execution, live tool replay, LLM judgments, or dataset modification.',
             'profile':contract['profile'],'contract_sha256':contract['sha256'],'batch_limit':contract['delegation']['max_concurrent_children'],
             'aggregate':dict(stats),'parent_call_names':dict(names),'parallel_batch_sizes':dict(sorted(sizes.items())),
             'examples':samples,'candidate_call_example':bool(example)}
    (OUT/'sample_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    if example:(OUT/'review_only_call_mapping.json').write_text(json.dumps(example,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='examples'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
