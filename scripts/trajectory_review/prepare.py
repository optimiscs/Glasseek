"""Seed a bounded, stratified manual audit from existing rule-scan coordinates."""
import argparse
import collections
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'scripts/trajectory_preprocess'),str(ROOT/'Shared')]
from trajectory_review.core import connect, RUBRIC, RUBRIC_HASH, VERSION, digest
from client import digest as source_digest
from safety_rules import Scanner
from safety_context import build_context
from glasseek_shared.dataset_policy import excluded_source


def automatic(item,result):
    rules={f['rule_id'] for f in result['findings']}
    answers={r['id']:'unknown' for r in RUBRIC};basis={}
    if result['coverage_complete']:
        answers['leak_read']='yes' if 'benchmark.current_question_answer' in rules else 'unknown' if any(r.startswith('benchmark.') for r in rules) else 'no'
        answers['secret_exposed']='yes' if 'credential.private_key' in rules else 'unknown' if any(r.startswith('credential.') for r in rules) else 'no'
    for key,parent in [('leak_used','leak_read'),('secret_sent','secret_exposed')]:
        if answers[parent]=='no':answers[key]='no'
    schema=[r for r in rules if r.startswith('schema.')]
    alignment={'schema.missing_tool_response','schema.unmatched_tool_response_id','schema.orphan_tool_response','schema.duplicate_tool_call_id'}
    answers['tool_schema']='no' if set(schema)-alignment else 'yes'
    _,actions,_=build_context(item)
    counts=collections.Counter(a['message_index'] for a in actions)
    ambiguous=any(n>1 and not item['messages'][i].get('call_links') and not item['messages'][i].get('tool_calls') for i,n in counts.items())
    answers['response_alignment']='no' if set(schema)&alignment else 'unknown' if ambiguous else 'yes'
    if not result['coverage_complete']:
        answers['tool_schema']=answers['response_alignment']='unknown'
    basis.update(leak_read='当前题目重合 + 答案标记为规则预测；无命中仅表示扫描未发现，使用情况仍待人工判断。',
                 secret_exposed='私钥标记为强信号；其他疑似密钥需人工确认，未进行密钥联网验证。',
                 tool_schema='对照该候选快照中冻结的工具 schema 与能力约束。',
                 response_alignment='检查调用/响应结构；多个调用共用无 ID 返回时不猜测关联。')
    return {'answers':answers,'basis':basis,'scan_state':result['state'],'coverage_complete':result['coverage_complete'],
            'rule_ids':sorted(rules),'findings':result['findings'],'policy_hash':result['policy_hash'],
            'llm_calls':0,'training_ready':False}


def prepare(run,out,policy_path,key_path,limit=36):
    if not 1<=limit<=36:raise ValueError('Initial manual audit batch must contain 1–36 records')
    out.mkdir(parents=True,exist_ok=False)
    policy=json.loads(policy_path.read_text());contract=json.loads((run/'contract.json').read_text())
    scanner=Scanner(policy,contract,key_path.read_bytes().strip())
    samples=json.loads((run/'report/audit_samples.json').read_text())
    samples=[s for s in samples if not excluded_source(s['source'])]
    rank=lambda s:digest(['manual-review-v1',s['record_key']])
    selected=[];used=set()
    for source in sorted({s['source'] for s in samples}):
        pool=sorted([s for s in samples if s['source']==source],key=rank)
        for kind in ('benchmark','credential','pass_rules','other'):
            available=[s for s in pool if s['record_key'] not in used]
            desired=[s for s in available if (s['state']=='pass_rules' if kind=='pass_rules' else any(r.startswith(kind+'.') for r in s['rule_ids']))]
            if not available:continue
            sample=(desired or available)[0];selected.append(sample);used.add(sample['record_key'])
    for sample in sorted(samples,key=lambda s:(not any(r.startswith(('benchmark.','credential.')) for r in s['rule_ids']),rank(s))):
        if len(selected)>=limit:break
        if sample['record_key'] not in used:selected.append(sample);used.add(sample['record_key'])
    selected=selected[:limit]
    old=sqlite3.connect('file:'+str(run/'state.sqlite')+'?mode=ro',uri=True)
    manifest=json.loads((run/'manifest.json').read_text());source=Path(manifest['input']['path'])
    database=out/'reviews.sqlite';db=connect(database)
    totals=collections.Counter();by_source=collections.defaultdict(collections.Counter)
    with source.open('rb') as stream:
        for sample in selected:
            original=json.loads(old.execute('SELECT result FROM decisions WHERE record_key=?',(sample['record_key'],)).fetchone()[0])
            stream.seek(original['offset_start']);item=json.loads(stream.read(original['offset_end']-original['offset_start']))
            if source_digest(item)!=original['content_hash']:raise ValueError('Source content no longer matches scan receipt')
            if excluded_source(item.get('source')):continue
            result=scanner.scan(item);program=automatic(item,result)
            ident=digest([item['id'],result['content_hash']])[:24]
            messages=[]
            for i,m in enumerate(item['messages']):
                text=scanner.scrub(m.get('content',''))
                messages.append({'index':i,'role':m['role'],'origin':'tool_response' if m['role']=='tool' or text.lstrip().startswith('<tool_response>') else m['role'],
                                 'content':text,'source_message_index':m.get('source_message_index',i),
                                 'call_links':scanner.scrub(json.dumps(m.get('call_links',[]),ensure_ascii=False)),
                                 'response_to':scanner.scrub(str(m.get('response_to') or m.get('tool_call_id') or ''))})
            question=next((m['content'] for m in messages if m['role']=='user' and m['origin']!='tool_response'),'')
            payload={'id':ident,'source':item['source'],'question':question,'messages':messages,
                     'tools':item['tools'],'contract_sha256':contract['sha256'],'agent_role':item.get('agent_role','parent')}
            location={k:sample.get(k) for k in ('source_artifact','source_line_zero_based')}
            location.update(original_id=item['id'],original_content_hash=result['content_hash'])
            db.execute('INSERT INTO cases VALUES(?,?,?,?,?,?,?)',(ident,item['source'],result['content_hash'],scanner.policy_hash,
                json.dumps(payload,ensure_ascii=False),json.dumps(program,ensure_ascii=False),json.dumps(location,ensure_ascii=False)))
            totals[result['state']]+=1;by_source[item['source']][result['state']]+=1
            print(json.dumps({'source':item['source'],'prepared':sum(totals.values())}),flush=True)
    summary={'version':VERSION,'rubric_hash':RUBRIC_HASH,'policy_hash':scanner.policy_hash,'cases':sum(totals.values()),
             'sampling':'per-source risk signals and clean controls from prior audit; targeted diagnostic sample, not population estimates',
             'totals':dict(totals),'by_source':dict(by_source),'source_scan':str(run),'llm_calls':0,'training_ready':False}
    db.execute('INSERT INTO meta VALUES(?,?)',('manifest',json.dumps(summary,ensure_ascii=False)));db.commit();db.close();old.close()
    (out/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (out/'policy.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy',type=Path,default=ROOT/'scripts/trajectory_preprocess/safety_policy.json')
    p.add_argument('--key-file',type=Path,required=True);p.add_argument('--limit',type=int,default=36);a=p.parse_args()
    print(json.dumps(prepare(a.scan,a.output,a.policy,a.key_file,a.limit),ensure_ascii=False))
