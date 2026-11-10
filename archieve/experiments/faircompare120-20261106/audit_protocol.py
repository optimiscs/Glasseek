"""Read-only audit of actual main prompts, tools, budgets and exported sessions."""
import hashlib
import json
from pathlib import Path
import sqlite3

D=Path(__file__).resolve().parent
E=D.parents[1]/'Evaluation'
H=E/'data/runtime/hermes'
questions={r['id']:r['question'] for r in map(json.loads,(D/'questions.jsonl').read_text().splitlines())}
template=(D/'user_prompt_template.txt').read_text().strip()
system=(D/'system_prompt.txt').read_text().strip()
report={}
tool_hashes=set()
for label in ['step39','step76','sft9b']:
    run=f'fair120_20261106_{label}'
    rows=[]
    for question in questions:
        for p in (H/'sessions'/run/'quest_qwen27b_128'/question/'main').glob('session_*.json'):
            s=json.loads(p.read_text())
            if s.get('parent_session_id'):continue
            first=next((m.get('content','') for m in s.get('messages',[]) if m.get('role')=='user'),'')
            tools=hashlib.sha256(json.dumps(s.get('tools'),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            tool_hashes.add(tools)
            rows.append(dict(question=question,session=s['session_id'],
                user_matches=first.strip()==template.replace('{question}',questions[question]),
                system_matches=s.get('system_prompt','').strip()==system,tool_schema_sha256=tools))
    bad_sessions=[]
    budget_violations=[]
    for p in (E/'output/preds'/run/'quest_qwen27b_128/conv').glob('*/*.json'):
        r=json.loads(p.read_text());profile=H/'profiles'/r['profile'];sid=r.get('session_id')
        if sid and (profile/'state.db').exists():
            with sqlite3.connect(f'file:{profile / "state.db"}?mode=ro',uri=True) as c:
                seen=set()
                while sid and sid not in seen:
                    seen.add(sid)
                    a=c.execute('SELECT source,parent_session_id FROM sessions WHERE id=?',(sid,)).fetchone()
                    if not a:break
                    if a[0]=='subagent':bad_sessions.append(r['question_id']);break
                    sid=a[1]
        if (profile/'tool_call_budget.json').exists():
            b=json.loads((profile/'tool_call_budget.json').read_text())
            if b['used']>120 or b['limit']!=120:budget_violations.append(r['question_id'])
    report[label]=dict(main_snapshots=len(rows),prompt_mismatches=[r for r in rows if not(r['user_matches'] and r['system_matches'])],
        subagent_export_violations=bad_sessions,tool_budget_violations=budget_violations,rows=rows)
report['shared_tool_schema_count']=len(tool_hashes)
(D/'actual-protocol-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:{a:b for a,b in v.items() if a!='rows'} if isinstance(v,dict) else v for k,v in report.items()},ensure_ascii=False))
