import json,time
from pathlib import Path
D=Path(__file__).resolve().parent;E=D.parents[2]/'Evaluation';base='glasseek_clean2899_full64k_step39_128';all_runs=[]
for i in range(1,4):
 name=base if i==1 else base+f'_rep{i}';root=E/'output/preds'/name/'quest_qwen27b_128';results={};judges={}
 for f in (root/'conv').rglob('*.json'):
  x=json.loads(f.read_text());results[x['question_id']]=x
 for f in (root/'judge_official').rglob('*.json'):
  x=json.loads(f.read_text());judges[x['question_id']]=x
 correct=wrong=budget_failed=errors=unscored=pending=0
 for line in (D.parent/'browsecomp128.jsonl').read_text().splitlines():
  q=json.loads(line)['id'];r=results.get(q);j=judges.get(q,{})
  if r is None or r.get('status')=='interrupted' or r.get('attempt_consumed') is False:pending+=1
  elif r.get('tool_call_budget_exceeded'):budget_failed+=1
  elif r.get('status')!='success':errors+=1
  elif j.get('status')!='judged':unscored+=1
  elif j.get('score')==1:correct+=1
  else:wrong+=1
 all_runs.append(dict(repeat=i,run=name,correct=correct,wrong=wrong,tool_budget_failed=budget_failed,runtime_errors=errors,unscored=unscored,pending=pending,accuracy=correct/128 if not (pending or unscored) else None,success_fraction_so_far=correct/128))
full=all(r['accuracy'] is not None for r in all_runs)
report=dict(updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),questions_per_repeat=128,runs=all_runs,Avg_at_3=sum(r['accuracy'] for r in all_runs)/3 if full else None,note='Avg@3 is mean of three correct/128 rates, not pass@3. Runtime errors are reported separately; they provide no successful answers. Repeat 1 retains pre-switch trajectories; tool counts over 120 are scored as budget failures.')
(D/'avg3.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
