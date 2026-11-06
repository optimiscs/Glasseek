"""Record user-terminated baseline trajectories without fabricating answers."""
import json
from pathlib import Path
import os
P=Path(os.environ.get('REEVAL_DIR','/root/autodl-tmp/Glasseek/checkpoint-reeval/analysis/checkpoint_reeval'))
E=Path(os.environ.get('GLASSEEK_EVAL_ROOT','/root/autodl-tmp/Glasseek/Evaluation'))
dataset='browsecomp_rl9b_non_timeout_34'; run='glasseek_base9b_rl9b_subset'
root=E/'output/preds'/run/dataset; rows=[json.loads(x) for x in (P/f'{dataset}.jsonl').read_text().splitlines() if x.strip()]
existing={}
for path in (root/'conv').glob('*/*.json'):
 try:
  data=json.loads(path.read_text()); existing[data['question_id']]=(data,path)
 except (ValueError,OSError,KeyError): pass
profiles=E/'data/runtime/hermes/profiles'
marked=[]
for row in rows:
 q=row['id']
 previous=existing.get(q)
 if previous and previous[0].get('status') != 'interrupted': continue
 prefix=f'{run}_{q}__r1_a1-'
 found=sorted(x.name for x in profiles.glob(prefix+'*'))
 record={'question_id':q,'run_id':f'{q}__r1','repeat_index':1,'attempt':1,'question':row['question'],'answer':row['answer'],'type':row['type'],'model':'glasseek-base-9b','save_name':run,'status':'failed','failure_type':'user_terminated','failure_outcome':'failure','error':'用户要求终止基座长尾；计为失败，未生成最终答案','attempt_consumed':True,'trace_complete':False,'model_response':'','profile':found[-1] if found else '', 'user_terminated':True}
 if previous:
  old, path = previous
  old.update(record); record=old
 else:
  path=root/'conv'/q/f'{q}__r1.json'
 path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n'); temp.replace(path)
 with (root/'summary.jsonl').open('a') as f: f.write(json.dumps(record,ensure_ascii=False)+'\n')
 marked.append(q)
print('user_terminated_as_failed',json.dumps(marked))
