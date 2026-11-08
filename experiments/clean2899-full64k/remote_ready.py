import json,sys
from pathlib import Path
E=Path('/root/autodl-tmp/Glasseek/Evaluation');sys.path.insert(0,str(E))
from browse_comp_eval.judge import load_prediction_record,prediction_fingerprint
name='glasseek_full64k_step76_rl9b_subset';p=E/'output/preds'/name/'browsecomp_rl9b_non_timeout_34'
active=0
for proc in Path('/proc').glob('[0-9]*/cmdline'):
 try:cmd=proc.read_bytes().decode(errors='replace').split('\0')
 except OSError:continue
 if ('browse_comp_eval.run' in cmd and name in cmd) or any(x.startswith(name+'_row') for x in cmd):active+=1
convs=list((p/'conv').rglob('*.json'));judges={}
for f in (p/'judge_official').rglob('*.json'):
 x=json.loads(f.read_text());judges[(x.get('question_id'),x.get('repeat_index'))]=x
complete=0
for f in convs:
 x=json.loads(f.read_text());j=judges.get((x.get('question_id'),x.get('repeat_index')),{});rec=load_prediction_record(f)
 if x.get('completed') and j.get('status') in ('judged','inference_error') and rec is not None and j.get('pred_fingerprint')==prediction_fingerprint(rec):complete+=1
print(json.dumps(dict(active=active,convs=len(convs),scored_complete=complete,ready=active==0 and len(convs)==34 and complete==34)))
