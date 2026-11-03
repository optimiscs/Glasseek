"""Start the requested 44-backed RL-4B evaluation after its server is ready."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT=Path('/root/autodl-tmp/Glasseek/Evaluation')
R=ROOT/'data/runtime'
RUN='glasseek_rl4_quest128_ssh44'
ID='glasseek-rl4-browsecomp-128'
SITE=Path('/root/autodl-tmp/Quest/web/trace-viewer')

def get_json(url, timeout=4):
 with urllib.request.urlopen(url,timeout=timeout) as response:return json.load(response)

print('waiting for SSH-forwarded 4B model',flush=True)
while True:
 try:
  models=get_json('http://127.0.0.1:18084/v1/models')
  if any(x.get('id')=='glasseek-rl-4b' for x in models.get('data',[])):
   metric=urllib.request.urlopen('http://127.0.0.1:18084/metrics',timeout=5).read().decode()
   if 'vllm:num_preemptions_total' not in metric:raise RuntimeError('vLLM preemption metric missing')
   break
 except Exception as e:
  print('not ready',type(e).__name__,flush=True)
 time.sleep(20)

# Publish a distinct workspace; the Quest importer watches this data manifest.
index=SITE/'datasets/index.json'; manifest=json.loads(index.read_text(encoding='utf8'))
if not any(item.get('id')==ID for item in manifest['datasets']):
 manifest['datasets'].append({'id':ID,'label':'Glasseek RL-4B · BrowseComp · Quest 128',
   'description':'SSH 44 · RL-4B · TP1 · BF16 KV · DeepSeek Flash；同一 128 题',
   'folder':f'datasets/{ID}','data':f'datasets/{ID}/traces.json','benchmark':'BrowseComp','visible':True})
 tmp=index.with_name('.index-glasseek-rl4.tmp');tmp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(tmp,index)
env=os.environ.copy();env.update(GLASSEEK_VIEWER_RUN=RUN,GLASSEEK_VIEWER_MODEL='Glasseek RL-4B',GLASSEEK_VIEWER_LABEL='Glasseek RL-4B · BrowseComp · Quest 128')
output=SITE/'datasets'/ID/'traces.json'
viewer=subprocess.Popen([str(R/'venv/bin/python'),'-u',str(ROOT/'quest_viewer_adapter.py'),'--output',str(output),'--interval','10'],env=env,stdout=open(R/'viewer-publisher-rl4-ssh44.log','ab',buffering=0),stderr=subprocess.STDOUT,start_new_session=True)
(R/'viewer-publisher-rl4-ssh44.pid').write_text(str(viewer.pid))
print('viewer publisher',viewer.pid,flush=True)
script=str(R/'services/run_rl4_ssh44_eval.sh')
print('starting 128-question evaluation',flush=True)
completed=subprocess.run([script,'full'],check=False)
print('inference process returned',completed.returncode,flush=True)
# Judge successful predictions only after every requested case has a final record.
summary=ROOT/'output/preds'/RUN/'quest_qwen27b_128/summary.jsonl'
if summary.exists():
 records=[json.loads(line) for line in summary.read_text().splitlines() if line.strip()]
 ids={row.get('question_id') for row in records if row.get('status') in ('success','failed','policy_violation')}
 if len(ids)==128:
  print('all cases terminal; starting judge',flush=True)
  judged=subprocess.run([script,'judge'],check=False)
  print('judge process returned',judged.returncode,flush=True)
  sys.exit(judged.returncode)
print('run incomplete; judge not started',flush=True)
sys.exit(completed.returncode or 1)
