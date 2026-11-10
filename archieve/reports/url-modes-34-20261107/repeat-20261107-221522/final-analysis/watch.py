import json,hashlib,time,os,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/Glasseek'); out=root/'reports/url-modes-34-20261107/repeat-20261107-221522'
expected=json.loads((out/'implementation_sha256.json').read_text()); changed=set(); pid=int((out/'controller.pid').read_text())
while True:
 for name,sha in expected.items():
  try: actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
  except OSError: actual='missing'
  if actual!=sha and name not in changed:
   changed.add(name)
   with (out/'source-drift.jsonl').open('a') as f: f.write(json.dumps({'timestamp':time.time(),'path':name,'expected':sha,'actual':actual})+'\n')
 try:
  stat=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]
  if stat=='Z': break
 except OSError: break
 time.sleep(15)
try:
 state=json.loads((out/'status.json').read_text())
 if not all(x.get('status')!='running' and x.get('judge',{}).get('status')=='completed' for x in state['runs'].values()):
  print('Comparison deferred: incomplete evaluations or judges',flush=True);raise SystemExit(1)
 subprocess.run(['python',str(out/'final-analysis/analyze.py')],check=True)
 before=json.loads((root/'reports/url-modes-34-20261107/live-v3-full-c8/final-analysis/report.json').read_text())
 after=json.loads((out/'final-analysis/report.json').read_text())
 old={(r['mode'],r['question_id']):r for r in before['rows']}; new={(r['mode'],r['question_id']):r for r in after['rows']}; paired={}
 for mode in after['summary']:
  keys=[k for k in new if k[0]==mode and k in old]
  paired[mode]={'n':len(keys),'improved':[k[1] for k in keys if new[k]['correct'] and not old[k]['correct']],'regressed':[k[1] for k in keys if old[k]['correct'] and not new[k]['correct']]}
 comparison={'previous':before['summary'],'repeat':after['summary'],'paired':paired,'source_drift_detected':sorted(changed),'note':'Compare paired questions and review service outages and code drift before attributing causality.'}
 (out/'final-analysis/comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2))
 print('Comparison written',flush=True)
except Exception as exc: print('Comparison failed:',type(exc).__name__,str(exc),flush=True);raise
