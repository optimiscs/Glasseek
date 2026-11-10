import os,time,json,urllib.request,subprocess
from pathlib import Path
ROOT=Path('/root/autodl-tmp/Glasseek');OUT=ROOT/'reports/url-modes-34-20261107/repeat-20261107-221522'
model_pid=int((ROOT/'Evaluation/data/runtime/tool-services/resume-model.pid').read_text())
for _ in range(360):
 try:
  with urllib.request.urlopen('http://127.0.0.1:19109/v1/models',timeout=3) as r:
   ids=[x['id'] for x in json.load(r)['data']]
   if 'glasseek-base-9b' not in ids:raise ValueError('unexpected model')
  for port,path in [(18083,'/health'),(18081,'/health'),(18213,'/health'),(18215,'/health'),(18217,'/health'),(6080,'/api/v1/health/ready')]:
   with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}',timeout=5) as r:
    if r.status!=200:raise ValueError(f'health failed: {port}')
  break
 except Exception as e:
  if not Path(f'/proc/{model_pid}').exists():raise RuntimeError('model server exited') from e
  time.sleep(5)
else:raise TimeoutError('model/service readiness timed out')
with (OUT/'resume-controller.log').open('a') as log:
 p=subprocess.Popen(['python',str(OUT/'resume_controller.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(OUT/'controller.pid').write_text(str(p.pid));print('resumed controller',p.pid,flush=True)
for _ in range(20):
 try:
  s=json.loads((OUT/'status.json').read_text())
  if s.get('resumed_at') and all(r.get('status')=='running' for r in s['runs'].values()):break
 except (OSError,ValueError):pass
 if p.poll() is not None:raise RuntimeError('resume controller exited')
 time.sleep(1)
else:raise RuntimeError('resume controller did not initialize')
for name,argv in [('monitor',['python',str(ROOT/'scripts/trajectory_benchmark/monitor_url_modes.py'),'--output',str(OUT)]),('analysis',['python',str(OUT/'final-analysis/watch.py')])]:
 with (OUT/(name+'.resume.log')).open('a') as log:
  child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (OUT/(name+'.pid')).write_text(str(child.pid));print(name,child.pid,flush=True)
