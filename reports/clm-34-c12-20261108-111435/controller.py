import os,sys,json,time,subprocess,urllib.request,hashlib
from pathlib import Path
ROOT=Path('/root/autodl-tmp/Glasseek');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'Evaluation/services'));from eval_run import build
run='clm_official_34_c12_'+OUT.name.split('c12-',1)[1].replace('-','_')
overrides={'dataset':'browsecomp_rl9b_non_timeout_34','run':run,'workers_per_endpoint':12,'expected_questions':34,'max_rounds':200,'timeout_seconds':3600,'context_length':131072,'max_tokens':16384,'repeats':1}
os.environ['GLASSEEK_URL_MODE']='raw';os.environ['GLASSEEK_CONTEXT_MANAGER']='clm'
argv,env=build('base9b','full',overrides);argv+=['--context-manager','clm','--url-mode','raw']
env['PYTHONPATH']=str(ROOT/'Evaluation_backend')+os.pathsep+env.get('PYTHONPATH','')
state={'status':'waiting_for_model','dataset':overrides['dataset'],'context_manager':'clm','url_mode':'raw','workers':12,'context_length':131072,'max_tokens':16384,'runs':{}}
def save():
 tmp=OUT/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(OUT/'status.json')
save()
for _ in range(240):
 try:
  with urllib.request.urlopen('http://127.0.0.1:19109/v1/models',timeout=3) as r:
   if 'glasseek-base-9b' not in [x['id'] for x in json.load(r)['data']]:raise ValueError('wrong model')
  for port,path in [(18083,'/health'),(18081,'/health'),(18213,'/health'),(18215,'/health'),(18217,'/health'),(6080,'/api/v1/health/ready')]:
   with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}',timeout=4) as r:assert r.status==200
  break
 except Exception:
  model_pid=int((OUT/'vllm.pid').read_text())
  if not Path(f'/proc/{model_pid}').exists():raise RuntimeError('vLLM exited')
  time.sleep(5)
else:raise TimeoutError('model readiness timed out')
state['started_at']=time.time();state['status']='running'
with (OUT/'eval.log').open('a') as f:proc=subprocess.Popen(argv,cwd=ROOT/'Evaluation',env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
judge_argv,judge_env=build('base9b','watch-judge',overrides);judge_argv+=['--parent-pid',str(proc.pid),'--viewer-model','glasseek-base-9b']
with (OUT/'judge.log').open('a') as f:judge=subprocess.Popen(judge_argv,cwd=ROOT/'Evaluation',env=judge_env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
state['runs']['clm']={'run':run,'pid':proc.pid,'status':'running','argv':argv,'judge':{'pid':judge.pid,'status':'running'}};save()
with (OUT/'monitor.log').open('a') as f:monitor=subprocess.Popen(['python',str(ROOT/'scripts/trajectory_benchmark/monitor_url_modes.py'),'--output',str(OUT)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(OUT/'monitor.pid').write_text(str(monitor.pid))
expected=json.loads((OUT/'implementation_sha256.json').read_text());drift=set()
while proc.poll() is None or judge.poll() is None:
 for name,sha in expected.items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:drift.add(name)
 for item,p in [(state['runs']['clm'],proc),(state['runs']['clm']['judge'],judge)]:
  code=p.poll()
  if code is not None and item['status']=='running':item.update(status='completed' if code==0 else 'failed',exit_code=code,finished_at=time.time())
 state['source_drift']=sorted(drift);save();time.sleep(5)
state['status']='completed' if proc.returncode==0 and judge.returncode==0 else 'completed_with_errors';state['finished_at']=time.time();save()
subprocess.run(['python',str(OUT/'analyze.py')],check=True)
