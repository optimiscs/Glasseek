"""Conservative online concurrency ramp; preserve evaluation sessions at transitions."""
import collections,fcntl,json,os,re,shutil,signal,subprocess,time,urllib.request,sys
from pathlib import Path
P=Path('/root/autodl-tmp/Glasseek');D=P/'experiments/clean2899-full64k';R=D/'concurrency-ramp';DRIVER=str(D/'concurrency16/run_repeats.sh');PREFIX='glasseek_clean2899_full64k_step39_128'
lock=(R/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
open_url=urllib.request.build_opener(urllib.request.ProxyHandler({})).open

def state(stage,**kw):
 data={'at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'stage':stage,**kw};(R/'status.json').write_text(json.dumps(data,indent=2));print(json.dumps(data),flush=True)

def processes():
 for f in Path('/proc').glob('[0-9]*/cmdline'):
  try:yield int(f.parent.name),f.read_bytes().decode(errors='replace').strip('\0').split('\0')
  except (OSError,ValueError):continue

def evals():return [pid for pid,a in processes() if 'browse_comp_eval.run' in a and any(v.startswith(PREFIX) for v in a)]
def drivers():return [pid for pid,a in processes() if len(a)>1 and Path(a[0]).name=='bash' and a[1]==DRIVER]

def stop_evaluation():
 for pid in evals():os.kill(pid,signal.SIGINT)
 deadline=time.monotonic()+180
 while evals() or drivers():
  if time.monotonic()>deadline:raise RuntimeError('Graceful evaluator stop timed out; leaving services intact')
  time.sleep(2)
 # The shell EXIT trap terminates publishers; allow their last children to release flock.
 time.sleep(3)

def snapshot(tag):
 out=R/tag;out.mkdir(exist_ok=True)
 for run in (P/'Evaluation/output/preds').glob(PREFIX+'*'):
  source=run/'quest_qwen27b_128'/'conv'
  if source.exists():shutil.copytree(source,out/run.name/'conv',dirs_exist_ok=True)
 budgets={}
 for f in (P/'Evaluation/data/runtime/hermes/profiles').glob(PREFIX+'*/tool_call_budget.json'):budgets[f.parent.name]=json.loads(f.read_text())
 (out/'budgets.json').write_text(json.dumps(budgets));return out

def start_stage(n):
 stop_evaluation();snapshot(f'preserved-before-{n}-{int(time.time())}')
 for gpu in (0,1):subprocess.run(['screen','-S',f'clean2899-vllm-{gpu}','-X','quit'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 deadline=time.monotonic()+90
 while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip():
  if time.monotonic()>deadline:raise RuntimeError('GPU not released; refusing overlapping model launch')
  time.sleep(2)
 env=dict(os.environ,EVAL_PER_GPU_CONCURRENCY=str(n))
 for gpu,port in ((0,19139),(1,19239)):
  subprocess.run(['screen','-dmS',f'clean2899-vllm-{gpu}','bash','-c','exec bash "$1" "$2" "$3" > "$4" 2>&1','_',str(D/'start_model.sh'),str(gpu),str(port),str(R/f'vllm-c{n}-{gpu}.log')],env=env,check=True)
 state(n,phase='loading_models',per_gpu=n,global_concurrency=2*n)
 with (R/f'checks-c{n}.log').open('a') as log:subprocess.run(['python',str(D/'check_models.py')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1300)
 subprocess.run(['screen','-dmS','clean2899-avg3','bash','-c','exec bash "$1" >> "$2" 2>&1','_',DRIVER,str(D/'concurrency16/pipeline.log')],env=env,check=True)
 state(n,phase='evaluating',per_gpu=n,global_concurrency=2*n,tool_call_cap=120)
 deadline=time.monotonic()+180
 while not evals():
  if time.monotonic()>deadline:raise RuntimeError('Evaluator did not start after model readiness')
  time.sleep(5)

PAT=re.compile(r'^(vllm:[^ {]+)(?:\{(.*?)\})?\s+(\S+)$')
def sample():
 row={'time':time.time(),'engines':{}}
 for port in (19139,19239):
  raw=open_url(f'http://127.0.0.1:{port}/metrics',timeout=10).read().decode();v=collections.defaultdict(float)
  for line in raw.splitlines():
   m=PAT.match(line)
   if not m:continue
   k,labels,value=m.groups();labels=labels or '';num=float(value)
   direct={'vllm:num_preemptions_total':'preemptions','vllm:request_success_total':'completed','vllm:generation_tokens_total':'decode','vllm:prompt_tokens_total':'input','vllm:request_prompt_tokens_count':'prompt_count','vllm:kv_cache_usage_perc':'kv','vllm:num_requests_running':'running','vllm:num_requests_waiting':'waiting'}
   if k in direct:v[direct[k]]+=num
   if k=='vllm:prompt_tokens_by_source_total' and 'source="local_compute"' in labels:v['prefill']+=num
   if k=='vllm:request_prompt_tokens_bucket' and 'le="100000.0"' in labels:v['prompt_le100k']+=num
  v['long_requests']=v['prompt_count']-v['prompt_le100k'];row['engines'][str(port)]=dict(v)
 row['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();return row

def summary(rows):
 dt=rows[-1]['time']-rows[0]['time'];out={'seconds':dt,'engines':{}}
 for port in rows[0]['engines']:
  a,b=rows[0]['engines'][port],rows[-1]['engines'][port];v={key:b.get(key,0)-a.get(key,0) for key in ('preemptions','completed','long_requests')}
  v.update({key+'_tps':(b.get(key,0)-a.get(key,0))/dt for key in ('prefill','decode','input')});v['peak_kv']=max(r['engines'][port].get('kv',0) for r in rows);v['waiting_growth']=b.get('waiting',0)-a.get('waiting',0);out['engines'][port]=v
 out['preemptions']=sum(v['preemptions'] for v in out['engines'].values());out['completed']=sum(v['completed'] for v in out['engines'].values());out['long_requests']=sum(v['long_requests'] for v in out['engines'].values());out['preemption_per_completed']=out['preemptions']/max(1,out['completed']);return out

def finished():
 subprocess.run(['python',str(D/'concurrency16/avg3.py')],stdout=subprocess.DEVNULL,check=True)
 x=json.loads((D/'concurrency16/avg3.json').read_text());return all(v['pending']==0 and v['unscored']==0 for v in x['runs'])

n=32;previous=16;allow_escalation=True;stage_start=None;rows=[];errors=0
try:
 if '--observe-current32' not in sys.argv:start_stage(n)
 stage_start=time.monotonic()
 while True:
  try:r=sample();errors=0
  except Exception as exc:
   errors+=1;state(n,phase='metrics_unavailable',error=str(exc),consecutive=errors)
   if errors>=3:
    if n==previous:
     stop_evaluation();state(n,phase='paused_unstable_fallback',reason='metrics unavailable at fallback');break
    state(n,phase='rolling_back',reason='metrics unavailable',target=previous);start_stage(previous);n=previous;allow_escalation=False;rows=[];stage_start=time.monotonic();errors=0
   time.sleep(10);continue
  rows.append(r)
  with (R/f'metrics-c{n}.jsonl').open('a') as out:out.write(json.dumps(r)+'\n')
  elapsed=time.monotonic()-stage_start
  if len(rows)>=2:
   report=summary(rows);(R/f'summary-c{n}.json').write_text(json.dumps(report,indent=2))
  else:time.sleep(10);continue
  if finished():state(n,phase='all_three_repeats_complete',report=report);break
  engine_error=any(any(s in (R/f'vllm-c{n}-{g}.log').read_text(errors='replace') for s in ('CUDA out of memory','EngineDeadError','Engine core initialization failed')) for g in (0,1) if (R/f'vllm-c{n}-{g}.log').exists())
  bad=engine_error or (elapsed>=180 and report['preemptions']>=10 and report['preemption_per_completed']>.01)
  if bad:
   if n==previous:
    stop_evaluation();state(n,phase='paused_unstable_fallback',report=report);break
   state(n,phase='rolling_back',target=previous,report=report);start_stage(previous);n=previous;allow_escalation=False;rows=[];stage_start=time.monotonic();continue
  enough_long=report['long_requests']>=10 and all(v['long_requests']>=3 for v in report['engines'].values())
  stable=elapsed>=600 and enough_long and report['preemption_per_completed']<=.01 and all(v['peak_kv']<.90 and v['waiting_growth']<=n for v in report['engines'].values())
  state(n,phase='observing',elapsed_seconds=round(elapsed),enough_long_samples=enough_long,stable=stable,allow_escalation=allow_escalation,report=report)
  if stable and allow_escalation and n<64:
   previous=n;n=48 if n==32 else 64
   try:start_stage(n)
   except Exception as exc:
    state(n,phase='startup_failed_rolling_back',target=previous,error=str(exc));start_stage(previous);n=previous;allow_escalation=False
   rows=[];stage_start=time.monotonic();continue
  time.sleep(10)
except Exception as exc:
 state(n,phase='controller_error',error=str(exc));raise
