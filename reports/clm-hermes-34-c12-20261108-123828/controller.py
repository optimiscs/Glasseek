"""Own one evaluation, incremental judge, metrics sampler and comparison report."""
import hashlib,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path('/root/autodl-tmp/Glasseek');OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'Evaluation/services'),str(ROOT/'scripts/trajectory_benchmark')]
from eval_run import build
from clm_comparison import atomic,build_report

state=json.loads((OUT/'status.json').read_text())
run=state['runs']['clm']['run'];dataset=state['dataset']
settings=dict(dataset=dataset,run=run,workers_per_endpoint=12,expected_questions=34,
    max_rounds=200,timeout_seconds=3600,context_length=131072,max_tokens=16384,max_retries=1,repeats=1)
os.environ['GLASSEEK_URL_MODE']='raw';os.environ['GLASSEEK_CONTEXT_MANAGER']='clm'
argv,env=build('base9b','full',settings)
argv[argv.index('--data-path')+1]=str(OUT/'dataset.jsonl')
argv+=['--context-manager','clm','--url-mode','raw']
env['PYTHONPATH']=str(ROOT/'Evaluation_backend')+os.pathsep+env.get('PYTHONPATH','')
# Refuse to silently share the model with an older evaluation.
preflight={}
for port,path in [(19109,'/v1/models'),(18083,'/health'),(18081,'/health'),(18213,'/health'),(18215,'/health'),(18217,'/health'),(6080,'/api/v1/health/ready')]:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}',timeout=8) as response:
        preflight[str(port)]=json.load(response)
if 'glasseek-base-9b' not in [d['id'] for d in preflight['19109']['data']]:
    raise RuntimeError('Unexpected served model')
atomic(OUT/'preflight.json',preflight)
with (OUT/'eval.log').open('a') as log:
    proc=subprocess.Popen(argv,cwd=ROOT/'Evaluation',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
judge_argv,judge_env=build('base9b','watch-judge',settings)
judge_argv+=['--parent-pid',str(proc.pid),'--viewer-model','glasseek-base-9b']
with (OUT/'judge.log').open('a') as log:
    judge=subprocess.Popen(judge_argv,cwd=ROOT/'Evaluation',env=judge_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
state.update(status='running',started_at=time.time())
state['runs']['clm'].update(pid=proc.pid,status='running',argv=argv,
    judge=dict(pid=judge.pid,status='running'))
atomic(OUT/'status.json',state)
with (OUT/'monitor.log').open('a') as log:
    monitor=subprocess.Popen([sys.executable,str(ROOT/'scripts/trajectory_benchmark/monitor_url_modes.py'),'--output',str(OUT)],
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(OUT/'monitor.pid').write_text(str(monitor.pid))
(OUT/'viewer-url.txt').write_text('http://127.0.0.1:6080/trace-viewer-zh/?dataset='+run)
(OUT/'tool-dashboard-url.txt').write_text('http://127.0.0.1:6080/tool-dashboard.html?dataset='+run+'&follow=1')
expected=json.loads((OUT/'implementation_sha256.json').read_text());drift=set()
try:
    while proc.poll() is None or judge.poll() is None:
        for name,sha in expected.items():
            try:actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            except OSError:actual=None
            if actual!=sha:drift.add(name)
        state['source_drift']=sorted(drift)
        for item,child in [(state['runs']['clm'],proc),(state['runs']['clm']['judge'],judge)]:
            code=child.poll()
            if code is not None and item['status']=='running':
                item.update(status='completed' if code==0 else 'failed',exit_code=code,finished_at=time.time())
        atomic(OUT/'status.json',state)
        try:
            report=build_report(OUT)
            counts=report['current_naturally_terminal']
            state['progress']={k:counts[k] for k in ('cases','success','failed','scored','correct','context_edits')}
            atomic(OUT/'status.json',state)
        except Exception as exc:
            print('Report refresh:',repr(exc),flush=True)
        time.sleep(10)
    state.update(status='completed' if proc.returncode==0 and judge.returncode==0 else 'completed_with_errors',finished_at=time.time())
    atomic(OUT/'status.json',state)
    report=build_report(OUT)
    atomic(OUT/'report.json',report)
    print(json.dumps({'status':state['status'],'summary':report['current_naturally_terminal']}),flush=True)
finally:
    if monitor.poll() is None:monitor.terminate()
