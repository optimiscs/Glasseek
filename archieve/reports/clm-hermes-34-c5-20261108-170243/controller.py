"""Own the fresh c5 CLM repeat, incremental judge, viewer and metric reports."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/root/autodl-tmp/Glasseek')
OUT = Path(__file__).resolve().parent
ownership = (OUT/'controller.lock').open('a')
fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
sys.path[:0] = [str(ROOT/'Evaluation/services'), str(ROOT/'scripts/trajectory_benchmark')]
from eval_run import build
from clm_comparison import atomic, build_report
from compare_clm_to_hermes import freeze_baselines, update

state = json.loads((OUT/'status.json').read_text())
if state['status'] not in ('starting',):
    raise RuntimeError('This fresh-run controller has already started; use checkpoint resume deliberately')
run = state['runs']['clm']['run']
settings = dict(dataset=state['dataset'], run=run, workers_per_endpoint=5, expected_questions=34,
    max_rounds=120, max_tool_calls=120, timeout_seconds=None, context_length=131072,
    max_tokens=16384, max_retries=1, repeats=1,
    system_prompt=str(ROOT/'data/glasseek/sft-eval-fixed-128k/eval_system_body.txt'))
os.environ['GLASSEEK_URL_MODE'] = 'raw'
os.environ['GLASSEEK_CONTEXT_MANAGER'] = 'clm'
argv, env = build('base9b', 'full', settings)
argv[argv.index('--data-path')+1] = str(OUT/'dataset.jsonl')
argv += ['--context-manager', 'clm', '--url-mode', 'raw']
env['PYTHONPATH'] = str(ROOT/'Evaluation_backend')+os.pathsep+env.get('PYTHONPATH', '')
env['GLASSEEK_VIEWER_LABEL'] = 'CLM + Hermes · 34题重跑 · 并发5 · 累计120工具轮'
preflight = {}
for port, path in [(19109,'/v1/models'),(18083,'/health'),(18081,'/health'),
                   (18213,'/health'),(18215,'/health'),(18217,'/health'),(6080,'/api/v1/health/ready')]:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=8) as response:
        preflight[str(port)] = json.load(response)
if 'glasseek-base-9b' not in [d['id'] for d in preflight['19109']['data']]:
    raise RuntimeError('Unexpected served model')
atomic(OUT/'preflight.json', preflight)
if (ROOT/'Evaluation/output/preds'/run/state['dataset']).exists():
    raise RuntimeError('Fresh output already exists; refusing to reuse old results')
expected = json.loads((OUT/'implementation_sha256.json').read_text())
baselines = freeze_baselines(OUT)
with (OUT/'eval.log').open('a') as log:
    proc = subprocess.Popen(argv, cwd=ROOT/'Evaluation', env=env, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
judge_argv, judge_env = build('base9b', 'watch-judge', settings)
judge_argv += ['--parent-pid', str(proc.pid), '--viewer-model', 'glasseek-base-9b']
with (OUT/'judge.log').open('a') as log:
    judge = subprocess.Popen(judge_argv, cwd=ROOT/'Evaluation', env=judge_env,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
state.update(status='running', started_at=time.time(), controller_pid=os.getpid())
state['runs']['clm'].update(pid=proc.pid, status='running', argv=argv,
    judge=dict(pid=judge.pid, status='running'))
atomic(OUT/'status.json', state)
with (OUT/'monitor.log').open('a') as log:
    monitor = subprocess.Popen([sys.executable, str(ROOT/'scripts/trajectory_benchmark/monitor_url_modes.py'),
        '--output', str(OUT)], stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True)
(OUT/'monitor.pid').write_text(str(monitor.pid))
(OUT/'viewer-url.txt').write_text('http://localhost:60007/?dataset='+run)
(OUT/'tool-dashboard-url.txt').write_text('http://localhost:60007/tool-dashboard.html?dataset='+run+'&all=1')
drift = set()

def refresh():
    for name, sha in expected.items():
        try: actual = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        except OSError: actual = None
        if actual != sha: drift.add(name)
    state['source_drift'] = sorted(drift)
    for item, child in [(state['runs']['clm'], proc),(state['runs']['clm']['judge'], judge)]:
        code = child.poll()
        if code is not None:
            item.update(status='completed' if code == 0 else 'failed', exit_code=code)
            item.setdefault('finished_at', time.time())
    atomic(OUT/'status.json', state)
    report = build_report(OUT)
    counts = report['current_naturally_terminal']
    state['progress'] = {k:counts[k] for k in ('cases','success','failed','scored','correct','context_edits')}
    atomic(OUT/'status.json', state)
    update(OUT, baselines)
    return report

try:
    while proc.poll() is None or judge.poll() is None:
        try: refresh()
        except Exception as exc: print('Report refresh:', repr(exc), flush=True)
        time.sleep(10)
    state.update(status='completed' if proc.returncode == 0 and judge.returncode == 0 else 'completed_with_errors',
        finished_at=time.time())
    report = refresh()
    atomic(OUT/'report.json', report)
    print(json.dumps({'status':state['status'],'summary':state['progress']}), flush=True)
except KeyboardInterrupt:
    state.update(status='interrupted', finished_at=time.time())
    try: os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError: pass
    atomic(OUT/'status.json', state)
    raise
finally:
    if monitor.poll() is None: monitor.terminate()
