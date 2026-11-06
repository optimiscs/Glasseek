"""Three fresh, sequential 128-question evaluations with identical conditions."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

D = Path(__file__).resolve().parent
P = D.parents[1]
E = P / 'Evaluation'
PY = str(E / 'data/runtime/venv/bin/python')
VLLM = '/root/autodl-tmp/Quest/workspace/env/bin/vllm'
SITE = Path('/root/autodl-tmp/Quest/web/trace-viewer')
DATASET = 'quest_qwen27b_128'
PORTS = (19139, 19239)
MODELS = [
    ('step39', P/'analysis/clean2899-full64k/step39-hf'),
    ('step76', P/'analysis/full64k-reeval/step76-hf'),
    ('sft9b', E/'data/runtime/models/Glasseeker-9B-SFT'),
]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
children = []
stop_requested = False


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def status(state, **kw):
    value = dict(state=state, updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **kw)
    atomic(D/'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def spawn(args, log, env=None):
    with open(log, 'a') as f:
        child = subprocess.Popen(args, cwd=P, env=env, stdout=f, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    children.append(child)
    return child


def terminate(child, sig=signal.SIGTERM, timeout=90):
    if child.poll() is not None:
        return
    os.killpg(child.pid, sig)
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=30)


def interrupt(*_):
    global stop_requested
    stop_requested = True


def ensure_running():
    if stop_requested:
        raise InterruptedError('Pause requested; existing trajectories preserved')


def get(port, path, body=None, timeout=15):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}',
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type': 'application/json'})
    return OPENER.open(req, timeout=timeout).read()


def names(label):
    return f'fair120_20261106_{label}', f'fair120-20261106-{label}', f'fair120-{label}'


def stage_env(label):
    run, workspace, model = names(label)
    env = os.environ.copy()
    env.update(GLASSEEK_VIEWER_RUN=run, GLASSEEK_VIEWER_WORKSPACE=workspace,
               GLASSEEK_VIEWER_MODEL=model, GLASSEEK_VIEWER_DATASET=DATASET,
               GLASSEEK_VIEWER_LABEL=f'{label} · 公平对比128题 · 工具120次 · 64并发/卡',
               GLASSEEK_MAX_TOOL_CALLS='120', GLASSEEK_RESUME_INTERRUPTED='1')
    return env


def publish(env):
    workspace = env['GLASSEEK_VIEWER_WORKSPACE']
    output = SITE/'datasets'/workspace/'traces.json'
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic(output, dict(generatedAt=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            run=dict(name=env['GLASSEEK_VIEWER_LABEL'], model=env['GLASSEEK_VIEWER_MODEL'],
                     benchmark='BrowseComp 128', requestedCases=128, totalCases=0,
                     judgePassed=0, judgeScored=0, inProgress=False), cases=[]))
    with (D/'viewer.log').open('a') as f:
        subprocess.run([PY, str(E/'quest_viewer_checkpoint_adapter.py'), '--once',
            '--output', str(output)],
            cwd=P, env=env, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=120)
        subprocess.run(['/root/autodl-tmp/Quest/workspace/viewer/venv/bin/quest-viewer',
            'import', '--site', 'trace-viewer'], cwd=SITE.parent,
            env=env, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=120)


def summarize():
    rows = []
    scored = {}
    for label, _ in MODELS:
        run, _, _ = names(label)
        root = E/'output/preds'/run/DATASET
        records = {r['question_id']: r for p in (root/'conv').glob('*/*.json')
                   for r in [json.loads(p.read_text())]}
        judges = {r['question_id']: r for p in (root/'judge_official').glob('*/*.json')
                  for r in [json.loads(p.read_text())]}
        finished = {q:r for q,r in records.items() if r.get('status') in
                    ('success', 'failed', 'policy_violation') and r.get('attempt_consumed') is not False}
        good = {q:r for q,r in finished.items() if r['status']=='success'}
        scores = {q:judges[q]['score'] for q in good if judges.get(q,{}).get('status')=='judged'}
        scored[label] = scores
        durations = [r.get('duration_seconds') or r.get('wall_seconds') for r in good.values()]
        durations = [x for x in durations if isinstance(x,(int,float))]
        budgets = []
        for r in finished.values():
            b = E/'data/runtime/hermes/profiles'/r['profile']/'tool_call_budget.json'
            if b.exists():
                budgets.append(json.loads(b.read_text())['used'])
        complete = len(finished)==128 and len(scores)==len(good)
        rows.append(dict(model=label, run=run, finished=len(finished), scored=len(scores),
            correct=sum(v==1 for v in scores.values()), wrong=sum(v==0 for v in scores.values()),
            runtime_errors=len(finished)-len(good), awaiting_judge=len(good)-len(scores),
            accuracy=sum(v==1 for v in scores.values())/128 if complete else None,
            mean_duration_seconds=statistics.mean(durations) if durations else None,
            mean_executed_tool_calls=statistics.mean(budgets) if budgets else None,
            max_executed_tool_calls=max(budgets) if budgets else None,
            tool_budget_reached=sum(x==120 for x in budgets)))
    paired = []
    for a,b in [('step39','step76'),('step39','sft9b'),('step76','sft9b')]:
        qs=scored[a].keys() & scored[b].keys()
        paired.append(dict(models=[a,b], both_scored=len(qs),
            both_correct=sum(scored[a][q]==scored[b][q]==1 for q in qs),
            first_only=sum(scored[a][q]==1 and scored[b][q]==0 for q in qs),
            second_only=sum(scored[a][q]==0 and scored[b][q]==1 for q in qs)))
    atomic(D/'comparison.json', dict(updated=time.time(), runs=rows, paired=paired,
        denominator=128, note='One fresh run per model. Errors are separate; final accuracy uses all 128 questions.'))
    return rows


def capture_metrics(label):
    snap = {'time':time.time(), 'model':label}
    for port in PORTS:
        try:
            # Preserve counters and histograms so rates and long-context preemption
            # can be calculated without relying on vLLM log sampling.
            snap[str(port)] = get(port, '/metrics').decode()
        except Exception as e:
            snap[str(port)] = {'error':str(e)}
    snap['gpu'] = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'], text=True)
    with (D/f'metrics-{label}.jsonl').open('a') as f:
        f.write(json.dumps(snap)+'\n')


def verify_protocol(manifest):
    for path, digest in manifest['protocol_files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Protocol changed during comparison: {path}')


def stage(label, model_path, manifest):
    verify_protocol(manifest)
    env=stage_env(label)
    run, workspace, served=names(label)
    if (D/f'{label}.complete').exists():
        return
    status('loading', model=label, per_gpu_concurrency=64, global_concurrency=128)
    servers=[]
    for gpu,port in enumerate(PORTS):
        ve=env.copy()
        ve.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
            PATH='/root/miniconda3/envs/ms-swift/bin:'+os.environ['PATH'], CUDA_HOME='/usr/local/cuda',
            VLLM_CACHE_ROOT=str(D/f'vllm-cache-{gpu}'), TRITON_CACHE_DIR=str(D/f'triton-cache-{gpu}'))
        cmd=[VLLM,'serve',str(model_path),'--served-model-name',served,'--host','127.0.0.1',
            '--port',str(port),'--tensor-parallel-size','1','--max-model-len','131072',
            '--dtype','bfloat16','--gpu-memory-utilization','0.92','--kv-cache-dtype','fp8',
            '--max-num-seqs','64','--reasoning-parser','qwen3','--enable-auto-tool-choice',
            '--tool-call-parser','qwen3_coder','--enable-prompt-tokens-details',
            '--default-chat-template-kwargs','{"enable_thinking":true}',
            '--generation-config','vllm','--seed','42',
            '--override-generation-config','{"temperature":1.0,"top_p":1.0,"top_k":-1,"min_p":0.0,"repetition_penalty":1.0}']
        atomic(D/f'{label}-server-{gpu}-command.json',cmd)
        servers.append(spawn(cmd,D/f'{label}-vllm-{gpu}.log',ve))
    deadline=time.monotonic()+1800
    ready=set()
    while len(ready)<2:
        ensure_running()
        if any(p.poll() is not None for p in servers):
            raise RuntimeError('vLLM exited while loading; inspect stage logs')
        for port in PORTS:
            if port in ready: continue
            try: r=json.loads(get(port,'/v1/models'))
            except Exception: continue
            if [x['id'] for x in r['data']] != [served]:
                raise RuntimeError(f'Model routing mismatch on port {port}')
            ready.add(port)
        if time.monotonic()>deadline: raise TimeoutError('vLLM readiness timeout')
        if len(ready)<2:time.sleep(5)
    for port in PORTS:
        smoke=json.loads(get(port,'/v1/chat/completions',dict(model=served,
            messages=[dict(role='user',content='What is 2 + 2?')],max_tokens=128,
            temperature=0,chat_template_kwargs={'enable_thinking':True}),timeout=180))
        if smoke.get('model')!=served or not smoke.get('choices'):
            raise RuntimeError('Generation smoke check failed')
        atomic(D/f'{label}-smoke-{port}.json',smoke)
    subprocess.run([sys.executable,str(P/'analysis/clean2899-full64k/preflight.py')],check=True,env=env)
    judge_stop=D/f'{label}-judge.stop'
    judge_stop.unlink(missing_ok=True)
    judge=spawn([PY,str(P/'analysis/checkpoint_reeval/async_judge.py'),'--run',run,
        '--dataset',DATASET,'--handoff-stop',str(judge_stop)],D/f'{label}-judge.log',env)
    cmd=['bash',str(E/'scripts/run_eval.sh'),'--model',served,
        '--base-urls',','.join(f'http://127.0.0.1:{p}/v1' for p in PORTS),
        '--workers-per-endpoint','64','--dataset',DATASET,'--data-path',str(D/'questions.jsonl'),
        '--expected-questions','128','--repeats','1','--save-name',run,'--numbered-output',
        '--tool-whitelist','web,code_execution,delegation,vision','--enable-custom-system-prompt','true',
        '--custom-system-prompt-file',str(D/'system_prompt.txt'),'--max-rounds','200',
        '--max-retries','1','--max-tokens','16384','--temperature','1.0','--context-length','131072',
        '--fetch-server-url','http://127.0.0.1:18081','--fetch-watchdog-target','https://example.com']
    atomic(D/f'{label}-eval-command.json',cmd)
    evaluation=spawn(cmd,D/f'{label}-eval.log',env)
    status('evaluating',model=label,run=run,per_gpu_concurrency=64,global_concurrency=128)
    while evaluation.poll() is None:
        ensure_running()
        if any(p.poll() is not None for p in servers):
            raise RuntimeError('Inference server exited; pause without silently reducing concurrency')
        if judge.poll() is not None:raise RuntimeError('Asynchronous judge exited unexpectedly')
        capture_metrics(label)
        summarize()
        try:publish(env)
        except Exception as e:print(f'Viewer publish retry: {e}',flush=True)
        for _ in range(6):
            if evaluation.poll() is not None:break
            ensure_running();time.sleep(5)
    if evaluation.returncode not in (0,1):
        raise RuntimeError(f'Evaluation interrupted, exit={evaluation.returncode}')
    row=next(x for x in summarize() if x['model']==label)
    if row['finished']!=128:
        raise RuntimeError(f'Only {row["finished"]}/128 terminal records; preserve and pause')
    status('final_judging',model=label)
    # Give the asynchronous judge a chance to pick up the final records.
    deadline=time.monotonic()+600
    while row['awaiting_judge'] and time.monotonic()<deadline:
        ensure_running();time.sleep(5)
        row=next(x for x in summarize() if x['model']==label)
    judge_stop.touch()
    try:judge.wait(timeout=180)
    except subprocess.TimeoutExpired:terminate(judge)
    if row['awaiting_judge']:
        raise RuntimeError('Unscored successful trajectories remain; inspect judge log')
    publish(env)
    for server in servers:terminate(server)
    (D/f'{label}.complete').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    status('model_complete',model=label,summary=row)


def main():
    signal.signal(signal.SIGINT,interrupt)
    signal.signal(signal.SIGTERM,interrupt)
    with (D/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=json.loads((D/'manifest.json').read_text())
        try:
            for label,model in MODELS:stage(label,model,manifest)
            status('complete',comparison=summarize())
        except BaseException as e:
            status('paused',error=str(e))
            raise
        finally:
            for child in reversed(children):
                terminate(child,signal.SIGINT if 'run_eval.sh' in str(child.args) else signal.SIGTERM)


if __name__=='__main__':main()
