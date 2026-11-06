"""Drain the finished baseline, then start isolated, verified checkpoint runs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

P = Path(__file__).resolve().parent
E = Path(os.environ.get('GLASSEEK_EVAL_ROOT', '/root/autodl-tmp/Glasseek/Evaluation'))
sys.path.insert(0, str(E))
from browse_comp_eval.endpoint_registry import registry_lock, load_registry, atomic_write_registry
from browse_comp_eval.judge import load_prediction_record, prediction_fingerprint

DATASET = 'browsecomp_rl9b_non_timeout_34'
BASE = E/'output/preds/glasseek_base9b_rl9b_subset'/DATASET
STATE = P/'checkpoint-supervisor-state.json'
steps = list(range(10,81,10))

def note(phase, **details):
    body={'phase':phase,'updated_at':time.time(),**details}
    tmp=STATE.with_suffix('.tmp'); tmp.write_text(json.dumps(body,indent=2)+'\n'); tmp.replace(STATE)
    print(json.dumps(body), flush=True)

def get_json(url, timeout=4):
    with urllib.request.urlopen(url,timeout=timeout) as response: return json.load(response)

def screen_exists(name):
    result=subprocess.run(['screen','-ls'],capture_output=True,text=True)
    return any(f'.{name}\t' in line for line in result.stdout.splitlines())

def baseline_done():
    if screen_exists('glasseek-base-supervisor') or screen_exists('glasseek-base-live-judge'): return False
    expected={json.loads(line)['id'] for line in (P/f'{DATASET}.jsonl').read_text().splitlines() if line.strip()}
    predictions={}
    for file in (BASE/'conv').glob('*/*.json'):
        try: data=json.loads(file.read_text())
        except (OSError, ValueError): continue
        if data.get('status') in ('success','failed','policy_violation') and data.get('attempt_consumed',True):
            rec=load_prediction_record(file)
            if rec: predictions[str(data.get('question_id'))]=prediction_fingerprint(rec)
    if not expected <= set(predictions): return False
    judged={}
    for file in (BASE/'judge_official').glob('*/*.json'):
        try: data=json.loads(file.read_text())
        except (OSError,ValueError): continue
        if data.get('status') in ('judged','inference_error','skipped_infer_failed','skipped_empty_prediction','policy_violation_incorrect') and data.get('pred_fingerprint') == predictions.get(str(data.get('question_id'))): judged[str(data.get('question_id'))]=data
    return expected <= set(judged)

def listening(port):
    with socket.socket() as s: return s.connect_ex(('127.0.0.1',port))==0

def launch_screen(name,command,log):
    import shlex
    line='exec '+shlex.join([str(x) for x in command])+' >> '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['screen','-dmS',name,'bash','-c',line],check=True)

def wait_ready(step, timeout=600):
    port=18100+step
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if not screen_exists(f'glasseek-eval-step{step}-vllm'): raise RuntimeError(f'step {step} vLLM exited; see log')
        try:
            data=get_json(f'http://127.0.0.1:{port}/v1/models')
            entries=[x for x in data['data'] if x['id']==f'step{step}']
            if len(entries)!=1 or Path(entries[0]['root']).resolve()!=(P/f'adapters/step_{step}').resolve() or entries[0].get('parent')!='glasseek-base-9b':
                raise RuntimeError(f'step {step} adapter model metadata mismatch')
            req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',data=json.dumps({'model':f'step{step}','prompt':'Hello','temperature':0,'max_tokens':1}).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=90) as r: result=json.load(r)
            if result.get('model')!=f'step{step}': raise RuntimeError(f'step {step} response model mismatch')
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError): time.sleep(5)
    raise RuntimeError(f'step {step} startup timeout')

def gpu_free():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    values=[tuple(int(i.strip()) for i in l.split(',')) for l in output.splitlines()]
    return len(values)==8 and all(used<1024 for _,used in values)

def main():
    note('waiting_baseline')
    while not baseline_done(): time.sleep(15)
    note('baseline_complete')
    if shutil.disk_usage('/root/autodl-tmp').free < 5*1024**3: raise RuntimeError('less than 5 GiB free')
    for step in steps:
        if listening(18100+step): raise RuntimeError(f'port {18100+step} occupied')
        for file in (P/f'adapters/step_{step}/adapter_config.json',P/f'adapters/step_{step}/adapter_model.safetensors'):
            if not file.is_file(): raise RuntimeError(f'missing {file}')
    # Do not interrupt any outstanding model requests.
    for port in (18080,18085,18086,18087,18089,18090,18102,18105):
        if listening(port):
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics',timeout=5) as r: metrics=r.read().decode()
            for line in metrics.splitlines():
                if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{')) and float(line.split()[-1])!=0:
                    raise RuntimeError(f'active requests on {port}: {line}')
    registry=P/'base-registry.json'
    with registry_lock(registry):
        data=load_registry(registry); data['models'].pop('glasseek-base-9b',None); atomic_write_registry(registry,data)
    for name in ('glasseek-base-0','glasseek-base-1','glasseek-base-3','glasseek-base-4','glasseek-base-6','glasseek-base-7','glasseek-step10-gpu2','glasseek-step10-gpu5'):
        if screen_exists(name): subprocess.run(['screen','-S',name,'-X','stuff','\003'],check=True)
    for _ in range(60):
        if gpu_free(): break
        time.sleep(5)
    else: raise RuntimeError('GPU memory still occupied after drain; refusing to start')
    note('baseline_drained')
    # Step 10 must pass the original-environment comparison on GPU 0 first.
    launch_screen('glasseek-eval-step10-vllm',['bash',P/'start_checkpoint_replica.sh','0','18110','10'],P/'step10-vllm.log')
    wait_ready(10)
    subprocess.run([sys.executable,str(P/'verify_step10_service.py'),'--port','18110','--report',str(P/'step10_gpu0_verification.json')],check=True)
    note('step10_verified')
    for step in steps[1:]:
        launch_screen(f'glasseek-eval-step{step}-vllm',['bash',P/'start_checkpoint_replica.sh',str(step//10-1),str(18100+step),str(step)],P/f'step{step}-vllm.log')
    for step in steps[1:]: wait_ready(step)
    manifest=[]
    for step in steps:
        adapter=P/f'adapters/step_{step}'
        h=hashlib.sha256()
        with (adapter/'adapter_model.safetensors').open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
        registry={'version':1,'models':{f'step{step}':[{'base_url':f'http://127.0.0.1:{18100+step}/v1','status':'unknown','last_checked_at':None,'last_healthy_at':None,'unreachable_since':None,'last_error':None}]}}
        (P/f'step{step}-registry.json').write_text(json.dumps(registry,indent=2)+'\n')
        manifest.append({'step':step,'gpu':step//10-1,'port':18100+step,'model':f'step{step}','adapter':str(adapter),'sha256':h.hexdigest(),'max_agents':8,'max_rounds':200,'total_timeout':None,'tp':1,'max_context':131072,'kv_cache_dtype':'fp8'})
    (P/'checkpoint-eval-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for step in steps:
        name=f'glasseek_step{step}_rl9b_subset'
        judge_env=['env','HTTPS_PROXY=http://127.0.0.1:18200','HTTP_PROXY=http://127.0.0.1:18200','NO_PROXY=localhost,127.0.0.1','no_proxy=localhost,127.0.0.1']
        launch_screen(f'glasseek-eval-step{step}-judge',judge_env+[str(E/'data/runtime/venv/bin/python'),'-u',P/'async_judge.py','--run',name],P/f'step{step}-judge.log')
        viewer=f'glasseek-step{step}-rl9b-34'
        viewer_env=['env',f'GLASSEEK_VIEWER_RUN={name}',f'GLASSEEK_VIEWER_DATASET={DATASET}',f'GLASSEEK_VIEWER_WORKSPACE={viewer}',f'GLASSEEK_VIEWER_MODEL=Qwen3.5-9B step{step}',f'GLASSEEK_VIEWER_LABEL=Qwen3.5-9B step{step} BrowseComp 34']
        launch_screen(f'glasseek-eval-step{step}-viewer',viewer_env+[str(E/'data/runtime/venv/bin/python'),'-u',E/'quest_viewer_checkpoint_adapter.py','--output',P.parent.parent/'viewer'/f'{viewer}.json','--interval','10'],P/f'step{step}-viewer.log')
    for step in steps:
        launch_screen(f'glasseek-eval-step{step}-run',['bash',P/'run_checkpoint_eval.sh',str(step)],P/f'step{step}-eval.log')
    note('checkpoints_running',manifest=str(P/'checkpoint-eval-manifest.json'))
    failed = {}
    while True:
        running = []
        for step in steps:
            run_name = f'glasseek-eval-step{step}-run'
            if not screen_exists(run_name): continue
            running.append(step)
            try:
                registry = load_registry(P/f'step{step}-registry.json')
                models = registry['models']
                entries = models.get(f'step{step}', [])
                if set(models) != {f'step{step}'} or len(entries) != 1 or entries[0]['base_url'].rstrip('/') != f'http://127.0.0.1:{18100+step}/v1':
                    raise ValueError('registry does not match checkpoint route')
                models = get_json(f'http://127.0.0.1:{18100+step}/v1/models')
                matched = [x for x in models['data'] if x['id']==f'step{step}']
                if len(matched)!=1 or Path(matched[0]['root']).resolve() != (P/f'adapters/step_{step}').resolve():
                    raise ValueError('server adapter does not match checkpoint')
            except (urllib.error.URLError,TimeoutError,ConnectionError):
                # Endpoint watchdog handles temporary connection failures.
                continue
            except Exception as exc:
                failed[str(step)] = str(exc)
                subprocess.run(['screen','-S',run_name,'-X','stuff','\003'],check=False)
                print(f'stopped step {step}: {exc}',flush=True)
        note('checkpoints_running' if running else 'checkpoints_finished',running=running,failures=failed)
        if not running: break
        time.sleep(15)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        note('blocked',error=str(exc)); raise
