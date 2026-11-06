"""Start baseline evaluation only after model and tool preflight succeeds."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

root = Path(os.environ.get('REEVAL_DIR', Path(__file__).resolve().parent))
eval_root = Path(os.environ.get('GLASSEEK_EVAL_ROOT', '/root/autodl-tmp/Glasseek/Evaluation'))
credentials_path = Path(os.environ.get('REEVAL_CREDENTIALS', '/root/autodl-tmp/Glasseek/checkpoint-reeval/private/eval-credentials.json'))
run_name = os.environ.get('REEVAL_RUN_NAME', 'glasseek_base9b_rl9b_subset')
deadline = time.monotonic() + 1800
ports = (18080, 18082, 18085, 18086, 18087, 18088, 18089, 18090,
         18083, 18081, 18217, 18218)
while time.monotonic() < deadline:
    if shutil.disk_usage('/root/autodl-tmp').free < 1024 ** 3 or shutil.disk_usage('/').free < 1024 ** 3:
        raise SystemExit('Storage below 1 GiB; evaluation not started')
    try:
        for port in ports:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as r:
                if r.status != 200:
                    raise RuntimeError(f'Unhealthy service {port}')
        subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=3', os.environ.get('HERMES_REMOTE_PURE_COMPUTE', 'Lab3090'), 'true'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=6)
        break
    except Exception:
        time.sleep(15)
else:
    raise SystemExit('Services did not become healthy within 30 minutes')

request = urllib.request.Request('http://127.0.0.1:18081/fetch',
    data=json.dumps({'url': 'https://example.com'}).encode(),
    headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=90) as response:
    if not json.load(response).get('success'):
        raise SystemExit('Fetch preflight failed; evaluation not started')
request = urllib.request.Request('http://127.0.0.1:18083/search',
    data=json.dumps({'q': 'example domain'}).encode(), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=60) as response:
    if response.status != 200:
        raise SystemExit('Search preflight failed; evaluation not started')
print('Preflight passed; launching 34-question baseline', flush=True)
with (root / 'base-evaluation.log').open('ab', buffering=0) as output:
    result = subprocess.run(['bash', str(root / 'run_base_eval.sh')], stdout=output, stderr=subprocess.STDOUT)
print(f'Evaluation process exited with status {result.returncode}', flush=True)
if result.returncode:
    raise SystemExit(result.returncode)
credentials = json.loads(credentials_path.read_text())
if not credentials.get('judge'):
    raise SystemExit('Judge credential missing; evaluation outputs preserved')
env = os.environ.copy()
env['JUDGE_API_KEY'] = credentials['judge']
env['HTTPS_PROXY'] = env['HTTP_PROXY'] = 'http://127.0.0.1:18200'
env['NO_PROXY'] = env['no_proxy'] = 'localhost,127.0.0.1'
print('Starting official answer judge', flush=True)
with (root / 'base-judge.log').open('ab', buffering=0) as output:
    result = subprocess.run([str(eval_root / 'scripts/judge.sh'),
        '--save-name', run_name, '--dataset', 'browsecomp_rl9b_non_timeout_34',
        '--judge-provider', 'openai', '--judge-base-url', 'https://api.deepseek.com/v1',
        '--judge-model', 'deepseek-flash', '--prompt-version', 'official', '--token-param', 'max_tokens',
        '--no-json-schema', '--temperature', '0'], cwd=eval_root,
        env=env, stdout=output, stderr=subprocess.STDOUT)
print(f'Judge exited with status {result.returncode}', flush=True)
raise SystemExit(result.returncode)
