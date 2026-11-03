"""Resume both evaluations only after their real external dependencies recover."""
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path('/root/autodl-tmp/Glasseek/Evaluation')
R = ROOT / 'data/runtime'
STATE = R / 'resume-both-state.json'


def get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def check():
    problems = []
    for port, model in [(18080, 'glasseek-sft-9b'), (18082, 'glasseek-sft-9b'),
                        (18085, 'glasseek-sft-9b'), (18086, 'glasseek-sft-9b'), (18084, 'glasseek-rl-4b')]:
        try:
            if model not in {x.get('id') for x in get(f'http://127.0.0.1:{port}/v1/models').get('data', [])}:
                problems.append(f'model {port} unavailable')
        except Exception:
            problems.append(f'model {port} unavailable')
    try:
        probe = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', 'Lab3090',
                                'docker info >/dev/null 2>&1'],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8)
        if probe.returncode:
            problems.append('Lab3090 SSH or Docker unavailable')
    except Exception:
        problems.append('Lab3090 SSH or Docker unavailable')
    try:
        payload = json.dumps({'url': 'https://example.com'}).encode()
        request = urllib.request.Request('http://127.0.0.1:18081/fetch', data=payload,
                                         headers={'Content-Type': 'application/json'})
        result = get(request, timeout=20)
        if not result.get('success') or not result.get('data'):
            problems.append('Jina functional fetch failed')
    except Exception:
        problems.append('Jina functional fetch failed')
    for port in (18083, 18213, 18215, 18217):
        try:
            get(f'http://127.0.0.1:{port}/health')
        except Exception:
            problems.append(f'service {port} unavailable')
    return problems


last = None
while True:
    problems = check()
    state = {'state': 'waiting' if problems else 'starting', 'problems': problems, 'time': time.time()}
    tmp = STATE.with_suffix('.tmp'); tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2)); os.replace(tmp, STATE)
    if problems != last:
        print('preflight:', problems or 'passed', flush=True); last = problems
    if not problems:
        break
    time.sleep(30)

# The 4B supervisor starts its own publisher. Remove the temporary publisher first.
for proc in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args = proc.read_bytes().split(b'\0')
        if (b'--output' in args and b'/root/autodl-tmp/Quest/web/trace-viewer/datasets/glasseek-rl4-browsecomp-128-c16/traces.json' in args
                and any(b'quest_viewer_adapter.py' in arg for arg in args)):
            os.kill(int(proc.parent.name), 15)
    except (OSError, ValueError):
        pass
children = {}
for name, script in [('9b', 'run_sft9b_16_and_judge.py'), ('4b', 'run_rl4_c16_supervisor.py')]:
    log = (R / f'resumed-{name}-supervisor.log').open('ab', buffering=0)
    process = subprocess.Popen([str(R/'venv/bin/python'), '-u', str(ROOT/'services/remote44'/script)],
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    children[name] = process.pid
    print('started', name, process.pid, flush=True)
state.update(state='started', processes=children, time=time.time())
tmp = STATE.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=2)); os.replace(tmp, STATE)
