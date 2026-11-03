"""Wait for detached in-flight agents, then run 16-way eval and compare TPS."""
import datetime as dt
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from judge_after_existing_run import ROOT, RUNTIME

MANIFEST = RUNTIME / 'sft9b-concurrency-transition.json'
OUTPUT = RUNTIME / 'sft9b-concurrency-transition-monitor.json'
PATTERN = re.compile(r'INFO (\d\d-\d\d \d\d:\d\d:\d\d).*Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s')
LOGS = [RUNTIME / 'vllm-gpu0-observability.log', RUNTIME / 'vllm-gpu1-restored.log']
CALL = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+.*API call #\d+:.*?in=(\d+) out=(\d+)')


def alive(item):
    try:
        path = Path('/proc') / str(item['pid'])
        stat = (path / 'stat').read_text().split(') ', 1)[1].split()
        cmdline = (path / 'cmdline').read_bytes().split(b'\0')
        return stat[0] != 'Z' and item['profile'].encode() in cmdline
    except (OSError, IndexError):
        return False


def log_window(start, end):
    totals = []
    for path in LOGS:
        samples = []
        for line in path.read_text(errors='replace').splitlines():
            match = PATTERN.search(line)
            if not match:
                continue
            stamp = dt.datetime.strptime(f'{start.year}-{match[1]}', '%Y-%m-%d %H:%M:%S').astimezone()
            if start <= stamp < end:
                samples.append((float(match[2]), float(match[3])))
        totals.append({'log': str(path), 'samples': len(samples),
                       'input_tps': statistics.mean(x[0] for x in samples) if samples else None,
                       'output_tps': statistics.mean(x[1] for x in samples) if samples else None})
    return totals


def save(data):
    tmp = OUTPUT.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(OUTPUT)


def completed_calls(start, end):
    inputs = outputs = count = 0
    profiles = RUNTIME / 'hermes/profiles'
    for path in profiles.glob('glasseek_sft9b_quest128_gpu_embedding_c10_browsecomp*/logs/agent.log'):
        for line in path.read_text(errors='replace').splitlines():
            match = CALL.search(line)
            if match and start <= dt.datetime.strptime(match[1], '%Y-%m-%d %H:%M:%S').astimezone() < end:
                inputs += int(match[2]); outputs += int(match[3]); count += 1
    return {'completed_calls': count, 'input_tokens': inputs, 'output_tokens': outputs,
            'input_tps': inputs / 300, 'output_tps': outputs / 300,
            'note': 'Counts whole completed calls in the window; boundary-crossing calls are not prorated.'}


items = json.loads(MANIFEST.read_text())
data = {'phase': 'waiting_for_previous_agents', 'agents': items,
        'baseline_start': '2026-11-02T23:54:00+08:00',
        'baseline_end': '2026-11-02T23:59:00+08:00'}
baseline_start = dt.datetime.fromisoformat(data['baseline_start'])
baseline_end = dt.datetime.fromisoformat(data['baseline_end'])
data['baseline'] = log_window(baseline_start, baseline_end)
data['baseline_completed_calls'] = completed_calls(baseline_start, baseline_end)
save(data)
print('waiting for existing detached agents:', len(items), flush=True)
last = None
while True:
    running = [item for item in items if alive(item)]
    if len(running) != last:
        print('detached agents remaining:', len(running), flush=True)
        data['remaining'] = running
        save(data)
        last = len(running)
    if not running:
        break
    time.sleep(10)

print('old agents exited; starting 16-agent evaluation', flush=True)
log = (RUNTIME / 'sft9b-16-supervisor.log').open('ab', buffering=0)
process = subprocess.Popen([str(RUNTIME/'venv/bin/python'), '-u', str(ROOT/'services/remote44/run_sft9b_16_and_judge.py')],
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
data.update(phase='waiting_for_16_agent_scheduler', supervisor_pid=process.pid)
save(data)
ramp = ROOT/'output/preds/glasseek_sft9b_quest128_gpu_embedding_c10_browsecomp/quest_qwen27b_128/concurrency.jsonl'
offset = ramp.stat().st_size if ramp.exists() else 0
deadline = time.monotonic() + 180
while time.monotonic() < deadline:
    if process.poll() is not None:
        data.update(phase='startup_failed', exit_code=process.returncode); save(data); sys.exit(1)
    if ramp.exists():
        with ramp.open() as f:
            f.seek(offset)
            if any(json.loads(line).get('event') == 'baseline' and json.loads(line).get('limit') == 16 for line in f if line.strip()):
                break
    time.sleep(3)
else:
    data['phase']='startup_monitor_timeout'; save(data); sys.exit(1)

# Let workers and model requests start before collecting the fixed window.
time.sleep(30)
start = dt.datetime.now().astimezone()
data.update(phase='monitoring_5_minutes', monitor_start=start.isoformat())
save(data)
print('five-minute monitoring started', flush=True)
time.sleep(305)  # vLLM emits 10-second averages; allow the last interval to flush.
end = start + dt.timedelta(seconds=300)
data.update(phase='complete', monitor_end=end.isoformat(), after=log_window(start, end),
            after_completed_calls=completed_calls(start, end), supervisor_running=process.poll() is None)
before = data['baseline_completed_calls']
after = data['after_completed_calls']
data['completed_call_change_percent'] = {
    key: (after[key] / before[key] - 1) * 100 if before[key] else None
    for key in ('input_tps', 'output_tps')
}
data['vllm_log_change_percent'] = {}
for key in ('input_tps', 'output_tps'):
    previous = sum(item[key] or 0 for item in data['baseline'])
    current = sum(item[key] or 0 for item in data['after'])
    data['vllm_log_change_percent'][key] = (current / previous - 1) * 100 if previous else None
save(data)
print('monitor complete:', OUTPUT, flush=True)
