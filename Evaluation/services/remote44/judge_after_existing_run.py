"""Judge an existing evaluation after its supervisor exits and all cases finish."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/Glasseek/Evaluation')
RUN = 'glasseek_sft9b_quest128_gpu_embedding_c10_browsecomp'
DATASET = 'quest_qwen27b_128'
SUPERVISOR_PID = 109516
SUPERVISOR_START = '778835332'
RUNTIME = ROOT / 'data/runtime'


def supervisor_alive():
    try:
        stat = Path(f'/proc/{SUPERVISOR_PID}/stat').read_text().split(') ', 1)[1].split()
        return stat[0] != 'Z' and stat[19] == SUPERVISOR_START
    except (FileNotFoundError, IndexError):
        return False


def complete(dataset, summary):
    expected = {json.loads(line)['id'] for line in dataset.read_text().splitlines() if line.strip()}
    if not summary.exists():
        return False, len(expected), 0
    latest = {}
    for line in summary.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record.get('question_id')] = record.get('status')
    terminal = {key for key, status in latest.items() if status in {'success', 'failed', 'policy_violation'}}
    return bool(expected) and expected <= terminal, len(expected), len(expected & terminal)


def main():
    with (RUNTIME / 'sft9b-auto-judge.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('another judge watcher is already running', flush=True)
            return 1
        print(f'waiting for supervisor {SUPERVISOR_PID}', flush=True)
        while supervisor_alive():
            time.sleep(15)
        dataset = RUNTIME / f'{DATASET}.jsonl'
        summary = ROOT / 'output/preds' / RUN / DATASET / 'summary.jsonl'
        done, total, terminal = complete(dataset, summary)
        print(f'inference stopped: terminal={terminal}/{total}', flush=True)
        if not done:
            print('inference incomplete; judge not started', flush=True)
            return 1
        print('starting judge; existing matching results will be reused', flush=True)
        result = subprocess.run([str(RUNTIME / 'services/run_gpu_embedding_eval_10_browsecomp.sh'), 'judge'], check=False)
        print(f'judge exited {result.returncode}', flush=True)
        return result.returncode


if __name__ == '__main__':
    sys.exit(main())
