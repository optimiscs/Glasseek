"""Run 16 SFT-9B agents and judge completed predictions incrementally."""
import subprocess
import sys
import threading
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from incremental_judging import watch_summary

from judge_after_existing_run import DATASET, ROOT, RUN, RUNTIME, complete


SCRIPT = ROOT / 'services/run_gpu_embedding_eval_10_browsecomp.sh'
SUMMARY = ROOT / 'output/preds' / RUN / DATASET / 'summary.jsonl'


def judge_loop(stop, *, poll_seconds=5, retry_seconds=60):
    return watch_summary(stop, script=SCRIPT, summary=SUMMARY, poll_seconds=poll_seconds,
                         retry_seconds=retry_seconds, run=subprocess.run, clock=time.monotonic)


def main():
    stop = threading.Event()
    judge_thread = threading.Thread(target=judge_loop, args=(stop,), name='incremental-judge', daemon=True)
    judge_thread.start()
    print('starting SFT-9B evaluation: 32 agents, 8 per endpoint; KV preemption allowed', flush=True)
    try:
        result = subprocess.run([str(SCRIPT), 'full'], check=False)
    finally:
        stop.set()
        judge_thread.join()
    print(f'inference exited {result.returncode}', flush=True)
    done, total, terminal = complete(RUNTIME / f'{DATASET}.jsonl', SUMMARY)
    print(f'terminal questions: {terminal}/{total}', flush=True)
    if not done:
        print('evaluation incomplete; incremental scores remain available', flush=True)
        return result.returncode or 1
    print('final judge check; matching existing scores are reused', flush=True)
    judge = subprocess.run([str(SCRIPT), 'judge'], check=False)
    print(f'judge exited {judge.returncode}', flush=True)
    return judge.returncode


if __name__ == '__main__':
    sys.exit(main())
