"""Run 16 SFT-9B agents and judge completed predictions incrementally."""
import subprocess
import sys
import threading
import time

from judge_after_existing_run import DATASET, ROOT, RUN, RUNTIME, complete


SCRIPT = RUNTIME / 'services/run_gpu_embedding_eval_10_browsecomp.sh'
SUMMARY = ROOT / 'output/preds' / RUN / DATASET / 'summary.jsonl'


def judge_loop(stop, *, poll_seconds=5, retry_seconds=60):
    """One judge process at a time; the judge reuses unchanged results."""
    previous = None
    last_run = 0.0
    while not stop.is_set():
        try:
            stat = SUMMARY.stat()
            signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        except FileNotFoundError:
            signature = None
        now = time.monotonic()
        if signature and (signature != previous or now - last_run >= retry_seconds):
            print('incremental judge: checking completed predictions', flush=True)
            result = subprocess.run([str(SCRIPT), 'judge'], check=False)
            print(f'incremental judge exited {result.returncode}', flush=True)
            previous = signature
            last_run = time.monotonic()
        stop.wait(poll_seconds)


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
