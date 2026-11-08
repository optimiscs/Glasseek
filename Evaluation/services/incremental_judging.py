"""Reusable summary watcher; no global run/script state shared by supervisors."""
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from browse_comp_eval.judge_policy import SummaryJudgePolicy


def watch_summary(stop, *, script, summary, poll_seconds=5, retry_seconds=60,
                  run=None, clock=None):
    run = run or subprocess.run
    clock = clock or time.monotonic
    policy = SummaryJudgePolicy(retry_seconds)
    previous, last_run = None, 0.0
    while not stop.is_set():
        try:
            stat = Path(summary).stat()
            signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        except FileNotFoundError:
            signature = None
        now = clock()
        if policy.due(signature, previous, now, last_run):
            result = run([str(script), 'judge'], check=False)
            print(f'incremental judge exited {result.returncode}', flush=True)
            previous, last_run = signature, clock()
        stop.wait(poll_seconds)
