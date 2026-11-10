#!/usr/bin/env python3
"""Replace the orphaned trainer after a complete checkpoint, without replaying steps."""
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path('/root/autodl-tmp/Glasseek')
checkpoint = root / 'checkpoints/glasseek/sft_128k_length_curriculum_20261104'
pid = 29413
proc = Path(f'/proc/{pid}')
start = (proc / 'stat').read_text().split()[21]
deadline = time.monotonic() + 3 * 3600
print('Waiting for completed checkpoint >= 50', flush=True)
while time.monotonic() < deadline:
    if not proc.exists() or (proc / 'stat').read_text().split()[21] != start:
        raise SystemExit('Original training process exited; no automatic restart performed')
    try:
        step = int((checkpoint / 'latest_checkpointed_iteration.txt').read_text())
    except (OSError, ValueError):
        step = 0
    if step >= 50:
        folder = checkpoint / f'global_step_{step}'
        files = ['data_0.pt', 'model_world_size_1_rank_0.pt',
                 'optim_world_size_1_rank_0.pt', 'extra_state_world_size_1_rank_0.pt']
        if all((folder / f).is_file() and (folder / f).stat().st_size > 0 for f in files):
            break
    time.sleep(5)
else:
    raise SystemExit('Checkpoint did not complete within 3 hours; training left running')

print(f'Checkpoint {step} complete; stopping orphan trainer {pid}', flush=True)
os.kill(pid, signal.SIGTERM)
for _ in range(60):
    if not proc.exists() or (proc / 'stat').read_text().split()[2] == 'Z':
        break
    time.sleep(1)
else:
    raise SystemExit('Trainer did not exit; refusing to start a second trainer')
print('Starting trainer with automatic checkpoint resume', flush=True)
os.chdir(root)
log = open(root / 'reports/experiments/sft128k/curriculum-resume.log', 'ab', buffering=0)
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)
os.execv('/bin/bash', ['bash', str(root / 'experiments/sft128k/run_curriculum.sh'),
                       f'trainer.resume_from_path={folder}'])
