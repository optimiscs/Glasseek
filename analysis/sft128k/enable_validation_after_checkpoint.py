#!/root/miniconda3/bin/python3
"""Switch the existing trainer to epoch validation after step 80 is durable."""
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/root/autodl-tmp/Glasseek')
CHECKPOINTS = ROOT / 'checkpoints/glasseek/sft_128k_length_curriculum_20261104'
WORKER = 80303
LAUNCHER = 74573
worker_stat = Path(f'/proc/{WORKER}/stat')
launcher_stat = Path(f'/proc/{LAUNCHER}/stat')
worker_start = worker_stat.read_text().split()[21]
launcher_start = launcher_stat.read_text().split()[21]


def alive(path, start):
    try:
        fields = path.read_text().split()
        return fields[21] == start and fields[2] != 'Z'
    except FileNotFoundError:
        return False


print('Waiting for completed checkpoint step 80; existing training continues', flush=True)
deadline = time.monotonic() + 8 * 3600
while time.monotonic() < deadline:
    if not alive(worker_stat, worker_start) or not alive(launcher_stat, launcher_start):
        raise SystemExit('Existing training or launcher exited; no automatic switch performed')
    try:
        step = int((CHECKPOINTS / 'latest_checkpointed_iteration.txt').read_text())
    except (OSError, ValueError):
        step = 0
    if step >= 80:
        folder = CHECKPOINTS / f'global_step_{step}'
        required = ('data_0.pt', 'model_world_size_1_rank_0.pt',
                    'optim_world_size_1_rank_0.pt', 'extra_state_world_size_1_rank_0.pt')
        if all((folder / f).is_file() and (folder / f).stat().st_size > 0 for f in required):
            break
    time.sleep(1)
else:
    raise SystemExit('Checkpoint not ready after 8 hours; existing training left running')

print(f'Checkpoint {step} complete; stopping launcher {LAUNCHER}', flush=True)
os.kill(LAUNCHER, signal.SIGTERM)
for _ in range(120):
    if not alive(launcher_stat, launcher_start) and not alive(worker_stat, worker_start):
        break
    time.sleep(1)
else:
    raise SystemExit('Existing processes did not exit; refusing to start another trainer')

os.chdir(ROOT)
env = os.environ.copy()
env['SFT_TRAIN_MODULE'] = 'glasseek_training.validation_history'
env['SFT_VALIDATION_STEPS'] = ','.join(str(i) for i in range(10, step + 1, 10))
env['SFT_VALIDATION_TENSORBOARD_DIR'] = str(ROOT / 'analysis/sft128k/tensorboard/length_curriculum_validation_history')
print(f'Backfilling validation for steps {env["SFT_VALIDATION_STEPS"]}', flush=True)
with open(ROOT / 'analysis/sft128k/validation-history.log', 'ab', buffering=0) as history_log:
    result = subprocess.run(
        ['bash', str(ROOT / 'analysis/sft128k/run_curriculum.sh'),
         f'trainer.resume_from_path={folder}', 'checkpoint.load_contents=[model]'],
        env=env, stdout=history_log, stderr=subprocess.STDOUT, check=False)
print(f'Historical validation exited with status {result.returncode}; resuming training', flush=True)
print(f'Starting from step {step} with validation after each epoch', flush=True)
log = open(ROOT / 'analysis/sft128k/curriculum-validation-resume.log', 'ab', buffering=0)
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)
os.execv('/bin/bash', ['bash', str(ROOT / 'analysis/sft128k/run_curriculum.sh'),
                       f'trainer.resume_from_path={folder}'])
