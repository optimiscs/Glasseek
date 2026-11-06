"""Shared paths and completion checks for evaluation supervisors."""
import os
import json
from pathlib import Path

ROOT = Path(os.environ.get('GLASSEEK_EVAL_ROOT', Path(__file__).resolve().parents[2]))
RUN = 'glasseek_sft9b_quest128_gpu_embedding_c10_browsecomp'
DATASET = 'quest_qwen27b_128'
RUNTIME = Path(os.environ.get('GLASSEEK_RUNTIME_DIR', ROOT / 'data/runtime'))


def complete(dataset, summary):
    expected = {json.loads(line)['id'] for line in dataset.read_text().splitlines() if line.strip()}
    if not summary.exists():
        return False, len(expected), 0
    latest = {}
    for line in summary.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record.get('question_id')] = record
    terminal = {key for key, record in latest.items() if record.get('status') in {'success', 'failed', 'policy_violation'} and record.get('attempt_consumed', True) is not False}
    return bool(expected) and expected <= terminal, len(expected), len(expected & terminal)
