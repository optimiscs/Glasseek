"""Publish and run the independent 44 RL-4B 128-question evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from judge_after_existing_run import DATASET, ROOT, RUNTIME, complete
import run_sft9b_16_and_judge as incremental

RUN = 'glasseek_rl9b_quest128_32'
ID = 'glasseek-rl9b-browsecomp-128-32'
SITE = Path('/root/autodl-tmp/Quest/web/trace-viewer')
SCRIPT = RUNTIME / 'services/run_rl9b_eval.sh'
SUMMARY = ROOT / 'output/preds' / RUN / DATASET / 'summary.jsonl'


def main():
    index = SITE / 'datasets/index.json'
    manifest = json.loads(index.read_text(encoding='utf8'))
    entry = next((item for item in manifest['datasets'] if item.get('id') == ID), None)
    if entry is None:
        entry = {'id': ID, 'folder': f'datasets/{ID}', 'data': f'datasets/{ID}/traces.json',
                 'benchmark': 'BrowseComp', 'visible': True}
        manifest['datasets'].append(entry)
    entry.update(label='Glasseek RL-9B · BrowseComp · Quest 128 · 32 并发',
                 description='本机四卡 · RL-9B · TP1 · FP8 KV · DeepSeek Flash；单卡 8 Agent')
    tmp = index.with_name('.index-glasseek-rl9b-32.tmp')
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    os.replace(tmp, index)
    env = os.environ.copy()
    env.update(GLASSEEK_VIEWER_RUN=RUN, GLASSEEK_VIEWER_MODEL='Glasseek RL-9B',
               GLASSEEK_VIEWER_LABEL='Glasseek RL-9B · BrowseComp · Quest 128 · 32 并发')
    out = SITE / 'datasets' / ID / 'traces.json'
    log = (RUNTIME / 'viewer-publisher-rl9b-32.log').open('ab', buffering=0)
    viewer = subprocess.Popen([str(RUNTIME / 'venv/bin/python'), '-u', str(ROOT / 'quest_viewer_adapter.py'),
                               '--output', str(out), '--interval', '10'], env=env, stdout=log,
                              stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print('viewer publisher', viewer.pid, flush=True)
    incremental.SCRIPT, incremental.SUMMARY = SCRIPT, SUMMARY
    stop = threading.Event()
    thread = threading.Thread(target=incremental.judge_loop, args=(stop,), daemon=True)
    thread.start()
    try:
        result = subprocess.run([str(SCRIPT), 'full'], check=False)
    finally:
        stop.set(); thread.join()
    print('inference exited', result.returncode, flush=True)
    done, total, terminal = complete(RUNTIME / f'{DATASET}.jsonl', SUMMARY)
    print(f'terminal questions {terminal}/{total}', flush=True)
    if not done:
        return result.returncode or 1
    judged = subprocess.run([str(SCRIPT), 'judge'], check=False)
    print('final judge exited', judged.returncode, flush=True)
    return judged.returncode


if __name__ == '__main__':
    sys.exit(main())
