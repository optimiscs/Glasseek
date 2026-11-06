"""Publish an evaluation workspace and judge completed answers while it runs."""
import json
import os
import subprocess
import threading
from pathlib import Path

from judge_after_existing_run import DATASET, ROOT, RUNTIME, complete
import run_sft9b_16_and_judge as incremental


def supervise(*, run, workspace_id, model, label, description, script, log_name):
    site = Path(os.environ.get('QUEST_VIEWER_SITE', ROOT.parent.parent / 'Quest/web/trace-viewer'))
    index = site / 'datasets/index.json'
    manifest = json.loads(index.read_text(encoding='utf8'))
    entry = next((item for item in manifest['datasets'] if item.get('id') == workspace_id), None)
    if entry is None:
        entry = {'id': workspace_id, 'folder': f'datasets/{workspace_id}',
                 'data': f'datasets/{workspace_id}/traces.json', 'benchmark': 'BrowseComp', 'visible': True}
        manifest['datasets'].append(entry)
    entry.update(label=label, description=description)
    tmp = index.with_name(f'.index-{workspace_id}-{os.getpid()}.tmp')
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    os.replace(tmp, index)
    env = os.environ.copy()
    env.update(GLASSEEK_VIEWER_RUN=run, GLASSEEK_VIEWER_MODEL=model, GLASSEEK_VIEWER_LABEL=label)
    with (RUNTIME / log_name).open('ab', buffering=0) as log:
        viewer = subprocess.Popen([str(RUNTIME / 'venv/bin/python'), '-u', str(ROOT / 'quest_viewer_adapter.py'),
                                   '--output', str(site / 'datasets' / workspace_id / 'traces.json'), '--interval', '10'],
                                  env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                  start_new_session=True)
    print('viewer publisher', viewer.pid, flush=True)
    summary = ROOT / 'output/preds' / run / DATASET / 'summary.jsonl'
    incremental.SCRIPT, incremental.SUMMARY = script, summary
    stop = threading.Event()
    thread = threading.Thread(target=incremental.judge_loop, args=(stop,), daemon=True)
    thread.start()
    try:
        result = subprocess.run([str(script), 'full'], check=False)
    finally:
        stop.set()
        thread.join()
    print('inference exited', result.returncode, flush=True)
    done, total, terminal = complete(RUNTIME / f'{DATASET}.jsonl', summary)
    print(f'terminal questions {terminal}/{total}', flush=True)
    if not done:
        return result.returncode or 1
    judged = subprocess.run([str(script), 'judge'], check=False)
    print('final judge exited', judged.returncode, flush=True)
    return judged.returncode
