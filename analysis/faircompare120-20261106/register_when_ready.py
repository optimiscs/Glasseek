"""Register viewer datasets only when they contain real trajectories."""
import json
import os
from pathlib import Path
import subprocess
import time

D=Path(__file__).resolve().parent
SITE=Path('/root/autodl-tmp/Quest/web/trace-viewer')
remaining={'step39','step76','sft9b'}
while remaining:
    for label in list(remaining):
        name=f'fair120-20261106-{label}'
        path=SITE/'datasets'/name/'traces.json'
        try: data=json.loads(path.read_text())
        except (OSError,ValueError): continue
        if not data.get('cases'):continue
        env=os.environ.copy()
        env.update(GLASSEEK_VIEWER_WORKSPACE=name,
                   GLASSEEK_VIEWER_LABEL=f'{label} · 公平对比128题 · 工具120次 · 64并发/卡')
        subprocess.run(['python',str(D.parent/'clean2899-full64k/concurrency16/register_repeat.py')],env=env,check=True)
        subprocess.run(['/root/autodl-tmp/Quest/workspace/viewer/venv/bin/quest-viewer','import','--site','trace-viewer'],cwd=SITE.parent,check=True)
        remaining.remove(label)
        print(f'Registered {name}',flush=True)
    time.sleep(20)
