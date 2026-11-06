import json, os, sys, tempfile
from pathlib import Path
site = Path(sys.argv[1]); step = int(sys.argv[2])
if step not in range(10,81,10): raise SystemExit('invalid checkpoint')
name = f'glasseek-step{step}-rl9b-34'
index = site/'datasets/index.json'
data = json.loads(index.read_text())
entry = {'id':name, 'label':f'Qwen3.5-9B · step {step} · 34 题', 'description':f'LoRA step {step}，独立实时轨迹及 DeepSeek Flash 评分', 'folder':f'datasets/{name}', 'data':f'datasets/{name}/traces.json', 'benchmark':'BrowseComp 34', 'visible':True}
items = data['datasets']
if not any(x.get('id') == name for x in items):
    items.append(entry)
    fd,tmp=tempfile.mkstemp(dir=index.parent,prefix='.index-')
    with os.fdopen(fd,'w') as f: json.dump(data,f,ensure_ascii=False,indent=2)
    os.replace(tmp,index)
