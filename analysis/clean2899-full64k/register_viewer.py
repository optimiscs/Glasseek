import json,os,tempfile
from pathlib import Path
p=Path('/root/autodl-tmp/Quest/web/trace-viewer/datasets/index.json');x=json.loads(p.read_text());name='glasseek-clean2899-full64k-step39-128'
if not any(y.get('id')==name for y in x['datasets']):
 x['datasets'].append({'id':name,'label':'Qwen3.5-9B 清洗去重全参数 · step 39','description':'128 题重新推理；双 PRO 6000，异步判题','folder':'datasets/'+name,'data':'datasets/'+name+'/traces.json','benchmark':'BrowseComp 128','visible':True})
 fd,tmp=tempfile.mkstemp(dir=p.parent,prefix='.index-')
 with os.fdopen(fd,'w') as f:json.dump(x,f,ensure_ascii=False,separators=(',',':'))
 os.replace(tmp,p)
