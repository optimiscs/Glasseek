import json,os,tempfile
from pathlib import Path
p=Path('/root/autodl-tmp/Quest/web/trace-viewer/datasets/index.json');x=json.loads(p.read_text());name='glasseek-full64k-step76-34'
if not any(y.get('id')==name for y in x['datasets']):
 x['datasets'].append({'id':name,'label':'Qwen3.5-9B 全参数 64K · step 76','description':'34 题重新推理；双 PRO 6000，异步判题','folder':'datasets/'+name,'data':'datasets/'+name+'/traces.json','benchmark':'BrowseComp 34','visible':True})
 fd,tmp=tempfile.mkstemp(dir=p.parent,prefix='.index-')
 with os.fdopen(fd,'w') as f:json.dump(x,f,ensure_ascii=False,separators=(',',':'))
 os.replace(tmp,p)
