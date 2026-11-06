import json,hashlib,subprocess,urllib.request
from pathlib import Path
D=Path(__file__).resolve().parent
assert hashlib.sha256((D/'browsecomp128.jsonl').read_bytes()).hexdigest()=='9672116e757038283aa5569e08959e5e3afa0f127eeec9f76c0dd6e58bc68ad6'
assert len((D/'browsecomp128.jsonl').read_text().splitlines())==128
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def req(port,path,body=None):
 r=opener.open(urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'}),timeout=100)
 return json.load(r)
x=req(18083,'/search',{'q':'OpenAI official website','num':1});assert x.get('organic'),str(x)[:100]
x=req(18081,'/fetch',{'url':'https://example.com'});assert x.get('success') and x.get('data'),str(x)[:100]
x=req(18213,'/health');assert x['model']=='QWEN3-EMBEDDING-8B'
x=req(18217,'/select',{'goal':'What does an example domain demonstrate?','text':('Example Domain. This domain is for use in illustrative examples in documents.\n\n' * 1200)});assert x.get('content') and x.get('audit')
subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','Lab3090','python3 -c "print(2+2)"'],check=True,timeout=20)
print('PASS: fixed 128 questions, search, fetch, embedding retrieval, Lab3090 execution',flush=True)
