import json,time,urllib.request
from pathlib import Path
D=Path(__file__).resolve().parent
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for port in (19139,19239):
 deadline=time.monotonic()+1200
 while True:
  try:
   x=json.load(opener.open(f'http://127.0.0.1:{port}/v1/models',timeout=10));assert [m['id'] for m in x['data']]==['clean2899-step39'];break
  except Exception:
   if time.monotonic()>deadline:raise
   time.sleep(5)
 body={'model':'clean2899-step39','messages':[{'role':'user','content':'What is 2 + 2?'}],'max_tokens':128,'temperature':0,'chat_template_kwargs':{'enable_thinking':True}}
 r=json.load(opener.open(urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=120));assert r['model']=='clean2899-step39' and r['choices']
 (D/f'smoke-{port}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
print('Both replicas passed exact model routing and generation smoke checks',flush=True)
