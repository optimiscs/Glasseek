import hashlib,json,os,time
from pathlib import Path
import httpx
from fastapi import FastAPI,Request
from fastapi.responses import StreamingResponse,Response
app=FastAPI();base='http://127.0.0.1:18100';root=Path('/root/autodl-tmp/Glasseek/checkpoint-reeval/analysis/prompt-audit');root.mkdir(parents=True,exist_ok=True)
expected=(root/'fixed_system_prompt.txt').read_text();reference=json.loads((root/'reference_tools.json').read_text())
def digest(o):return hashlib.sha256(json.dumps(o,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
@app.api_route('/{path:path}',methods=['GET','POST'])
async def forward(request:Request,path:str):
 raw=await request.body()
 if path=='v1/chat/completions':
  x=json.loads(raw);messages=x.get('messages',[]);system=[m.get('content') for m in messages if m.get('role')=='system'];tools=x.get('tools') or []
  rec={'time':time.time(),'model':x.get('model'),'system_count':len(system),'system_matches':system==[expected],'tool_names':[(t.get('function') or {}).get('name') for t in tools],'tools_match':tools==reference,'tools_hash':digest(tools),'reference_tools_hash':digest(reference),'message_count':len(messages),'stream':bool(x.get('stream'))}
  with (root/'requests.jsonl').open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
  if not (root/'first-request.json').exists():
   (root/'first-request.json').write_text(json.dumps({'system':system,'tools':tools},ensure_ascii=False,indent=2));os.chmod(root/'first-request.json',0o600)
 client=httpx.AsyncClient(timeout=None,trust_env=False);req=client.build_request(request.method,base+'/'+path,content=raw,headers={k:v for k,v in request.headers.items() if k.lower() not in ('host','content-length','connection')},params=request.query_params)
 try:up=await client.send(req,stream=True)
 except Exception:await client.aclose();raise
 async def stream():
  try:
   async for chunk in up.aiter_raw():yield chunk
  finally:await up.aclose();await client.aclose()
 return StreamingResponse(stream(),status_code=up.status_code,headers={k:v for k,v in up.headers.items() if k.lower() not in ('transfer-encoding','connection','content-encoding','content-length')})
