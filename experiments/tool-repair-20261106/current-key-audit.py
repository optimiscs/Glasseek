import json,hashlib,time,concurrent.futures,http.client,collections
from pathlib import Path
D=Path(__file__).resolve().parent
keys=list(dict.fromkeys(json.loads(Path('/root/autodl-tmp/Quest/workspace/strict-reproduction/.secrets/api_key_pool.json').read_text())['serper']))
state=json.loads((D/'search-key-state.json').read_text());candidates=[(i+1,k) for i,k in enumerate(keys) if state.get(hashlib.sha256(k.encode()).hexdigest(),{}).get('until',0)<=time.time()]
def check(item):
 i,key=item;conn=http.client.HTTPSConnection('google.serper.dev',timeout=20);r={'index':i,'fingerprint':hashlib.sha256(key.encode()).hexdigest()[:12]}
 try:
  conn.request('POST','/search',body=json.dumps({'q':'OpenAI official website','num':10}),headers={'Content-Type':'application/json','X-API-KEY':key});reply=conn.getresponse();x=json.loads(reply.read());r.update(status=reply.status,credits=x.get('credits'),result_count=len(x.get('organic',[])))
 except Exception as e:r['error']=type(e).__name__
 finally:conn.close()
 return r
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(check,candidates))
result={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'total_keys':len(keys),'candidates':len(candidates),'statuses':dict(collections.Counter(r.get('status',r.get('error')) for r in rows)),'rows':rows}
(D/'current-key-audit.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='rows'})
