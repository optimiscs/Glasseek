import collections,hashlib,http.client,json,os,time
from pathlib import Path
D=Path(__file__).resolve().parent
keys=list(dict.fromkeys(json.loads(Path('/root/autodl-tmp/Quest/workspace/strict-reproduction/.secrets/api_key_pool.json').read_text())['serper']))
state={};rows=[]
for i,key in enumerate(keys):
 started=time.monotonic();row={'index':i+1}
 conn=http.client.HTTPSConnection('google.serper.dev',timeout=8)
 try:
  conn.request('POST','/search',body=json.dumps({'q':'OpenAI official website','num':1}),headers={'Content-Type':'application/json','X-API-KEY':key})
  response=conn.getresponse();data=response.read();row['status']=response.status
  if response.status==400 and b'not enough credits' in data.lower():reason='quota_exhausted'
  elif response.status in (401,402):reason='invalid_key' if response.status==401 else 'quota_exhausted'
  elif response.status==429:reason='rate_limited'
  elif response.status==200:reason='healthy';row['result_count']=len(json.loads(data).get('organic',[]))
  else:reason='other_http_error'
  row['classification']=reason
  if reason in ('quota_exhausted','invalid_key','rate_limited'):state[hashlib.sha256(key.encode()).hexdigest()]={'reason':reason,'until':time.time()+(86400 if reason!='rate_limited' else 120)}
 except Exception as e:row.update(classification='transport_error',error=type(e).__name__)
 finally:conn.close()
 row['seconds']=round(time.monotonic()-started,3);rows.append(row)
 if (i+1)%10==0:print('checked',i+1,dict(collections.Counter(x['classification'] for x in rows)),flush=True)
 time.sleep(max(0,1-(time.monotonic()-started)))
path=D/'search-key-state.json';fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
with os.fdopen(fd,'w') as f:json.dump(state,f)
report={'checked':len(rows),'counts':dict(collections.Counter(x['classification'] for x in rows)),'rows':rows}
(D/'key-audit.json').write_text(json.dumps(report,indent=2));print(report['counts'],flush=True)
