import concurrent.futures,collections,json,math,os,signal,subprocess,time,urllib.request,urllib.error
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parents[1]
OP=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def request(path,body=None):
 req=urllib.request.Request('http://127.0.0.1:18083'+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
 try:r=OP.open(req,timeout=60)
 except urllib.error.HTTPError as e:r=e
 return r.status,json.loads(r.read())
def stop():
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:a=(p/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  if b'uvicorn' in a and b'serper_proxy:app' in a:os.kill(int(p.name),signal.SIGTERM)
 deadline=time.monotonic()+60
 while time.monotonic()<deadline:
  try:request('/health');time.sleep(.2)
  except Exception:return
 raise RuntimeError('Old proxy did not exit')
def start(conf):
 stop();time.sleep(3)
 env=os.environ.copy();env.update(conf)
 env['SERPER_KEY_POOL_FILE']='/root/autodl-tmp/Quest/workspace/strict-reproduction/.secrets/api_key_pool.json'
 env['SERPER_KEY_STATE_FILE']=str(D/'search-key-state.json')
 log=(D/'pool-benchmark-server.log').open('a')
 child=subprocess.Popen(['/root/autodl-tmp/Quest/workspace/env/bin/python','-m','uvicorn','serper_proxy:app','--host','127.0.0.1','--port','18083'],cwd=P/'Evaluation/services',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
 for _ in range(60):
  try:return request('/health')[1]
  except Exception:
   if child.poll() is not None:raise RuntimeError('Proxy startup failed')
   time.sleep(.2)
 raise RuntimeError('Proxy readiness timed out')
queries=['Jamie Uys The Gods Must Be Crazy','Trefor Jenkins award','Singh Is Bliing Lara Dutta review','Kid Trunks real name','New Castle Court House Museum history','Art Gallery New South Wales bequest 1951','Domenico Modugno biography','NASA Mars mission','University Oxford history','"Da Silva" Belvedere academy president federation re-elected researcher education']
stages=[(16,1,.25,124),(32,1,.5,186),(64,1,1.,248),(64,2,1.,248),(128,2,2.,372)]
report={'started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'stages':[]}
best=None
try:
 for workers,key_con,key_rps,n in stages:
  conf={'SERPER_PROXY_CONCURRENCY':str(workers),'SERPER_PROXY_QUEUE_LIMIT':'128','SERPER_PROXY_RPS':'0','SERPER_KEY_CONCURRENCY':str(key_con),'SERPER_KEY_RPS':str(key_rps),'SERPER_PROXY_QUEUE_TIMEOUT':'5','SERPER_UPSTREAM_BUDGET':'45','SERPER_UPSTREAM_TIMEOUT':'20'}
  health=start(conf);print('START',workers,key_con,key_rps,'eligible',health['eligible_keys'],flush=True)
  def work(i):
   t=time.monotonic()
   try:
    code,data=request('/search',{'q':queries[i%len(queries)],'num':10});ok=code==200 and data.get('success') is not False and isinstance(data.get('organic'),list)
    return {'status':code,'ok':ok,'error_code':data.get('error_code'),'latency':time.monotonic()-t,'results':len(data.get('organic',[]))}
   except Exception as e:return {'status':0,'ok':False,'error_code':type(e).__name__,'latency':time.monotonic()-t}
  begin=time.monotonic()
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(work,range(n)))
  elapsed=time.monotonic()-begin;lat=sorted(x['latency'] for x in rows);health=request('/health')[1]
  stage={'pool_concurrency':workers,'per_key_concurrency':key_con,'per_key_rps':key_rps,'requests':n,'success':sum(x['ok'] for x in rows),'elapsed_seconds':elapsed,'successful_qps':sum(x['ok'] for x in rows)/elapsed,'mean_latency':sum(lat)/n,'p95_latency':lat[math.ceil(n*.95)-1],'errors':dict(collections.Counter(x['error_code'] for x in rows if not x['ok'])),'health':health,'rows':rows,'config':conf}
  report['stages'].append(stage);(D/'pool-benchmark.json').write_text(json.dumps(report,indent=2));print('RESULT',json.dumps({k:v for k,v in stage.items() if k not in ('health','rows','config')}),flush=True)
  if stage['success']/n>=.99 and not health['metrics'].get('upstream_http_429',0):
   if best is None or stage['successful_qps']>best['successful_qps']:best=stage
  if stage['success']/n<.98 or health['metrics'].get('upstream_http_429',0):
   print('Stopping ramp after throttling or error rate above 2%; retaining best stage',flush=True);break
 if best is None:raise RuntimeError('No clean stage passed')
 # A larger repeat checks that the selected settings were not a short burst.
 conf=best['config'];health=start(conf);workers=best['pool_concurrency'];n=496
 begin=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(work,range(n)))
 elapsed=time.monotonic()-begin;health=request('/health')[1];lat=sorted(x['latency'] for x in rows)
 repeat={'requests':n,'success':sum(x['ok'] for x in rows),'elapsed_seconds':elapsed,'successful_qps':sum(x['ok'] for x in rows)/elapsed,'p95_latency':lat[math.ceil(n*.95)-1],'errors':dict(collections.Counter(x['error_code'] for x in rows if not x['ok'])),'health':health}
 report['validation']=repeat;report['selected_config']=conf
 if repeat['success']/n<.99 or health['metrics'].get('upstream_http_429',0):raise RuntimeError('Selected configuration failed validation')
 (D/'search-pool-config.json').write_text(json.dumps(conf,indent=2));report['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
 print('VALIDATED',json.dumps({k:v for k,v in repeat.items() if k!='health'}),flush=True)
finally:
 (D/'pool-benchmark.json').write_text(json.dumps(report,indent=2))
