"""Measure four individual real keys with the canonical service; bounded paid traffic."""
import argparse
import asyncio
from collections import Counter
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import uuid

import aiohttp
import httpx
import redis

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Evaluation/services'))
from search_pool import load_config

# Reserve ONLY the measured keys. Production keeps the rest; no key is used by
# two independent limiters at once. TTL bounds disruption if the driver dies.
RESERVE='''
local p=KEYS[1];local a=cjson.decode(ARGV[1]);local t=redis.call('TIME');local now=t[1]+t[2]/1000000
if redis.call('EXISTS',p..'key_probe_lock')==1 then return redis.error_reply('Another probe is active') end
for _,key in ipairs(a.keys) do
 local raw=redis.call('HGET',p..'state',key);local s=raw and cjson.decode(raw) or {}
 if s.reason=='quota_exhausted' or s.reason=='invalid_key' or s.reason=='retired' or (s['until'] or 0)>now then return redis.error_reply('Selected key unavailable') end
 if redis.call('ZCOUNT',p..'inflight:'..key,now,'+inf')>0 then return redis.error_reply('Selected key is busy') end
end
redis.call('SET',p..'key_probe_lock',a.run,'EX',300)
for _,key in ipairs(a.keys) do
 local raw=redis.call('HGET',p..'state',key)
 redis.call('HSET',p..'probe_backup:'..a.run,key,raw or '')
 redis.call('HSET',p..'state',key,cjson.encode({reason='benchmark_reserved',['until']=now+300,run=a.run}))
end
return 1
'''
RESTORE='''
local p=KEYS[1];local a=cjson.decode(ARGV[1]);local t=redis.call('TIME');local now=t[1]+t[2]/1000000
if redis.call('EXISTS',p..'probe_merged:'..a.run)==1 then return 0 end
for key,item in pairs(a.keys) do
 local raw=redis.call('HGET',p..'state',key);local current=raw and cjson.decode(raw) or {}
 if current.run==a.run then
  local backup=redis.call('HGET',p..'probe_backup:'..a.run,key) or ''
  if item.state~='' then redis.call('HSET',p..'state',key,item.state)
  elseif item.hold_until>now then redis.call('HSET',p..'state',key,cjson.encode({reason='benchmark_inflight',['until']=item.hold_until}))
  elseif backup~='' then redis.call('HSET',p..'state',key,backup)
  else redis.call('HDEL',p..'state',key) end
 end
 for name,value in pairs(item.usage) do redis.call('HINCRBYFLOAT',p..'usage:'..key,name,value) end
end
redis.call('SET',p..'probe_merged:'..a.run,1)
if redis.call('GET',p..'key_probe_lock')==a.run then redis.call('DEL',p..'key_probe_lock') end
redis.call('XADD',p..'events','*','event','key_rps_probe_merged','run',a.run,'at',now)
return 1
'''


def percentile(xs,p):
    xs=sorted(xs)
    return xs[min(len(xs)-1,math.ceil(len(xs)*p)-1)] if xs else None


def free_port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]


async def measure(urls,rps,seconds):
    start=time.monotonic()+.2;epoch=time.time()+.2
    results=[[] for _ in urls]
    async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=65),
                                    connector=aiohttp.TCPConnector(limit=256)) as client:
        async def one(k,i):
            await asyncio.sleep(max(0,start+i/rps-time.monotonic()))
            at=time.monotonic()
            # Distinct, ordinary queries across the run; no client response cache.
            technologies=['Python asyncio','Redis','FastAPI','aiohttp','PostgreSQL','HTTPX','Ray Serve','vLLM','Temporal','Docker','Kubernetes','Prometheus','SQLite','Nginx','OpenTelemetry','Linux TCP']
            subjects=['official documentation','configuration reference','timeout handling','performance tuning','connection pooling','monitoring metrics','deployment guide','error handling','retry policy','API reference']
            query=f'{technologies[i%len(technologies)]} {subjects[(i//len(technologies)+k)%len(subjects)]}'
            row={'index':i,'sent_s':at-start}
            try:
                async with client.post(urls[k]+'/search',json={'q':query,'num':10},headers={'Idempotency-Key':f'probe-{k}-{i}'}) as response:
                    data=await response.json()
                    row.update(status=response.status,error_code=data.get('error_code'),
                        queue_ms=float(response.headers.get('X-Search-Queue-Ms',0)),
                        request_id=response.headers.get('X-Search-Request-ID'))
            except (aiohttp.ClientError,TimeoutError,ValueError) as exc:row['error']=type(exc).__name__
            row.update(completed_s=time.monotonic()-start,latency_ms=(time.monotonic()-at)*1000)
            results[k].append(row)
        await asyncio.gather(*(one(k,i) for k in range(len(urls)) for i in range(int(rps*seconds))))
    return epoch,results


def run(output,rps=8,seconds=20,fingerprints=None,key_concurrency=32):
    if output.exists():raise ValueError('Output exists; refusing to accidentally repeat paid traffic')
    output.parent.mkdir(parents=True,exist_ok=True)
    # Reserve the output name before any upstream requests.
    output.write_text('{"status":"preparing"}\n')
    config=load_config();p='{'+config['SEARCH_POOL_NAMESPACE']+'}:'
    state=redis.Redis.from_url(config['SEARCH_POOL_REDIS_URL'],decode_responses=True)
    now=sum(x/y for x,y in zip(state.time(),(1,1e6)))
    keys=state.hkeys(p+'keys');eligible=[]
    for key in keys:
        if fingerprints and key[:12] not in fingerprints:continue
        s=json.loads(state.hget(p+'state',key) or '{}')
        if s.get('reason') in ('quota_exhausted','invalid_key','retired') or s.get('until',0)>now:continue
        if state.zcount(p+'inflight:'+key,now,'+inf'):continue
        successes=float(state.hget(p+'usage:'+key,'http_200') or 0)
        eligible.append((successes,key))
    chosen=[k for _,k in sorted(eligible,reverse=True)[:4]]
    if len(chosen)!=4:raise ValueError('Four idle eligible credentials required')
    run_id=uuid.uuid4().hex
    state.eval(RESERVE,1,p,json.dumps({'run':run_id,'keys':chosen}))
    processes=[];logs=[];prefixes=[];urls=[]
    report={'run':run_id,'target_rps_per_key':rps,'offered_seconds':seconds,
            'request_budget_per_key':int(rps*seconds),'upstream_attempt_ceiling':4*int(rps*seconds),
            'test_concurrency_per_key':key_concurrency,'production_config_before':config,'keys':[]}
    # Never include Redis passwords if configured on another host.
    report['production_config_before']=dict(config,SEARCH_POOL_REDIS_URL='[private endpoint omitted]')
    try:
        with tempfile.TemporaryDirectory(prefix='glasseek-key-rps-') as tmp:
            os.chmod(tmp,0o700)
            with httpx.Client(trust_env=False,timeout=3) as client:
                for n,key in enumerate(chosen):
                    d=Path(tmp)/str(n);d.mkdir(mode=0o700)
                    keyfile=d/'keys.json';keyfile.write_text(json.dumps({'serper':[state.hget(p+'keys',key)]}));keyfile.chmod(0o600)
                    port=free_port();namespace=f'key-rps-{run_id}-{n}'
                    cfg=dict(config,SEARCH_POOL_PORT=str(port),SEARCH_POOL_NAMESPACE=namespace,
                        SERPER_KEY_POOL_FILE=str(keyfile),SERPER_KEY_STATE_FILE=str(d/'state.json'),
                        SEARCH_POOL_WORKERS='32',SERPER_PROXY_CONCURRENCY='32',SERPER_PROXY_QUEUE_LIMIT='256',
                        SERPER_KEY_CONCURRENCY=str(key_concurrency),SERPER_KEY_RPS=str(rps),SERPER_PROXY_RPS=str(rps),
                        SERPER_PROXY_QUEUE_TIMEOUT='45',SERPER_UPSTREAM_BUDGET='45',SERPER_UPSTREAM_TIMEOUT='20',
                        SEARCH_POOL_LEASE_SECONDS='60',SEARCH_SERVER_TIMEOUT='60',
                        SEARCH_POOL_MAX_UPSTREAM_ATTEMPTS=str(int(rps*seconds)))
                    cfg.pop('SEARCH_SERVER_ENDPOINT',None);cfg.pop('GLASSEEK_SEARCH_POOL_CONFIG',None)
                    file=d/'config.json';file.write_text(json.dumps(cfg));file.chmod(0o600)
                    env={k:v for k,v in os.environ.items() if not k.startswith(('SEARCH_','SERPER_','GLASSEEK_SEARCH_'))}
                    env['GLASSEEK_SEARCH_POOL_CONFIG']=str(file)
                    log=(d/'service.log').open('w');logs.append(log)
                    proc=subprocess.Popen([sys.executable,str(ROOT/'Evaluation/services/search_pool.py'),'serve'],
                        env=env,stdout=log,stderr=log,start_new_session=True)
                    processes.append(proc);prefixes.append('{'+namespace+'}:');url=f'http://127.0.0.1:{port}';urls.append(url)
                    for _ in range(100):
                        if proc.poll() is not None:raise RuntimeError('Test service startup failed')
                        try:
                            if client.get(url+'/health').status_code==200:break
                        except httpx.HTTPError:pass
                        time.sleep(.05)
                    else:raise RuntimeError('Test service health timeout')
            epoch,results=asyncio.run(measure(urls,rps,seconds))
            report['observed_at_epoch']=epoch
            for n,(key,rows,prefix) in enumerate(zip(chosen,results,prefixes)):
                events=[v for _,v in state.xrange(prefix+'events')]
                attempts=[v for v in events if v['event']=='attempt']
                responses=[json.loads(v['details']) for v in events if v['event']=='response']
                ok=[r for r in rows if r.get('status')==200]
                elapsed=max(r['completed_s'] for r in rows)-min(r['sent_s'] for r in rows)
                windows=[{'start_s':w,'completed_rps':sum(w<=r['completed_s']<w+5 for r in ok)/5,
                          'dispatch_rps':sum(epoch+w<=float(a['at'])<epoch+w+5 for a in attempts)/5}
                         for w in range(5,int(seconds)-4,5)]
                ids=Counter(a['request_id'] for a in attempts)
                report['keys'].append(dict(label=f'K{n+1}',fingerprint=key[:12],requests=len(rows),
                    success=len(ok),success_rate=len(ok)/len(rows),whole_run_success_rps=len(ok)/elapsed,
                    elapsed_with_drain_s=elapsed,windows=windows,
                    minimum_5s_completion_rps=min((w['completed_rps'] for w in windows),default=None),
                    latency_p50_ms=percentile([r['latency_ms'] for r in rows],.5),
                    latency_p95_ms=percentile([r['latency_ms'] for r in rows],.95),
                    latency_p99_ms=percentile([r['latency_ms'] for r in rows],.99),
                    queue_p95_ms=percentile([r.get('queue_ms',0) for r in rows],.95),
                    queue_p99_ms=percentile([r.get('queue_ms',0) for r in rows],.99),
                    upstream_attempts=len(attempts),upstream_statuses=dict(Counter(str(r['status']) for r in responses)),
                    confirmed_credits=sum(r.get('credits') or 0 for r in responses),
                    duplicate_attempts=sum(max(0,n-1) for n in ids.values()),
                    errors=dict(Counter(r.get('error_code') or r.get('error') for r in rows if r.get('status')!=200)),
                    calls=rows))
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
        for log in logs:log.close()
        restore={}
        for n,key in enumerate(chosen):
            prefix=prefixes[n] if n<len(prefixes) else None
            usage=state.hgetall(prefix+'usage:'+key) if prefix else {}
            key_state=state.hget(prefix+'state',key) if prefix else ''
            leases=state.zrange(prefix+'inflight:'+key,0,-1,withscores=True) if prefix else []
            restore[key]={'usage':{k:float(v) for k,v in usage.items()},'state':key_state or '',
                          'hold_until':max((score for _,score in leases),default=0)}
        state.eval(RESTORE,1,p,json.dumps({'run':run_id,'keys':restore}))
        for prefix in prefixes:state.delete(prefix+'keys')
        report['production_keys_restored']=True
        report['production_scheduler_after']=state.hgetall(p+'config')
        report['note']='Individual-key short test, not a long-term guaranteed lower bound. Four keys temporarily excluded from production, then restored; observed quarantine and usage retained.'
        output.write_text(json.dumps(report,indent=2)+'\n');state.close()
    print(json.dumps({**{k:v for k,v in report.items() if k not in ('keys','production_config_before')},
                      'keys':[{k:v for k,v in row.items() if k!='calls'} for row in report['keys']]},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rps',type=float,default=8)
    parser.add_argument('--seconds',type=int,default=20)
    parser.add_argument('--key-fingerprints',nargs=4,help='Optional four 12-character SHA256 prefixes from a prior report')
    parser.add_argument('--key-concurrency',type=int,default=32)
    args=parser.parse_args()
    if not 0<args.rps<=8 or not 1<=args.seconds<=20:parser.error('Bounded to <=8 RPS/key and <=20 seconds')
    if not 1<=args.key_concurrency<=32:parser.error('Key concurrency must be 1..32')
    run(args.output,args.rps,args.seconds,args.key_fingerprints,args.key_concurrency)
