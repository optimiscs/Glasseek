"""Isolated real-HTTP/Redis load harness. Default upstream is mock; never loads production keys."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
import resource
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid

import httpx
import aiohttp
import redis

ROOT=Path(__file__).resolve().parents[2]
SERVICES=ROOT/'Evaluation/services'
sys.path.insert(0,str(SERVICES))
from search_pool import load_config, client_env


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


def mock_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    app=FastAPI()
    rows=[]; active=Counter(); peak=Counter()
    @app.get('/health')
    async def health(): return {'ok':True}
    @app.get('/stats')
    async def stats(): return {'rows':rows,'peak':dict(peak),'active':dict(active)}
    @app.post('/search')
    async def search(request: Request):
        payload=await request.json(); key=request.headers.get('X-API-KEY','')
        return await respond(payload, key)
    @app.get('/fetch/{target:path}')
    async def fetch(target: str, request: Request):
        payload={'q':target, 'mock_delay':request.query_params.get('delay', '.05'),
                 'mock_status':request.query_params.get('status', '200'),
                 'mock_bytes':int(request.query_params.get('bytes', '0'))}
        key=request.headers.get('Authorization', '').removeprefix('Bearer ')
        return await respond(payload, key, reader=True)
    async def respond(payload, key, reader=False):
        active[key]+=1; active['total']+=1
        peak[key]=max(peak[key],active[key]); peak['total']=max(peak['total'],active['total'])
        status=402 if key.startswith('dead') else int(payload.get('mock_status',200))
        row={'q':payload['q'],'key':key,'at':time.time(),'status':status,'charged':status==200}
        rows.append(row)
        try:
            await asyncio.sleep(float(payload.get('mock_delay',.05)))
            data={'organic':[] if payload['q']=='empty' else [
                {'title':'evidence','link':'https://example.com/source','snippet':payload['q']}],'credits':1}
            if payload['q']=='invalid': status=400; row['status']=400;row['charged']=False
            if status>=400: data={'message':'Not enough credits' if status==402 else 'mock failure'}
            row['finished']=time.time()
            if reader and status == 200:
                from fastapi.responses import PlainTextResponse
                return PlainTextResponse('Reader content for '+payload['q']+'x'*min(1048576,payload.get('mock_bytes',0)))
            return JSONResponse(data,status_code=status,headers={'Retry-After':'60'} if status==429 else {})
        finally:
            active[key]-=1; active['total']-=1
    return app


class Sandbox:
    def __init__(self, directory, overrides=None, keys=None, replicas=2):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.logs=[];self.processes=[];self.replicas=[];self.urls=[]
        self.redis_port=free_port();self.upstream_port=free_port()
        self.redis_url=f'redis://127.0.0.1:{self.redis_port}/0'
        self.upstream_url=f'http://127.0.0.1:{self.upstream_port}'
        self.count=replicas
        keyfile=self.directory/'keys.json';keyfile.write_text(json.dumps({'serper':keys or ['fake-key']}));keyfile.chmod(0o600)
        self.env={k:v for k,v in os.environ.items() if not k.startswith(('SERPER_','SEARCH_','GLASSEEK_SEARCH_'))}
        self.env.update(SEARCH_POOL_REDIS_URL=self.redis_url,SEARCH_POOL_NAMESPACE='test-'+uuid.uuid4().hex,
            SERPER_KEY_POOL_FILE=str(keyfile),SERPER_KEY_STATE_FILE=str(self.directory/'state.json'),
            SERPER_UPSTREAM_URL=self.upstream_url+'/search',SEARCH_POOL_WORKERS='16',
            SERPER_PROXY_CONCURRENCY='16',SERPER_PROXY_QUEUE_LIMIT='64',SERPER_KEY_CONCURRENCY='2',
            SERPER_KEY_RPS='1000',SERPER_PROXY_QUEUE_TIMEOUT='10',SERPER_UPSTREAM_TIMEOUT='2',
            SERPER_UPSTREAM_BUDGET='10',SEARCH_POOL_LEASE_SECONDS='3',SEARCH_SERVER_TIMEOUT='20')
        self.env.update({k:str(v) for k,v in (overrides or {}).items()})
        config=load_config(self.env)
        config.pop('SEARCH_SERVER_ENDPOINT',None)
        config.pop('GLASSEEK_SEARCH_POOL_CONFIG',None)
        configfile=self.directory/'config.json'
        configfile.write_text(json.dumps(config))
        self.env['GLASSEEK_SEARCH_POOL_CONFIG']=str(configfile)
        self.prefix='{'+self.env['SEARCH_POOL_NAMESPACE']+'}:'
        self.redis=redis.Redis.from_url(self.redis_url,decode_responses=True)
        self.http=httpx.Client(trust_env=False,timeout=10)

    def process(self, command, env, name):
        log=(self.directory/(name+'.log')).open('a');self.logs.append(log)
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=log,start_new_session=True)
        self.processes.append(proc);return proc

    def start_redis(self):
        binary=shutil.which('redis-server') or str(ROOT/'.cache/search-pool/redis/usr/bin/redis-server')
        env=dict(os.environ)
        env['LD_LIBRARY_PATH']=str(ROOT/'.cache/search-pool/redis/usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
        env['LUA_CPATH']=str(ROOT/'.cache/search-pool/redis/usr/lib/x86_64-linux-gnu/lua/5.1/?.so')+';;'
        state=self.directory/'redis';state.mkdir(exist_ok=True)
        self.redis_process=self.process([binary,'--bind','127.0.0.1','--port',str(self.redis_port),
            '--dir',str(state),'--appendonly','yes','--appendfsync','always','--save','','--maxmemory-policy','noeviction'],env,'redis')
        for _ in range(100):
            try:
                self.redis.ping();return
            except redis.RedisError:time.sleep(.05)
        raise RuntimeError('Redis did not start')

    def ready(self, proc, url):
        for _ in range(200):
            if proc.poll() is not None: raise RuntimeError(f'Process failed; inspect {self.directory}')
            try:
                if self.http.get(url+'/health').status_code==200:return
            except httpx.HTTPError:pass
            time.sleep(.05)
        raise RuntimeError(f'Health timed out: {url}')

    def add_replica(self):
        port=free_port();env=dict(self.env,SEARCH_POOL_PORT=str(port))
        proc=self.process([sys.executable,str(SERVICES/'search_pool.py'),'serve'],env,'pool-'+str(port))
        url=f'http://127.0.0.1:{port}';self.ready(proc,url)
        self.replicas.append(proc);self.urls.append(url)
        return proc,url

    def __enter__(self):
        try:
            self.start_redis()
            proc=self.process([sys.executable,str(Path(__file__).resolve()),'--mock-port',str(self.upstream_port)],self.env,'upstream')
            self.ready(proc,self.upstream_url)
            for _ in range(self.count):self.add_replica()
            self.env['SEARCH_POOL_PORT']=self.urls[0].rsplit(':',1)[1]
            self.env.update(client_env(load_config(self.env)))
            return self
        except BaseException:
            self.__exit__(None,None,None);raise

    def __exit__(self,*args):
        for proc in reversed(self.processes):
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
        self.http.close();self.redis.close()
        for log in self.logs:log.close()

    def stats(self):return self.http.get(self.upstream_url+'/stats').json()
    def health(self):return self.http.get(self.urls[-1]+'/health').json()


def percentile(values,p):
    data=sorted(values)
    return data[min(len(data)-1,int((len(data)-1)*p))] if data else None


async def load(sandbox, concurrency, steps=3):
    waits=[];latencies=[];statuses=Counter();completed=0
    # Use an async connector suitable for sustained thousands of connections.
    async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=60),
            connector=aiohttp.TCPConnector(limit=concurrency)) as client:
        gate=asyncio.Event()
        async def trajectory(i):
            nonlocal completed
            await gate.wait()
            for step in range(steps):
                started=time.monotonic()
                async with client.post(sandbox.urls[i%len(sandbox.urls)]+'/search',
                        json={'q':f'c{concurrency}-t{i}-s{step}','mock_delay':.1},
                        headers={'Idempotency-Key':f'c{concurrency}-t{i}-s{step}'}) as response:
                    await response.read()
                    latencies.append((time.monotonic()-started)*1000)
                    statuses[response.status]+=1
                    if 'X-Search-Queue-Ms' in response.headers:waits.append(float(response.headers['X-Search-Queue-Ms']))
                    if response.status!=200:return
            completed+=1
        tasks=[asyncio.create_task(trajectory(i)) for i in range(concurrency)]
        started=time.monotonic();gate.set();await asyncio.gather(*tasks);elapsed=time.monotonic()-started
    rows=[r for r in sandbox.stats()['rows'] if r['q'].startswith(f'c{concurrency}-')]
    bills=Counter(r['q'] for r in rows if r['charged'])
    return dict(concurrency=concurrency,synthetic_trajectories=concurrency,steps_per_trajectory=steps,
        completed=completed,elapsed_seconds=elapsed,synthetic_trajectories_per_hour=completed/elapsed*3600,
        status_counts=dict(statuses),tool_success_rate=statuses[200]/sum(statuses.values()),
        queue_p95_ms=percentile(waits,.95),queue_p99_ms=percentile(waits,.99),
        latency_p95_ms=percentile(latencies,.95),latency_p99_ms=percentile(latencies,.99),
        mock_billed_calls=sum(bills.values()),duplicate_billed_calls=sum(max(0,n-1) for n in bills.values()),
        model_utilization=None,model_utilization_reason=f'No LLM used; synthetic {steps}-search trajectories only')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mock-port',type=int)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--levels',type=int,nargs='+',default=[64,128,256,512,1024])
    parser.add_argument('--steps',type=int,default=3)
    args=parser.parse_args()
    if args.mock_port:
        import uvicorn
        uvicorn.run(mock_app(),host='127.0.0.1',port=args.mock_port,log_level='warning');return
    if not args.output:parser.error('--output is required')
    if any(n<1 or n>1024 for n in args.levels) or args.steps<1:
        parser.error('levels must be 1..1024 and steps must be positive')
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(max(soft,16384),hard),hard))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='search-load-') as directory:
        with Sandbox(directory,keys=[f'fake-{i}' for i in range(128)],overrides={
            'SEARCH_POOL_WORKERS':512,'SERPER_PROXY_CONCURRENCY':1024,'SERPER_PROXY_QUEUE_LIMIT':2048,
            'SERPER_KEY_CONCURRENCY':8,'SERPER_KEY_RPS':1000,'SERPER_UPSTREAM_TIMEOUT':10,
            'SERPER_UPSTREAM_BUDGET':30,'SEARCH_POOL_LEASE_SECONDS':15,
            'SERPER_PROXY_QUEUE_TIMEOUT':20,'SEARCH_SERVER_TIMEOUT':55}) as sandbox:
            results=[]
            for level in args.levels:
                try:
                    result=asyncio.run(load(sandbox,level,args.steps))
                except Exception:
                    for log in Path(directory).glob('*.log'):
                        shutil.copyfile(log,args.output.parent/log.name)
                    raise
                results.append(result)
                print(json.dumps(result),flush=True)
                args.output.write_text(json.dumps({'results':results,'health':sandbox.health()},indent=2))
                if result['tool_success_rate']!=1 or result['duplicate_billed_calls']:
                    raise SystemExit('Load gate failed; stopping escalation')


if __name__=='__main__':main()
