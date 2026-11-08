"""Validate unverified credentials once using the canonical pool; merge durable state."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Evaluation/services'))
from provider_services import load_service, registry
from provider_pool import ProviderPool
from search_pool_state import PoolState
from search_pool_key_rps import RESERVE, RESTORE


async def validate(service, expected, output):
    config=load_service(service);state=PoolState(config)
    run=uuid.uuid4().hex;probes=[];reserved=False
    if output.exists():raise ValueError('Output already exists; refusing to repeat validation traffic')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('{"status":"preparing"}\n')
    try:
        await state.initialize()
        health=await state.health()
        keys=await state.redis.hgetall(state.prefix+'keys')
        statuses=await state.redis.hgetall(state.prefix+'state')
        now=time.time();selected=[]
        for key,secret in keys.items():
            status=json.loads(statuses.get(key,'{}'))
            if status.get('reason') in ('quota_exhausted','invalid_key','retired') or status.get('until',0)>now:continue
            if float(await state.redis.hget(state.prefix+'usage:'+key,'http_200') or 0)>0:continue
            selected.append((key,secret))
        if len(selected)!=expected:raise ValueError(f'Expected {expected} unverified keys; found {len(selected)}')
        await state.redis.eval(RESERVE,1,state.prefix,json.dumps({'run':run,'keys':[key for key,_ in selected]}))
        reserved=True
        spec=registry()[service]
        import importlib
        module=importlib.import_module(spec['module'])
        adapter=getattr(module,'FetchProvider' if service=='jina' else 'SearchProvider')
        with tempfile.TemporaryDirectory(prefix='provider-key-check-') as temporary:
            async def one(index,key,secret):
                directory=Path(temporary)/str(index);directory.mkdir(mode=0o700)
                keyfile=directory/'keys.json';keyfile.write_text(json.dumps({spec['key_field']:[secret]}));keyfile.chmod(0o600)
                cfg=dict(config,SERPER_KEY_POOL_FILE=str(keyfile),SERPER_KEY_STATE_FILE=str(directory/'state.json'),
                         SEARCH_POOL_NAMESPACE=f'key-check-{run}-{index}',SEARCH_POOL_WORKERS='1',
                         SERPER_PROXY_CONCURRENCY='1',SERPER_PROXY_QUEUE_LIMIT='1',SEARCH_POOL_MAX_UPSTREAM_ATTEMPTS='1')
                pool=ProviderPool(cfg,adapter(cfg));probes.append((key,pool))
                started=time.monotonic()
                try:
                    await pool.start()
                    payload={'url':'https://example.com/','extractMode':'text'} if service=='jina' else {'q':'example domain IANA','num':1}
                    import hashlib
                    encoded=json.dumps(payload,sort_keys=True,separators=(',',':'))
                    await pool.state.call('enqueue',id='check',payload=encoded,digest=hashlib.sha256(encoded.encode()).hexdigest())
                    async with asyncio.timeout(float(cfg['SEARCH_SERVER_TIMEOUT'])+2):
                        while True:
                            job=await pool.state.job('check')
                            if job.get('phase')=='done':break
                            await asyncio.sleep(.1)
                    body=json.loads(job['body']);usage=await pool.state.redis.hgetall(pool.state.prefix+'usage:'+key)
                    status=json.loads(await pool.state.redis.hget(pool.state.prefix+'state',key) or '{}')
                    return {'fingerprint':key[:12],'success':int(job['status'])==200,'http_status':int(job['status']),
                            'error_code':body.get('error_code'),'quarantine_reason':status.get('reason'),
                            'upstream_http_statuses':{k:v for k,v in usage.items() if k.startswith('http_')},
                            'upstream_attempts':int(usage.get('attempts',0)),'seconds':time.monotonic()-started}
                finally:
                    # Stop I/O while retaining the connection needed to transfer state.
                    if pool.dispatcher:
                        pool.dispatcher.cancel()
                        await asyncio.gather(pool.dispatcher,return_exceptions=True)
                    for task in list(pool.tasks):task.cancel()
                    await asyncio.gather(*pool.tasks,return_exceptions=True)
                    if pool.client:await pool.client.close()
            outcomes=await asyncio.gather(*(one(i,k,s) for i,(k,s) in enumerate(selected)), return_exceptions=True)
            rows=[({'fingerprint':selected[i][0][:12], 'success':False, 'error_type':type(item).__name__}
                   if isinstance(item, BaseException) else item) for i,item in enumerate(outcomes)]
        result={'service':service,'run_id':run,'before':{k:health[k] for k in ['total_keys','eligible_keys','verified_eligible_keys','unverified_eligible_keys']},'results':rows}
        output.write_text(json.dumps(result,indent=2)+'\n')
    finally:
        if reserved:
            merged={}
            for key,pool in probes:
                prefix=pool.state.prefix;r=pool.state.redis
                usage=await r.hgetall(prefix+'usage:'+key)
                hold=await r.zrange(prefix+'inflight:'+key,-1,-1,withscores=True)
                merged[key]={'usage':{k:float(v) for k,v in usage.items()},'state':await r.hget(prefix+'state',key) or '',
                             'hold_until':hold[0][1] if hold else 0}
            # Include a reserved key even if its probe never initialized.
            for key,_ in selected:merged.setdefault(key,{'usage':{},'state':'','hold_until':0})
            await state.redis.eval(RESTORE,1,state.prefix,json.dumps({'run':run,'keys':merged}))
            for _,pool in probes:
                await pool.state.redis.delete(pool.state.prefix+'keys')
                await pool.state.close()
        after=await state.health()
        if output.exists():
            result=json.loads(output.read_text());result['after']={k:after[k] for k in ['total_keys','eligible_keys','verified_eligible_keys','unverified_eligible_keys','quarantined_keys']}
            output.write_text(json.dumps(result,indent=2)+'\n')
        await state.close()
    print(json.dumps({'validated':len(rows),'successful':sum(r['success'] for r in rows),'after':result['after']},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--service',choices=['serper','jina'],required=True)
    p.add_argument('--expected',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();asyncio.run(validate(a.service,a.expected,a.output))
