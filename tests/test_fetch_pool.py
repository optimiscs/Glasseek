import asyncio
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch, MagicMock
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/validation'),str(ROOT/'Evaluation/services'),str(ROOT/'Evaluation_backend')]
from fetch_pool_load import FetchSandbox
from test_search_pool_recovery import wait_for
from tools import url_safety


def test_rotation_cooldown_and_idempotency(tmp_path):
    with FetchSandbox(tmp_path,keys=['dead-1','dead-2','fake-1'],replicas=2) as s:
        body={'url':'https://example.com/good'};headers={'Idempotency-Key':'same'}
        reply=s.http.post(s.urls[0]+'/fetch',json=body,headers=headers)
        assert reply.status_code==200 and reply.json()['success']
        assert s.http.post(s.urls[1]+'/fetch',json=body,headers=headers).json()==reply.json()
        assert s.http.post(s.urls[1]+'/fetch',json={**body, 'jina_api_key':'must-not-persist'},headers=headers).json()==reply.json()
        assert all('must-not-persist' not in (s.redis.hget(k,'payload') or '') for k in s.redis.scan_iter(s.prefix+'job:*'))
        assert sum(r['charged'] for r in s.stats()['rows'])==1
        assert s.http.post(s.urls[1]+'/fetch',json={'url':'https://example.com/different'},headers=headers).status_code==409
        assert s.http.post(s.urls[0]+'/fetch',json={'url':'https://example.com/throttle?status=429'}).json()['error_code']=='fetch_rate_limited'
        before=len(s.stats()['rows'])
        assert s.http.post(s.urls[0]+'/fetch',json={'url':'https://example.com/cooldown'}).status_code==503
        assert len(s.stats()['rows'])==before
        # Advance the cooldown timestamp, retaining the rate_limited reason.
        # Unlike permanent quarantine, it becomes eligible without restore-key.
        from search_pool_state import key_id
        s.redis.hset(s.prefix+'state', key_id('fake-1'), json.dumps({'reason':'rate_limited','until':time.time()-1}))
        assert s.http.post(s.urls[1]+'/fetch',json={'url':'https://example.com/after-cooldown'}).status_code==200


def test_crash_queue_recovery_and_no_replay(tmp_path):
    with FetchSandbox(tmp_path,replicas=1,overrides={'SEARCH_POOL_WORKERS':1,'SERPER_PROXY_CONCURRENCY':1,
            'SERPER_KEY_CONCURRENCY':1,'SERPER_UPSTREAM_TIMEOUT':.5,'SEARCH_POOL_LEASE_SECONDS':1}) as s:
        from provider_services import load_service
        from search_pool_state import PoolState
        import os
        async def enqueue():
            with patch.dict(os.environ,{'GLASSEEK_FETCH_POOL_CONFIG':str(s.directory/'fetch-config.json')}):
                state=PoolState(load_service('jina'))
            try:
                for i in range(2):
                    payload=json.dumps({'url':f'https://example.com/crash-{i}?delay=.4'})
                    await state.call('enqueue',id=f'crash-{i}',payload=payload,digest=str(i))
            finally:await state.close()
        asyncio.run(enqueue());wait_for(lambda:len(s.stats()['rows'])==1)
        s.replicas[0].kill();s.replicas[0].wait();s.add_replica()
        wait_for(lambda:s.health()['pending']==0)
        assert len(s.stats()['rows'])==2
        assert s.health()['metrics']['billing_unknown']==1
        assert s.redis.hget(s.prefix+'job:crash-0','body').find('fetch_outcome_unknown')>=0


def test_global_key_limits_across_replicas(tmp_path):
    with FetchSandbox(tmp_path,keys=['fake-1'],overrides={'SERPER_KEY_RPS':4,'SERPER_KEY_CONCURRENCY':2}) as s:
        import httpx
        async def run():
            async with httpx.AsyncClient(trust_env=False,timeout=15) as c:
                return await asyncio.gather(*(c.post(s.urls[i%2]+'/fetch',json={'url':f'https://example.com/rate-{i}?delay=.3'}) for i in range(8)))
        assert all(r.status_code==200 for r in asyncio.run(run()))
        rows=sorted(s.stats()['rows'],key=lambda r:r['at'])
        assert all(b['at']-a['at']>=.23 for a,b in zip(rows,rows[1:]))
        assert s.stats()['peak']['fake-1']<=2


def test_dns_nxdomain_servfail_and_private_distinct():
    import requests,socket
    for status,code in [(3,'dns_not_found'),(2,'dns_resolution_failed')]:
        url_safety._doh_cache.clear()
        session=MagicMock();session.get.return_value.json.return_value={'Status':status}
        with patch.object(requests,'Session',return_value=session),patch.object(url_safety.socket,'getaddrinfo',side_effect=socket.gaierror()):
            result=url_safety.check_external_fetch_url('https://missing.test')
            assert not result[0] and result[1]==code
    assert url_safety.check_external_fetch_url('http://127.0.0.1/')[1]=='private_address'


def test_central_dns_singleflight_and_negative_cache(tmp_path):
    with FetchSandbox(tmp_path,replicas=1) as s:
        import os
        from provider_services import load_service
        from search_pool_state import PoolState
        from fetch_dns import FetchDNS
        async def run():
            with patch.dict(os.environ,{'GLASSEEK_FETCH_POOL_CONFIG':str(s.directory/'fetch-config.json')}):state=PoolState(load_service('jina'))
            try:
                one,two=FetchDNS(state),FetchDNS(state)
                with patch('fetch_dns.check_external_fetch_url',return_value=(False,'dns_not_found','NXDOMAIN')) as check:
                    results=await asyncio.gather(*[(one if i%2 else two).check('https://missing.test/'+str(i)) for i in range(64)])
                    assert all(r['error_code']=='dns_not_found' for r in results)
                    assert check.call_count==1
            finally:await state.close()
        asyncio.run(run())


def test_pooled_client_preserves_errors_and_never_replays_transport(monkeypatch):
    import httpx
    from plugins.web.fetch_server import provider
    monkeypatch.setenv('GLASSEEK_FETCH_POOL_CLIENT','1')
    calls=[]
    async def handler(request):
        calls.append(request)
        assert request.headers.get('Idempotency-Key')
        return httpx.Response(503,json={'success':False,'error':'unknown outcome','error_code':'fetch_outcome_unknown','retryable':False})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            result=await provider._fetch_one(c,'http://pool/fetch','https://example.com',
                    {'max_retries':3,'extract_mode':'text','jina_api_key':'','llm_base_url':'','llm_model':'','llm_api_key':''},asyncio.Semaphore(1))
            assert result['error_code']=='fetch_outcome_unknown' and result['retryable'] is False
    asyncio.run(run());assert len(calls)==1


def test_one_process_hosts_both_providers_and_preserves_quarantine_on_restart(tmp_path):
    import subprocess,os
    from search_pool_load import free_port,SERVICES
    with FetchSandbox(tmp_path,keys=['dead-1','fake-1'],replicas=1) as s:
        # Force discovery of the exhausted key without needing production secrets.
        for i in range(3):
            assert s.http.post(s.urls[0]+'/fetch',json={'url':f'https://example.com/first-{i}'}).status_code==200
        assert s.health()['quarantined_keys'].get('quota_exhausted')==1
        s.replicas[0].terminate();s.replicas[0].wait(timeout=10)
        search_port,fetch_port=free_port(),free_port()
        env=dict(s.env,SEARCH_POOL_PORT=str(search_port),SEARCH_POOL_NAMESPACE=s.env['SEARCH_POOL_NAMESPACE']+'-serper',
                 GLASSEEK_FETCH_POOL_CONFIG=str(s.directory/'fetch-config.json'),GLASSEEK_FETCH_URL=f'http://127.0.0.1:{fetch_port}')
        proc=s.process([sys.executable,str(SERVICES/'provider_services.py'),'serve','--services','serper,jina'],env,'combined')
        search=f'http://127.0.0.1:{search_port}';fetch=f'http://127.0.0.1:{fetch_port}'
        s.ready(proc,search);s.ready(proc,fetch)
        assert s.http.get(fetch+'/health').json()['quarantined_keys']['quota_exhausted']==1
        before=sum(r['key']=='dead-1' for r in s.stats()['rows'])
        assert s.http.post(fetch+'/fetch',json={'url':'https://example.com/after-restart'}).status_code==200
        assert sum(r['key']=='dead-1' for r in s.stats()['rows'])==before
        assert s.http.post(search+'/search',json={'q':'independent'}).status_code==200
        proc.terminate();proc.wait(timeout=10)


def test_key_validation_uses_one_attempt_and_merges_into_production_state(tmp_path,monkeypatch):
    from provider_keys import validate
    with FetchSandbox(tmp_path/'sandbox',keys=['fake-new-key'],replicas=1) as s:
        monkeypatch.setenv('GLASSEEK_FETCH_POOL_CONFIG',str(s.directory/'fetch-config.json'))
        output=tmp_path/'validation.json'
        asyncio.run(validate('jina',1,output))
        report=json.loads(output.read_text())
        assert report['results'][0]['success'] is True
        assert report['results'][0]['upstream_attempts']==1
        assert s.health()['verified_eligible_keys']==1
        assert len(s.stats()['rows'])==1
        assert s.redis.get(s.prefix+'key_probe_lock') is None
        assert not list(s.redis.scan_iter('{key-check-*}:keys'))
