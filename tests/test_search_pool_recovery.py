import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time

import httpx

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/validation'))
from search_pool_load import Sandbox


def wait_for(test, seconds=8):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if test():return
        time.sleep(.03)
    raise AssertionError('Condition did not become true')


def test_worker_kill_recovers_queue_and_never_replays_sent_request(tmp_path):
    with Sandbox(tmp_path,replicas=1,overrides={'SEARCH_POOL_WORKERS':2,'SERPER_PROXY_CONCURRENCY':2,
        'SERPER_KEY_CONCURRENCY':1,'SERPER_UPSTREAM_TIMEOUT':.5,'SEARCH_POOL_LEASE_SECONDS':1}) as s:
        # Enqueue via production state transition; no HTTP waiter is needed for durable work.
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Evaluation/services'))
        from search_pool_state import PoolState
        from search_pool import load_config
        async def enqueue():
            state=PoolState(load_config(s.env))
            try:
                for i in range(3):
                    payload=json.dumps({'q':f'crash-{i}','mock_delay':.4})
                    await state.call('enqueue',id=f'crash-{i}',payload=payload,digest=hashlib.sha256(payload.encode()).hexdigest())
            finally:await state.close()
        asyncio.run(enqueue())
        wait_for(lambda:len(s.stats()['rows'])==1 and s.redis.zcard(s.prefix+'active')==2)
        s.replicas[0].kill();s.replicas[0].wait()
        # Waiter, worker and queue owner can all disappear; another replica resumes the queue.
        _,url=s.add_replica()
        wait_for(lambda:s.health()['pending']==0)
        rows=s.stats()['rows']
        assert len(rows)==3
        assert len({r['q'] for r in rows})==3
        health=s.health()
        assert health['metrics']['billing_unknown']==1
        assert health['metrics']['recovered_queued']==1
        assert health['metrics']['recovered_uncertain']==1
        assert health['metrics']['success']==2
        assert s.http.post(url+'/search',json={'q':'after-crash'}).status_code==200


def test_redis_failure_fails_closed_and_aof_preserves_results(tmp_path):
    with Sandbox(tmp_path) as s:
        body={'q':'persisted'};headers={'Idempotency-Key':'persisted'}
        assert s.http.post(s.urls[0]+'/search',json=body,headers=headers).status_code==200
        s.redis_process.kill();s.redis_process.wait()
        response=s.http.post(s.urls[0]+'/search',json={'q':'must-not-send'})
        assert response.status_code==503
        assert response.json()['error_code']=='search_state_unavailable'
        assert len(s.stats()['rows'])==1
        s.start_redis()
        wait_for(lambda:s.http.get(s.urls[1]+'/health').status_code==200)
        assert s.http.post(s.urls[1]+'/search',json=body,headers=headers).status_code==200
        assert len(s.stats()['rows'])==1
        assert s.health()['metrics']['deduplicated']==1


def test_fencing_prevents_expired_worker_dispatch(tmp_path):
    with Sandbox(tmp_path,overrides={'SEARCH_POOL_WORKERS':1}) as s:
        for worker in s.replicas:worker.kill();worker.wait()
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Evaluation/services'))
        from search_pool_state import PoolState
        from search_pool import load_config
        async def scenario():
            state=PoolState(load_config(s.env))
            try:
                await state.call('enqueue',id='fenced',payload='{"q":"fenced"}',digest='digest')
                assert await state.call('claim',owner='old',lease=.05)
                await asyncio.sleep(.1)
                await state.call('recover')
                assert await state.call('claim',owner='new',lease=3)
                assert await state.call('admit',id='fenced',owner='old',lease=3,excluded={},attempt=1)==['lost']
                assert await state.call('renew',id='fenced',owner='old',lease=3)==0
                assert (await state.call('admit',id='fenced',owner='new',lease=3,excluded={},attempt=1))[0]=='ok'
            finally:await state.close()
        asyncio.run(scenario())
        assert len(s.stats()['rows'])==0


def test_mismatched_replica_cannot_multiply_limits(tmp_path):
    import subprocess
    from search_pool_load import SERVICES, free_port
    with Sandbox(tmp_path) as s:
        env=dict(s.env,SERPER_KEY_CONCURRENCY='50',SEARCH_POOL_PORT=str(free_port()))
        result=subprocess.run([sys.executable,str(SERVICES/'search_pool.py'),'serve'],env=env,
                              capture_output=True,text=True,timeout=10)
        assert result.returncode!=0
        assert 'configuration mismatch' in result.stderr
        assert s.health()['per_key_concurrency']==2


def test_reconfiguration_preserves_credentials_and_usage(tmp_path):
    import subprocess
    from search_pool_load import SERVICES, free_port
    with Sandbox(tmp_path) as s:
        assert s.http.post(s.urls[0]+'/search',json={'q':'before'}).status_code==200
        for proc in s.replicas:proc.terminate();proc.wait()
        s.env['SERPER_PROXY_CONCURRENCY']='32'
        result=subprocess.run([sys.executable,str(SERVICES/'search_pool.py'),'configure'],env=s.env,
                              capture_output=True,text=True,timeout=10)
        assert result.returncode==0,result.stderr
        _,url=s.add_replica()
        assert s.health()['max_concurrency']==32
        assert s.health()['metrics']['success']==1
        assert s.http.post(url+'/search',json={'q':'after'}).status_code==200


def test_queue_time_counts_against_original_request_deadline(tmp_path):
    from test_serper_proxy_concurrency import requests
    with Sandbox(tmp_path,overrides={'SEARCH_POOL_WORKERS':1,'SERPER_PROXY_CONCURRENCY':1,
        'SERPER_PROXY_QUEUE_TIMEOUT':.8,'SEARCH_SERVER_TIMEOUT':3}) as s:
        started=time.monotonic()
        replies=asyncio.run(requests(s,2,delay=.7))
        assert sum(r.status_code==200 for r in replies)==1
        assert sum(r.json().get('error_code')=='search_outcome_unknown' for r in replies)==1
        assert time.monotonic()-started<2
        assert s.health()['metrics']['upstream_attempts']==2


def test_paid_probe_report_works_on_redis6_and_cannot_accidentally_repeat(tmp_path):
    import subprocess
    from search_pool_load import ROOT
    with Sandbox(tmp_path/'sandbox') as s:
        output=tmp_path/'probe.json'
        command=[sys.executable,str(ROOT/'scripts/validation/search_pool_real_probe.py'),'--output',str(output)]
        result=subprocess.run(command,env=s.env,capture_output=True,text=True,timeout=20)
        assert result.returncode==0,result.stderr
        report=json.loads(output.read_text())
        assert report['upstream_attempts']==4
        assert report['confirmed_response_credits']==4
        assert report['idempotent_replay_verified']
        assert len(s.stats()['rows'])==4
        repeated=subprocess.run(command,env=s.env,capture_output=True,text=True,timeout=20)
        assert repeated.returncode!=0
        assert len(s.stats()['rows'])==4
