import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/validation'))
from search_pool_load import Sandbox, SERVICES


def test_quota_rotates_and_shared_quarantine_survives_restart(tmp_path):
    dead=next('dead'+str(i) for i in range(100) if hashlib.sha256(('dead'+str(i)).encode()).hexdigest()<hashlib.sha256(b'working').hexdigest())
    with Sandbox(tmp_path,keys=[dead,'working']) as s:
        assert s.http.post(s.urls[0]+'/search',json={'q':'one'}).status_code==200
        assert [r['key'] for r in s.stats()['rows']]==[dead,'working']
        s.replicas[0].kill();s.replicas[0].wait()
        _,url=s.add_replica()
        assert s.http.post(url+'/search',json={'q':'two'}).status_code==200
        assert [r['key'] for r in s.stats()['rows']]==[dead,'working','working']
        assert s.health()['quarantined_keys']=={'quota_exhausted':1}
        assert s.health()['metrics']['upstream_attempts']==3


def test_429_cools_key_without_replaying_current_request(tmp_path):
    with Sandbox(tmp_path,keys=['a','b']) as s:
        response=s.http.post(s.urls[0]+'/search',json={'q':'limited','mock_status':429})
        assert response.json()['error_code']=='search_rate_limited'
        assert len(s.stats()['rows'])==1
        assert s.http.post(s.urls[1]+'/search',json={'q':'next'}).status_code==200
        assert len({r['key'] for r in s.stats()['rows']})==2
        assert s.health()['quarantined_keys']=={'rate_limited':1}


def test_ambiguous_timeout_is_not_retried_or_double_billed(tmp_path):
    with Sandbox(tmp_path,keys=['a','b'],overrides={'SERPER_UPSTREAM_TIMEOUT':.2,'SEARCH_POOL_LEASE_SECONDS':1}) as s:
        body={'q':'timeout','mock_delay':.7};headers={'Idempotency-Key':'timeout-id'}
        response=s.http.post(s.urls[0]+'/search',json=body,headers=headers)
        assert response.json()['error_code']=='search_outcome_unknown'
        assert response.json()['retryable'] is False
        response=s.http.post(s.urls[1]+'/search',json=body,headers=headers)
        assert response.json()['error_code']=='search_outcome_unknown'
        assert len(s.stats()['rows'])==1
        assert s.health()['metrics']['billing_unknown']==1


def test_keys_sync_and_restore_are_online(tmp_path):
    with Sandbox(tmp_path,keys=['dead-a']) as s:
        assert s.http.post(s.urls[0]+'/search',json={'q':'one'}).status_code==503
        file=Path(s.env['SERPER_KEY_POOL_FILE']);file.write_text('{"serper":["working"]}')
        result=subprocess.run([sys.executable,str(SERVICES/'search_pool.py'),'sync-keys'],env=s.env,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        assert s.http.post(s.urls[1]+'/search',json={'q':'two'}).status_code==200
        ident=hashlib.sha256(b'working').hexdigest()
        s.redis.hset(s.prefix+'state',ident,json.dumps({'reason':'quota_exhausted','until':0}))
        result=subprocess.run([sys.executable,str(SERVICES/'search_pool.py'),'restore-key','--key-id',ident],env=s.env,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        assert s.http.post(s.urls[1]+'/search',json={'q':'three'}).status_code==200
