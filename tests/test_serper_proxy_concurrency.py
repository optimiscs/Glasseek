"""Exercise shared limits across real service processes, never the paid provider."""
import asyncio
import sys
from pathlib import Path
import time

import httpx
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/validation'))
from search_pool_load import Sandbox


async def requests(s, count, delay=.1, same=False):
    async with httpx.AsyncClient(trust_env=False,timeout=20) as client:
        return await asyncio.gather(*(client.post(s.urls[i%len(s.urls)]+'/search',
            json={'q':'same' if same else f'q{i}','mock_delay':delay},
            headers={'Idempotency-Key':'same' if same else f'id{i}'}) for i in range(count)))


def test_limits_are_global_across_replicas(tmp_path):
    with Sandbox(tmp_path,keys=['a','b'],overrides={'SERPER_KEY_RPS':10,'SERPER_PROXY_RPS':15}) as s:
        responses=asyncio.run(requests(s,16))
        assert all(r.status_code==200 for r in responses)
        stats=s.stats()
        assert stats['peak']['total']<=4
        for key in ('a','b'):
            assert stats['peak'][key]<=2
            times=sorted(r['at'] for r in stats['rows'] if r['key']==key)
            assert all(y-x>=.085 for x,y in zip(times,times[1:]))
        times=sorted(r['at'] for r in stats['rows'])
        assert all(y-x>=.05 for x,y in zip(times,times[1:]))
        assert s.health()['pending']==0


def test_concurrent_duplicate_requests_bill_once(tmp_path):
    with Sandbox(tmp_path) as s:
        responses=asyncio.run(requests(s,20,same=True))
        assert all(r.status_code==200 for r in responses)
        assert len(s.stats()['rows'])==1
        assert s.health()['metrics']['deduplicated']==19
        bad=s.http.post(s.urls[1]+'/search',json={'q':'different'},headers={'Idempotency-Key':'same'})
        assert bad.status_code==409


def test_queue_is_bounded_and_capacity_recovers(tmp_path):
    with Sandbox(tmp_path,overrides={'SERPER_PROXY_CONCURRENCY':2,'SERPER_PROXY_QUEUE_LIMIT':2}) as s:
        responses=asyncio.run(requests(s,20,delay=.7))
        assert sum(r.status_code==200 for r in responses)==4
        assert sum(r.json().get('error_code')=='search_queue_full' for r in responses)==16
        assert s.health()['pending']==0
        assert s.http.post(s.urls[1]+'/search',json={'q':'after'}).status_code==200


def test_queue_deadline_and_error_body(tmp_path):
    with Sandbox(tmp_path,overrides={'SERPER_PROXY_CONCURRENCY':1,'SERPER_PROXY_QUEUE_TIMEOUT':.1}) as s:
        responses=asyncio.run(requests(s,4,delay=.5))
        assert sum(r.status_code==200 for r in responses)==1
        assert sum(r.json().get('error_code')=='search_queue_timeout' for r in responses)==3
        assert s.health()['pending']==0


def test_attempt_budget_is_global(tmp_path):
    with Sandbox(tmp_path,overrides={'SEARCH_POOL_MAX_UPSTREAM_ATTEMPTS':3}) as s:
        responses=asyncio.run(requests(s,12))
        assert sum(r.status_code==200 for r in responses)==3
        assert len(s.stats()['rows'])==3
        assert s.health()['metrics']['upstream_attempts']==3
