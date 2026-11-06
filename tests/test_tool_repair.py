import asyncio
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

P=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(P/'Evaluation/services'),str(P/'Evaluation_backend'),str(P/'Evaluation')]
import serper_proxy as proxy
from tools import url_safety as safety
import quest_viewer_checkpoint_adapter as viewer
from plugins.web.search_server import provider

class Repairs(unittest.TestCase):
 def setUp(self):
  self.usage_dir=tempfile.TemporaryDirectory();self.addCleanup(self.usage_dir.cleanup)
  self.usage_patch=patch.object(proxy,'USAGE_PATH',Path(self.usage_dir.name)/'usage.jsonl');self.usage_patch.start();self.addCleanup(self.usage_patch.stop)
  while proxy.idle_connections:
   proxy.idle_connections.pop()[0].close()
 def test_expired_exhausted_key_stays_disabled(self):
  proxy.key_state={proxy.key_id('dead'):{'reason':'quota_exhausted','until':0}}
  self.assertTrue(proxy.unavailable(proxy.key_id('dead'),time.time()))
  proxy.key_state={proxy.key_id('cool'):{'reason':'rate_limited','until':0}}
  self.assertFalse(proxy.unavailable(proxy.key_id('cool'),time.time()))
 def test_durable_credit_records(self):
  ident=proxy.key_id('record-test')
  proxy.record_usage(ident,'attempt','req',requested_num=11)
  proxy.record_usage(ident,'response','req',status=200,credits=1)
  rows=[json.loads(x) for x in proxy.USAGE_PATH.read_text().splitlines()]
  proxy.lifetime_metrics.clear()
  for r in rows:proxy.count_usage(r)
  self.assertEqual(proxy.lifetime_metrics[ident]['attempts'],1)
  self.assertEqual(proxy.lifetime_metrics[ident]['confirmed_credits'],1)
 def test_transport_retry_is_bounded_and_uses_another_eligible_key(self):
  import collections
  proxy.key_state={};proxy.inflight_keys=collections.Counter();proxy.key_next_request={};proxy.cursor=0;proxy.next_request=0
  response=MagicMock();response.status=200;response.read.return_value=b'{"organic":[]}';response.getheader.return_value=None
  with patch.object(proxy,'search_keys',return_value=['a','b']),patch.object(proxy,'RPS',0),patch.object(proxy.http.client,'HTTPSConnection') as conn:
   conn.return_value.getresponse.side_effect=[TimeoutError(),response]
   self.assertEqual(proxy.upstream_search({'q':'x'})[0],200)
   self.assertEqual(conn.return_value.request.call_count,2)
   self.assertEqual(sum(proxy.inflight_keys.values()),0)
 def test_each_key_has_independent_rate_and_concurrency(self):
  import collections,concurrent.futures,threading
  proxy.key_state={};proxy.inflight_keys=collections.Counter();proxy.key_next_request={};proxy.cursor=0
  starts=[];guard=threading.Lock()
  def work(_):
   selected=proxy.acquire_key(['a','b'],set(),time.monotonic()+3)
   self.assertIsNotNone(selected)
   key,ident=selected
   with guard:starts.append((key,time.monotonic()))
   with proxy.lock:self.assertLessEqual(proxy.inflight_keys[ident],2)
   time.sleep(.05)
   with proxy.lock:proxy.inflight_keys[ident]-=1
  with patch.object(proxy,'KEY_CONCURRENCY',2),patch.object(proxy,'KEY_RPS',50):
   with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(work,range(24)))
  for key in ['a','b']:
   ticks=sorted(t for k,t in starts if k==key)
   self.assertGreater(len(ticks),1)
   self.assertTrue(all(b-a>=.019 for a,b in zip(ticks,ticks[1:])))
  self.assertEqual(sum(proxy.inflight_keys.values()),0)
 def test_key_quarantine_and_no_repeat(self):
  with tempfile.TemporaryDirectory() as directory, patch.object(proxy,'STATE_PATH',Path(directory)/'state.json'), patch.object(proxy,'search_keys',return_value=['dead','good']), patch.object(proxy,'RPS',10000), patch.object(proxy,'KEY_RPS',10000):
   proxy.key_state={};proxy.inflight_keys=__import__("collections").Counter();proxy.key_next_request={};proxy.cursor=0;proxy.next_request=0
   responses=[]
   for status,body in [(400,b'{"message":"Not enough credits"}'),(200,b'{"organic":[]}'),(200,b'{"organic":[]}')]:
    r=MagicMock();r.status=status;r.read.return_value=body;r.getheader.return_value=None;responses.append(r)
   with patch.object(proxy.http.client,'HTTPSConnection') as conn:
    conn.return_value.getresponse.side_effect=responses
    self.assertEqual(proxy.upstream_search({'q':'test'})[0],200)
    self.assertEqual(proxy.upstream_search({'q':'test'})[0],200)
    keys=[c.kwargs['headers']['X-API-KEY'] for c in conn.return_value.request.call_args_list]
    self.assertEqual(keys,['dead','good','good'])
   state=json.loads(proxy.STATE_PATH.read_text());self.assertEqual(state[proxy.key_id('dead')]['reason'],'quota_exhausted')
   self.assertNotIn('dead',proxy.STATE_PATH.read_text())
 def test_429_cools_down_without_pool_sweep(self):
  with tempfile.TemporaryDirectory() as directory, patch.object(proxy,'STATE_PATH',Path(directory)/'state.json'),patch.object(proxy,'search_keys',return_value=['a','b']):
   proxy.key_state={};proxy.inflight_keys=__import__("collections").Counter();proxy.key_next_request={};proxy.cursor=0;proxy.next_request=0
   with patch.object(proxy.http.client,'HTTPSConnection') as conn:
    r=conn.return_value.getresponse.return_value;r.status=429;r.read.return_value=b'{}';r.getheader.return_value='120'
    status,body=proxy.upstream_search({'q':'x'})
    self.assertEqual(status,503);self.assertEqual(json.loads(body)['error_code'],'search_rate_limited');self.assertEqual(conn.call_count,1)
    self.assertGreater(proxy.key_state[proxy.key_id('a')]['until'],time.time()+100)
 def test_bounded_queue_and_slot_cleanup(self):
  class Request:
   async def json(self):return {'q':'x'}
   async def is_disconnected(self):return False
  def slow(_):time.sleep(.05);return 200,b'{"organic":[]}'
  async def run():
   proxy.slots=asyncio.Semaphore(2);proxy.pending=0
   with patch.object(proxy,'MAX_CONCURRENCY',2),patch.object(proxy,'MAX_QUEUE',2),patch.object(proxy,'upstream_search',side_effect=slow):
    replies=await asyncio.gather(*(proxy.search(Request()) for _ in range(12)))
    self.assertEqual(sum(x.status_code==200 for x in replies),4)
    self.assertEqual(sum(x.status_code==503 for x in replies),8)
    self.assertEqual(proxy.pending,0)
  asyncio.run(run())
 def test_private_address_never_falls_back(self):
  local=[(socket.AF_INET,socket.SOCK_STREAM,6,'',('10.0.0.1',0))]
  with patch.object(safety.socket,'getaddrinfo',return_value=local),patch.object(safety,'_doh_public_addresses') as doh:
   result=safety.check_external_fetch_url('https://example.org')
   self.assertEqual(result[1],'private_address');doh.assert_not_called()
 def test_poisoned_and_failed_dns_require_public_verification(self):
  local=[(socket.AF_INET6,socket.SOCK_STREAM,6,'',('2001::1',0,0,0))]
  with patch.object(safety.socket,'getaddrinfo',return_value=local),patch.object(safety,'_doh_public_addresses',return_value=[ipaddress.ip_address('1.1.1.1')]):
   self.assertTrue(safety.check_external_fetch_url('https://example.org')[0])
  with patch.object(safety.socket,'getaddrinfo',side_effect=socket.gaierror()),patch.object(safety,'_doh_public_addresses',side_effect=safety.DNSVerificationError('resolver down')):
   result=safety.check_external_fetch_url('https://example.org');self.assertFalse(result[0]);self.assertEqual(result[1],'dns_verification_failed')
 def test_doh_proxy_cache_and_fallback(self):
  import requests
  safety._doh_cache.clear()
  sessions=[MagicMock(),MagicMock()]
  sessions[0].get.side_effect=requests.ConnectionError()
  response=MagicMock();response.json.return_value={'Status':0,'Answer':[{'type':1,'data':'1.1.1.1','TTL':60}]};sessions[1].get.return_value=response
  with patch.dict(os.environ,{'GLASSEEK_DOH_PROXY':'http://127.0.0.1:18200'}),patch.object(requests,'Session',side_effect=sessions) as factory:
   self.assertTrue(safety._doh_public_addresses('example.org'));self.assertTrue(safety._doh_public_addresses('example.org'))
   self.assertEqual(factory.call_count,2);self.assertFalse(sessions[1].trust_env)
   sessions[1].proxies.update.assert_called_with({'https':'http://127.0.0.1:18200'})
 def test_structured_throttle_is_not_retried(self):
  response=MagicMock();response.json.return_value={'success':False,'error':'cooldown','error_code':'search_rate_limited','retry_after':60,'retryable':True}
  cfg={'endpoint':'http://localhost/search','api_key':'test','max_retries':3,'timeout':30,'retry_delay':2}
  with patch.object(provider,'_read_search_server_cfg',return_value=cfg),patch.object(provider.httpx,'post',return_value=response) as post:
   result=provider._search_server_response('test');self.assertEqual(result['error_code'],'search_rate_limited');self.assertEqual(post.call_count,1)
 def test_viewer_distinguishes_failed_empty_pending(self):
  wrapped='<untrusted_tool_result>'+json.dumps({'success':False,'error':'timeout'})+'\n[Tool loop warning]</untrusted_tool_result>'
  _,bad=viewer.tool_view('web_search',{'query':'x'},wrapped);self.assertEqual(bad['queries'][0]['status'],'error');self.assertIsNone(bad['queries'][0]['count'])
  _,empty=viewer.tool_view('web_search',{'query':'x'},'{"success":true,"data":{"web":[]}}');self.assertEqual(empty['queries'][0]['status'],'success');self.assertEqual(empty['queries'][0]['count'],0)
  _,pending=viewer.tool_view('web_search',{'query':'x'},'');self.assertEqual(pending['queries'][0]['status'],'pending')

if __name__=='__main__':unittest.main()
