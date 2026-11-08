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
from tools import url_safety as safety
import quest_viewer_checkpoint_adapter as viewer
from plugins.web.search_server import provider

class Repairs(unittest.TestCase):
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
