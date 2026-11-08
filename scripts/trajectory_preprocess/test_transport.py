import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from client import Client, ProviderError


class ProviderTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.provider={'url':'https://example.invalid/v1','key':'','model':'test','max_concurrent_requests':1}
        self.config={'providers':{'default':self.provider},'execution':{'workers':2},
                     'stages':{'inventory':{'provider':'default'}},'generation':{'attempts':3,'max_tokens':20}}
        self.client=Client(self.config,self.tmp.name,'test')

    def fake_session(self, responses):
        s=MagicMock();s.__enter__.return_value=s;s.request.side_effect=responses
        return s

    def test_balance_error_opens_circuit(self):
        response=MagicMock(status_code=402);s=self.fake_session([response])
        with patch('client.requests.Session',return_value=s):
            for _ in range(2):
                with self.assertRaises(ProviderError):self.client.request(self.provider,'POST','/chat/completions',{})
        self.assertEqual(s.request.call_count,1)

    def test_rate_limit_retries_same_request_with_backoff(self):
        limited=MagicMock(status_code=429,headers={'Retry-After':'3'})
        ok=MagicMock(status_code=200);ok.json.return_value={'usage':{}}
        s=self.fake_session([limited,ok]);body={'messages':[{'role':'user','content':'unchanged'}]}
        with patch('client.requests.Session',return_value=s),patch('client.time.sleep') as sleep,patch('client.random.random',return_value=0):
            result=self.client.request(self.provider,'POST','/chat/completions',body)
        self.assertEqual(result['_transport_attempts'],2)
        self.assertEqual([c.kwargs['json'] for c in s.request.call_args_list],[body,body])
        sleep.assert_called_once_with(3)

    def test_fatal_provider_error_is_not_a_semantic_retry(self):
        with patch.object(self.client,'request',side_effect=ProviderError(402)) as request:
            with self.assertRaises(RuntimeError):self.client.call('inventory','x','prompt','system',lambda v:v)
        self.assertEqual(request.call_count,1)
        rows=[json.loads(l) for l in (Path(self.tmp.name)/'calls.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['failure_category'],'provider_unavailable')


if __name__=='__main__':unittest.main()
