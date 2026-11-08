import asyncio
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io
import json
import sys
from pathlib import Path
import threading
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'Shared'),str(ROOT/'RL_training')]
from glasseek_shared.hermes.tool_errors import ToolFailure,http_json,exception_failure
from glasseek_training.web_tools import WebTools


class Opener:
    def __init__(self,body,status=200):self.body=body;self.status=status
    def open(self,request,timeout):
        if self.status!=200:
            raise urllib.error.HTTPError('http://service/test',self.status,'failure',{},io.BytesIO(self.body))
        return io.BytesIO(self.body)


def test_http_failure_preserves_status_reason_and_retryability():
    for status in (401,404,429,503):
        value=http_json(Opener(json.dumps({'success':False,'error':'Provider unavailable',
            'error_code':'provider_unavailable','retryable':True}).encode(),status),None,1)
        assert isinstance(value,ToolFailure)
        assert value.payload['http_status']==status and value.payload['error']=='Provider unavailable'
        assert value.payload['error_code']=='provider_unavailable'


def test_empty_invalid_and_non_json_fail_explicitly_but_zero_search_succeeds():
    for body in (b'',b'not json',b'{}',b'[]'):
        value=http_json(Opener(body),None,1)
        assert isinstance(value,ToolFailure) and value.payload['error_code']=='tool_invalid_response'
    assert http_json(Opener(b'{"organic":[]}'),None,1)=={'organic':[]}
    assert exception_failure(TimeoutError()).payload['error_code']=='tool_timeout'


def test_extract_failure_is_typed_not_inferred_from_webpage_json(monkeypatch):
    monkeypatch.setenv('GLASSEEK_SEARCH_POOL_ENDPOINT','http://127.0.0.1:1/search')
    tools=WebTools()
    tools._execute=lambda call:ToolFailure('DNS lookup failed','fetch_dns_error',True)
    bad=json.loads(asyncio.run(tools.execute({'name':'web_extract','arguments':{'urls':['https://example.com']}})))
    assert bad['results'][0]['error']=='DNS lookup failed'
    assert bad['results'][0]['error_code']=='fetch_dns_error'
    tools._execute=lambda call:'{"error":"This is an article example, not a transport failure"}'
    good=json.loads(asyncio.run(tools.execute({'name':'web_extract','arguments':{'urls':['https://example.com']}})))
    assert good['results'][0]['error'] is None and 'article example' in good['results'][0]['content']


def test_bridge_exception_returns_structured_output_and_records_failed_call(tmp_path,monkeypatch):
    from glasseek_shared.context.official.web import WebBridge
    monkeypatch.setenv('GLASSEEK_SEARCH_POOL_ENDPOINT','http://127.0.0.1:1/search')
    tools=WebTools()
    async def execute(call):raise TimeoutError()
    tools.execute=execute
    bridge=WebBridge(tmp_path,tools)
    try:
        request=urllib.request.Request(bridge.url+'/session/test',data=json.dumps({'name':'web_search','arguments':{'query':'test'}}).encode())
        try:urllib.request.urlopen(request,timeout=3)
        except urllib.error.HTTPError as exc:
            data=json.load(exc)
            assert data['success'] is False
            assert json.loads(data['output'])['error_code']=='tool_timeout'
            assert bridge.counts=={'web_search':1}
            assert json.loads(bridge.counts_path.read_text())==bridge.counts
        else:raise AssertionError('HTTP failure was hidden')
    finally:bridge.close()


def test_generated_web_helper_returns_failure_payload_to_model(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from glasseek_shared.context.official.run import run
    from litellm import ModelResponse
    monkeypatch.setenv('GLASSEEK_SEARCH_POOL_ENDPOINT','http://127.0.0.1:1/search')
    tools=WebTools()
    async def execute(call):raise TimeoutError()
    tools.execute=execute
    calls=[]
    async def query(agent,model,messages):
        calls.append(messages)
        if len(calls)==1:
            helper=str(Path(agent.glasseek_paths['workspace'])/'glasseek-web')
            message={'role':'assistant','content':'','tool_calls':[{'id':'request','type':'function','function':
                {'name':'bash','arguments':json.dumps({'command':helper+' search test'})}}]}
        else:
            returned=next(m['content'] for m in reversed(messages) if m['role']=='tool')
            assert 'tool_timeout' in returned and 'Traceback' not in returned
            message={'role':'assistant','content':'The tool reported a timeout.'}
        return ModelResponse(choices=[{'message':message,'finish_reason':'tool_calls' if len(calls)==1 else 'stop'}],
            usage={'prompt_tokens':100,'completion_tokens':10})
    args=SimpleNamespace(output=tmp_path,question='Check tool failure handling.',environment='local',
        model='test',base_url='',max_steps=4,context_budget=10000,resume_from=None)
    result=asyncio.run(run(args,query_hook=query,tools=tools))
    assert result['stop_reason']=='answer' and result['web_calls']=={'web_search':1}
