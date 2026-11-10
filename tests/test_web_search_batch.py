"""Real search entrypoints with mocked providers; no network or model calls."""
import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock,patch

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/p) for p in ('Evaluation_backend','RL_training','Shared')]
from tools import web_tools as web
from glasseek_training.web_tools import WebTools,parse_tool_calls
from glasseek_shared.hermes.tool_errors import ToolFailure
from glasseek_shared.search_protocol import validate_queries


@pytest.mark.parametrize('encoded',[False,True])
def test_evaluation_batch_preserves_order_duplicates_and_partial_failure(encoded):
    def search(q,limit):
        return {'success':False,'error':'unavailable'} if q=='B' else {'success':True,'data':{'web':[{'title':q,'url':'https://example.org/'+q}]}}
    provider=SimpleNamespace(name='test',supports_search=lambda:True,search=Mock(side_effect=search))
    with ExitStack() as stack:
        stack.enter_context(patch.object(web,'_ensure_web_plugins_loaded'))
        stack.enter_context(patch.object(web,'_get_search_backend',return_value='test'))
        stack.enter_context(patch('agent.web_search_registry.get_provider',return_value=provider))
        stack.enter_context(patch.object(web,'_debug'))
        stack.enter_context(patch('tools.interrupt.is_interrupted',return_value=False))
        query=['A','B','A'];query=json.dumps(query) if encoded else query
        batch=json.loads(web.registry.get_entry('web_search').handler({'query':query,'limit':7}))
        assert [r['query'] for r in batch['results']]==['A','B','A']
        assert batch['results'][1]['result']=={'success':False,'error':'unavailable'}
        assert batch['results'][2]['result']['data']['web'][0]['title']=='A'
        assert len(provider.search.call_args_list)==3
        assert all(c.args[1]==7 for c in provider.search.call_args_list)
        single=json.loads(web.web_search_tool('A',limit=7))
        assert single['success'] is True and 'results' not in single


@pytest.mark.parametrize('bad',[[],['A']*6,['A',''],[True],' '])
def test_evaluation_bad_batch_is_rejected_before_provider(bad):
    with patch.object(web,'_ensure_web_plugins_loaded') as provider:
        assert 'error' in json.loads(web.web_search_tool(bad))
        provider.assert_not_called()


@pytest.mark.parametrize('mode',['raw','ref','hybrid','evidence'])
def test_training_batch_preserves_query_status_and_source_references(mode):
    seen=[]
    class Local(WebTools):
        def _execute(self,call):
            seen.append(call)
            return ToolFailure('unavailable','tool_timeout',True) if call['arguments']['query']=='B' else json.dumps({'data':{'web':[{'url':'https://example.org/a','title':'A'}]}})
    with patch.dict(os.environ,{'GLASSEEK_SEARCH_POOL_ENDPOINT':'http://pool/search'}):
        tool=Local(url_mode=mode)
    result=json.loads(asyncio.run(tool.execute({'name':'web_search','arguments':{'query':['A','B'],'limit':3}})))
    assert [r['query'] for r in result['results']]==['A','B']
    assert result['results'][1]['result']['error_code']=='tool_timeout'
    assert all(c['arguments']['limit']==3 for c in seen)
    seen.clear()
    failure=json.loads(asyncio.run(tool.execute({'name':'web_search','arguments':{'query':['A']*6}})))
    assert failure['error_code']=='invalid_tool_arguments' and seen==[]


def test_xml_query_array_is_typed_and_empty_queries_are_invalid():
    text='<tool_call><function=web_search><parameter=query>["A", "B"]</parameter></function></tool_call>'
    assert parse_tool_calls(text)[0]['arguments']['query']==['A','B']
    with pytest.raises(ValueError):validate_queries(['A',' '])


def test_qwen_raw_string_parameters_preserve_entities_and_whitespace():
    text='<tool_call><function=web_search><parameter=query>  A &amp; B  </parameter><parameter=limit>\n3\n</parameter></function></tool_call>'
    assert parse_tool_calls(text)[0]['arguments']=={'query':'  A &amp; B  ','limit':3}


@pytest.mark.parametrize('query', [
    'Paris & London',
    'Paris &amp; London',
    'A &lt; B &gt; C &#38; D',
    '  中文 &amp; London  ',
    '\nA &amp; B\n',
    ['  A &amp; B  ', 'C &#38; D', '  A &amp; B  '],
])
def test_canonical_xml_preserves_query_values_through_execution(query, monkeypatch):
    from tool_xml import serialize

    seen = []
    class Local(WebTools):
        def _execute(self, call):
            seen.append(call)
            return json.dumps({'data': {'web': []}})

    monkeypatch.setenv('GLASSEEK_SEARCH_POOL_ENDPOINT', 'http://pool/search')
    original = {'name': 'web_search', 'arguments': {'query': query, 'limit': 3}}
    parsed = parse_tool_calls(serialize(original))
    assert parsed == [original]
    result = json.loads(asyncio.run(Local(url_mode='raw').execute(parsed[0])))
    queries = query if isinstance(query, list) else [query]
    # Batch workers may start in any order; retain every exact provider argument,
    # including duplicates, while the public result must retain input order.
    from collections import Counter
    assert Counter(call['arguments']['query'] for call in seen) == Counter(queries)
    assert all(call['arguments']['limit'] == 3 for call in seen)
    if isinstance(query, list):
        assert [row['query'] for row in result['results']] == query


def test_runtime_rejects_legacy_outer_control_wrapper():
    call='<tool_call><function=web_search><parameter=query>A</parameter></function></tool_call>'
    with pytest.raises(ValueError,match='Noncanonical'):
        parse_tool_calls('<tool_calls_begin>\n'+call+'\n</tool_calls_end>')
    assert parse_tool_calls('<think>Mention <tool_calls_begin> in quoted reasoning.</think>\n'+call)[0]['arguments']['query']=='A'

@pytest.mark.parametrize('encoded',[False,True])
def test_search_server_plugin_override_keeps_batch_queries(encoded):
    from plugins.web import search_server as plugin
    calls=[]
    def search(q,limit):
        calls.append((q,limit))
        return {'success':False,'error':'query B failed'} if q=='B' else {'success':True,'data':{'web':[]}}
    ctx=Mock();plugin.register(ctx)
    handler=ctx.register_tool.call_args.kwargs['handler']
    with patch.object(web,'_ensure_web_plugins_loaded'), patch.object(web,'_get_search_backend',return_value='search_server'), patch('agent.web_search_registry.get_provider',return_value=plugin._SEARCH), patch.object(plugin._SEARCH,'search',side_effect=search), patch('tools.interrupt.is_interrupted',return_value=False), patch.dict(os.environ,{'GLASSEEK_SEARCH_TOP10':'1'}):
        q=['A','B','A'];result=json.loads(handler({'query':json.dumps(q) if encoded else q,'limit':3}))
    assert [r['query'] for r in result['results']]==q
    assert result['results'][1]['result']['success'] is False
    assert sorted(calls)==[('A',10),('A',10),('B',10)]


def test_search_provider_rejects_array_instead_of_stringifying():
    from glasseek_shared.hermes.search_provider import _search_server_response
    with patch('glasseek_shared.hermes.search_provider._read_search_server_cfg') as config:
        result=_search_server_response(['A','B'])
    assert result['success'] is False
    config.assert_not_called()


def test_batch_discovers_plugins_before_worker_threads():
    import threading
    main_thread=threading.get_ident();discovery=[]
    provider=SimpleNamespace(name='test',supports_search=lambda:True,search=lambda q,limit:{'success':True,'data':{'web':[]}})
    with patch.object(web,'_ensure_web_plugins_loaded',side_effect=lambda:discovery.append(threading.get_ident())), patch.object(web,'_get_search_backend',return_value='test'), patch('agent.web_search_registry.get_provider',return_value=provider), patch('tools.interrupt.is_interrupted',return_value=False):
        result=json.loads(web.web_search_tool(['A','B']))
    assert discovery[0]==main_thread
    assert len(result['results'])==2


def test_viewer_batch_preserves_duplicate_queries_and_partial_failure():
    import quest_viewer_adapter,quest_viewer_checkpoint_adapter
    batch={'results':[{'query':'A','result':{'success':True,'data':{'web':[{'url':'https://example.org','title':'A'}]}}},{'query':'B','result':{'success':False,'error':'timeout'}},{'query':'A','result':{'success':True,'data':{'web':[]}}}]}
    for module in (quest_viewer_adapter,quest_viewer_checkpoint_adapter):
        _,tool=module.tool_view('web_search',{'query':['A','B','A']},json.dumps(batch))
        assert [q['query'] for q in tool['queries']]==['A','B','A']
        assert tool['queryCount']==3 and tool['resultCount']==1
        assert tool['status']=='partial'
        assert tool['queries'][1]['count'] is None
