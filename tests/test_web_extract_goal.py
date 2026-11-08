"""Goal propagation without live network requests or paid model calls."""
import asyncio
import copy
import io
import json
import os
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ('Evaluation_backend', 'Shared', 'RL_training')]
from tools import web_tools as web
from glasseek_shared.hermes.url_registry import URLRegistry, annotate_schemas
from glasseek_training.web_tools import WebTools, search_pool_worker_env


@pytest.mark.parametrize('chunk', [False, True])
def test_goal_reaches_normal_and_chunk_prompt_without_changing_default(chunk):
    async def run(goal):
        with patch.object(web, '_resolve_web_extract_auxiliary', return_value=(object(), 'test', {})), \
             patch.object(web, 'async_call_llm', new_callable=AsyncMock, return_value='Evidence') as llm, \
             patch.object(web, 'extract_content_or_reasoning', side_effect=lambda x:x):
            assert await web._call_summarizer_llm('Source text', 'Title: Page\n', None, is_chunk=chunk, goal=goal) == 'Evidence'
            return llm.call_args.kwargs['messages']
    plain = asyncio.run(run(''))
    goal = 'Find dates & contradictory figures; 中文'
    guided = asyncio.run(run(goal))
    assert guided[1]['content'].endswith(plain[1]['content'])
    assert json.dumps(goal, ensure_ascii=False) in guided[1]['content']
    assert 'prior knowledge' in guided[0]['content']
    assert 'conflicting evidence' in guided[0]['content']
    assert 'Extraction goal' not in plain[1]['content']


def test_large_document_propagates_goal_to_all_chunks_and_synthesis():
    async def run():
        with patch.object(web, '_call_summarizer_llm', new_callable=AsyncMock, return_value='Chunk evidence') as chunk, \
             patch.object(web, '_resolve_web_extract_auxiliary', return_value=(object(), 'test', {})), \
             patch.object(web, 'async_call_llm', new_callable=AsyncMock, return_value='Merged evidence') as llm, \
             patch.object(web, 'extract_content_or_reasoning', side_effect=lambda x:x):
            assert await web._process_large_content_chunked('abcdefgh', '', None, 4, 1000, goal='Find dates') == 'Merged evidence'
            assert len(chunk.call_args_list) == 2
            assert all(c.kwargs['goal'] == 'Find dates' for c in chunk.call_args_list)
            assert 'Find dates' in llm.call_args.kwargs['messages'][1]['content']
        with patch.object(web, '_process_large_content_chunked', new_callable=AsyncMock, return_value='Merged') as large:
            assert await web.process_content_with_llm('x'*500001, goal='Find dates') == 'Merged'
            assert large.call_args.kwargs['goal'] == 'Find dates'
    asyncio.run(run())


@pytest.mark.parametrize('mode,safety', [('legacy', False), ('legacy', True), ('prefix_overflow', False)])
def test_actual_extraction_dispatch_preserves_goal_through_all_summary_routes(mode, safety):
    async def run():
        body = 'SOURCE '*4000
        provider = SimpleNamespace(name='test', supports_extract=lambda:True,
            extract=AsyncMock(return_value=[{'url':'https://example.org', 'raw_content':body}]))
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {'GLASSEEK_QUEST_PAGE_RETRIEVAL':'0','GLASSEEK_FETCH_DOH_SAFETY':'0'}))
            stack.enter_context(patch.object(web, 'async_is_safe_url', new_callable=AsyncMock, return_value=True))
            stack.enter_context(patch.object(web, '_get_extract_backend', return_value='test'))
            stack.enter_context(patch.object(web, '_ensure_web_plugins_loaded'))
            stack.enter_context(patch('agent.web_search_registry.get_provider', return_value=provider))
            stack.enter_context(patch.object(web, '_get_default_summarizer_model', return_value='test'))
            stack.enter_context(patch.object(web, 'check_auxiliary_model', return_value=True))
            summary = stack.enter_context(patch.object(web, 'process_content_with_llm', new_callable=AsyncMock,
                                                       side_effect=[None, 'Evidence'] if safety else None, return_value='Evidence'))
            stack.enter_context(patch.object(web, '_debug'))
            result=json.loads(await web.web_extract_tool(['https://example.org'], content_mode=mode, goal='Find dates'))
            assert 'Evidence' in result['results'][0]['content']
            assert len(summary.call_args_list) == (2 if safety else 1)
            assert all(c.kwargs['goal']=='Find dates' for c in summary.call_args_list)
            if mode=='prefix_overflow':
                assert result['results'][0]['content'].startswith(body[:15000])
    asyncio.run(run())


def test_schema_registry_and_invalid_goal():
    schema=web.WEB_EXTRACT_SCHEMA
    assert schema['parameters']['properties']['goal']['type']=='string'
    assert schema['parameters']['required']==['urls']
    async def run():
        with patch.object(web, 'web_extract_tool', new_callable=AsyncMock, return_value='ok') as extract:
            handler=web.registry.get_entry('web_extract').handler
            assert await handler({'urls':['https://example.org'],'goal':'Find dates'})=='ok'
            assert extract.call_args.kwargs['goal']=='Find dates'
        assert 'goal must be a string' in await web.web_extract_tool([], goal=['bad'])
    asyncio.run(run())


def test_reference_schemas_keep_goal_and_goal_reads_do_not_use_old_snippets():
    schema=copy.deepcopy(web.WEB_EXTRACT_SCHEMA)
    for mode in ('ref','hybrid','evidence'):
        adapted=annotate_schemas([schema], mode)[0]
        assert adapted['parameters']['properties']['goal']==schema['parameters']['properties']['goal']
    registry=URLRegistry('evidence')
    registry.observe('web_extract', json.dumps({'results':[{'url':'https://example.org','content':'Old goal evidence'}]}))
    registry.begin_turn()
    args={'urls':['D1@v1'], 'goal':'A different goal'}
    assert registry.cached_evidence('web_extract', {'urls':['D1@v1']}) is not None
    assert registry.cached_evidence('web_extract', args) is None
    assert registry.resolve_args('web_extract', args)=={'urls':['https://example.org'],'goal':'A different goal'}


@pytest.mark.parametrize('mode', ['raw','ref','hybrid','evidence'])
def test_rl_goal_is_forwarded_in_every_url_mode(mode):
    actual=[]
    class FakeTools(WebTools):
        def _execute(self,call):
            actual.append(call)
            return 'Evidence'
    with patch.dict(os.environ, {'GLASSEEK_SEARCH_POOL_ENDPOINT':'http://127.0.0.1:1/search'}):
        tools=FakeTools(url_mode=mode)
    tools.url_registry.observe('web_search', json.dumps({'url':'https://example.org'}))
    tools.begin_turn()
    args={'url':'https://example.org'} if mode=='raw' else {'urls':['U1']}
    asyncio.run(tools.execute({'name':'web_extract','arguments':{**args,'goal':'Find dates'}}))
    assert actual[0]['arguments']=={'url':'https://example.org','goal':'Find dates'}


def test_rl_selects_before_truncating_and_does_not_silently_ignore_goal():
    env={'GLASSEEK_SEARCH_POOL_ENDPOINT':'http://pool/search', 'GLASSEEK_QUEST_RETRIEVAL_URL':'http://selector',
         'GLASSEEK_QUEST_RETRIEVAL_TIMEOUT':'90'}
    with patch.dict(os.environ,env):
        tools=WebTools(url_mode='raw', max_chars=20)
        opener=Mock()
        opener.open.return_value=io.BytesIO(json.dumps({'content':'Relevant evidence'}).encode())
        with patch('urllib.request.urlopen', return_value=io.BytesIO(b'x'*10000+b'TARGET EVIDENCE')), \
             patch('urllib.request.build_opener',return_value=opener):
            result=tools._execute({'name':'web_extract','arguments':{'url':'https://example.org','goal':'Find target'}})
        assert result=='Relevant evidence'
        payload=json.loads(opener.open.call_args.args[0].data)
        assert payload['goal']=='Find target'
        assert payload['text'].endswith('TARGET EVIDENCE')
        assert search_pool_worker_env()['GLASSEEK_QUEST_RETRIEVAL_URL']=='http://selector'
        with patch.dict(os.environ,{'GLASSEEK_QUEST_RETRIEVAL_URL':''}), patch('urllib.request.urlopen') as fetch:
            assert 'requires GLASSEEK_QUEST_RETRIEVAL_URL' in tools._execute({'name':'web_extract','arguments':{'url':'https://example.org','goal':'Find'}})
            fetch.assert_not_called()
