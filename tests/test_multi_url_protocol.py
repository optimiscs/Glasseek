"""Actual tool entrypoints with mocked network/model backends only."""
import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/p) for p in ('Evaluation_backend','RL_training','Shared')]
from tools import web_tools as web
from glasseek_training.web_tools import WebTools, parse_tool_calls


def test_evaluation_registry_rejects_six_urls_before_network():
    async def run():
        with patch.object(web,'async_is_safe_url',new_callable=AsyncMock) as network:
            result=await web.registry.get_entry('web_extract').handler({'urls':['https://example.org']*6})
            assert '1 to 5' in json.loads(result)['error']
            network.assert_not_called()
    asyncio.run(run())


def test_evaluation_mixed_results_preserve_url_failure_and_order():
    async def run():
        urls=['https://example.org/a','https://example.org/b']
        provider=SimpleNamespace(name='test',supports_extract=lambda:True,extract=AsyncMock(return_value=[
            {'url':urls[1],'content':'','error':'page failed'},
            {'url':urls[0],'content':'Page A','error':None}]))
        with ExitStack() as s:
            s.enter_context(patch.dict(os.environ,{'GLASSEEK_QUEST_PAGE_RETRIEVAL':'0','GLASSEEK_FETCH_DOH_SAFETY':'0'}))
            s.enter_context(patch.object(web,'async_is_safe_url',new_callable=AsyncMock,return_value=True))
            s.enter_context(patch.object(web,'_get_extract_backend',return_value='test'))
            s.enter_context(patch.object(web,'_ensure_web_plugins_loaded'))
            s.enter_context(patch('agent.web_search_registry.get_provider',return_value=provider))
            s.enter_context(patch.object(web,'_debug'))
            llm=s.enter_context(patch.object(web,'process_content_with_llm',new_callable=AsyncMock))
            result=json.loads(await web.web_extract_tool(urls,use_llm_processing=False))
            assert [r['url'] for r in result['results']]==urls
            assert result['results'][0]['content']=='Page A'
            assert result['results'][0]['error'] is None
            assert result['results'][1]['error']=='page failed'
            llm.assert_not_called()
    asyncio.run(run())


@pytest.mark.parametrize('mode',['raw','ref','hybrid','evidence'])
def test_training_entrypoint_multi_url_and_per_page_errors(mode):
    calls=[]
    class Local(WebTools):
        def _execute(self,call):
            calls.append(call)
            return 'Tool error: unavailable' if call['arguments']['url'].endswith('/b') else 'Page A'
    with patch.dict(os.environ,{'GLASSEEK_SEARCH_POOL_ENDPOINT':'http://pool/search'}):
        env=Local(url_mode=mode)
    urls=['https://example.org/a','https://example.org/b']
    result=json.loads(asyncio.run(env.execute({'name':'web_extract','arguments':{'urls':urls,'goal':'Find dates'}})))
    assert len(result['results'])==2
    assert result['results'][0]['content'].endswith('Page A')
    assert result['results'][1]['content']==''
    assert result['results'][1]['error']=='Tool error: unavailable'
    assert all(c['arguments']['goal']=='Find dates' for c in calls)
    calls.clear()
    result=json.loads(asyncio.run(env.execute({'name':'web_extract','arguments':{'urls':urls*3}})))
    assert 'error' in result and calls==[]


def test_xml_url_array_stays_typed():
    text='<tool_call><function=web_extract><parameter=urls>["https://example.org/a", "https://example.org/b"]</parameter><parameter=goal>Find dates</parameter></function></tool_call>'
    call=parse_tool_calls(text)[0]
    assert call['arguments']['urls']==['https://example.org/a','https://example.org/b']
    assert call['arguments']['goal']=='Find dates'
