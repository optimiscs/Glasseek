import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Evaluation_backend'))
from tools.quest_page_retrieval import select_pages

class QuestPageRetrievalTest(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_goal_overrides_global_question(self):
        seen=[]
        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={'content':'Evidence', 'audit':{}})
        real=httpx.AsyncClient
        with patch.dict(os.environ, {'BROWSE_COMP_QUESTION':'Global task', 'GLASSEEK_QUEST_RETRIEVAL_URL':'http://selector'}), \
             patch('tools.quest_page_retrieval.httpx.AsyncClient', side_effect=lambda **kw:real(transport=httpx.MockTransport(handler), **kw)):
            result=await select_pages([{'url':'https://example.org', 'content':'Source'}], goal='Specific child task')
        self.assertEqual(seen[0]['goal'],'Specific child task')
        self.assertEqual(result[0]['retrieval']['goal_source'],'tool_goal')

    async def test_selected_evidence_replaces_raw(self):
        seen = []
        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={'content': 'relevant evidence', 'audit': {'reason': 'selected', 'selected_indices': [2]}})
        real = httpx.AsyncClient
        with patch.dict(os.environ, {'BROWSE_COMP_QUESTION': 'Which year?', 'GLASSEEK_QUEST_RETRIEVAL_URL': 'http://127.0.0.1:18217'}), \
             patch('tools.quest_page_retrieval.httpx.AsyncClient', side_effect=lambda **kw: real(transport=httpx.MockTransport(handler), **kw)):
            result = await select_pages([{'url': 'https://example.com', 'raw_content': 'long source text', 'content': 'old'}])
        self.assertEqual(seen[0]['goal'], 'Which year?')
        self.assertEqual(result[0]['content'], 'relevant evidence')
        self.assertNotIn('raw_content', result[0])
        self.assertEqual(result[0]['retrieval']['selected_indices'], [2])
        with patch.dict(os.environ, {'BROWSE_COMP_RUN_ID': 'test', 'BROWSE_COMP_QUESTION': 'Which year?', 'BROWSE_COMP_QUESTION_MATCH_MODE': 'evaluation'}):
            from plugins.browse_comp_guard import _filter_web_result
            self.assertIsNone(_filter_web_result(json.dumps({'results': result})))

    async def test_failed_retrieval_never_silently_returns_page(self):
        def handler(request): return httpx.Response(503)
        real = httpx.AsyncClient
        with patch.dict(os.environ, {'BROWSE_COMP_QUESTION': 'Which year?', 'GLASSEEK_QUEST_RETRIEVAL_URL': 'http://127.0.0.1:18217'}), \
             patch('tools.quest_page_retrieval.httpx.AsyncClient', side_effect=lambda **kw: real(transport=httpx.MockTransport(handler), **kw)):
            result = await select_pages([{'url': 'https://example.com', 'raw_content': 'source', 'content': 'source'}])
        self.assertEqual(result[0]['content'], '')
        self.assertIn('unavailable', result[0]['error'])
        self.assertNotIn('raw_content', result[0])

if __name__ == '__main__': unittest.main()
