"""Optional Quest BM25 + embedding selection for external webpage content."""
import logging
import os
import time
import re

import httpx

logger = logging.getLogger(__name__)

def page_failure(body):
    """Recognize access/error pages, not articles merely discussing CAPTCHA."""
    head = body[:1800].lower()
    patterns = (r'title:\s*(?:verifying your browser|just a moment|access denied)',
                r'warning:.*(?:requiring captcha|401 unauthorized|403 forbidden)',
                r'please complete the verification above',
                r'^(?:\s*#+\s*)?(?:sign in|log in) to (?:continue|view|access)',
                r'^(?:\s*#+\s*)?(?:403 forbidden|401 unauthorized|access denied)\s*$')
    return 'access_challenge' if any(re.search(p,head,re.M) for p in patterns) else None

async def select_pages(results, goal=""):
    if not isinstance(goal, str):
        raise ValueError('goal must be a string')
    goal_source = 'tool_goal' if goal.strip() else 'user_question'
    goal = goal if goal.strip() else os.environ.get('BROWSE_COMP_QUESTION', '').strip()
    endpoint = os.environ.get('GLASSEEK_QUEST_RETRIEVAL_URL', '').rstrip('/')
    if not goal or not endpoint:
        raise ValueError('Quest retrieval requires a question and service URL')
    timeout = float(os.environ.get('GLASSEEK_QUEST_RETRIEVAL_TIMEOUT', '180'))
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        # Preserve source order; the bridge enforces global CPU/model concurrency.
        for result in results:
            if not isinstance(result, dict) or result.get('error'):
                continue
            body = result.get('raw_content') or result.get('content') or ''
            failure = page_failure(body) if body.strip() else 'empty_content'
            if failure:
                result.update(content='', success=False, status='unavailable',
                              error='No source text obtained: ' + failure, error_type=failure)
                result.pop('raw_content', None)
                continue
            started = time.monotonic()
            try:
                response = await client.post(endpoint + '/select', json={'text': body, 'goal': goal})
                response.raise_for_status()
                payload = response.json()
                content = payload['content']
                if not isinstance(content, str) or not content.strip():
                    raise ValueError('retrieval returned empty content')
                audit = payload.get('audit', {})
                logger.info('quest page retrieval url=%s seconds=%.2f input_tokens=%s output_tokens=%s chunks=%s selected=%s reason=%s',
                            result.get('url', ''), time.monotonic() - started,
                            audit.get('input_tokens'), audit.get('output_tokens'), audit.get('chunks'),
                            audit.get('selected_indices'), audit.get('reason'))
                result['content'] = content
                result.pop('raw_content', None)
                result['retrieval'] = {'method': 'bm25_embedding', 'reason': audit.get('reason'),
                                       'input_tokens': audit.get('input_tokens'),
                                       'output_tokens': audit.get('output_tokens'),
                                       'selected_indices': audit.get('selected_indices'),
                                       'stage_seconds': audit.get('stage_seconds'),
                                       'total_seconds': audit.get('total_seconds'),
                                       'goal_source': goal_source}
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                logger.warning('quest page retrieval failed url=%s error=%s', result.get('url', ''), type(exc).__name__)
                result['content'] = ''
                result.pop('raw_content', None)
                result['error'] = 'Quest BM25 + embedding retrieval unavailable; page content not delivered'
    return results
