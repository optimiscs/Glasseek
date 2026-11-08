import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('role', ['Evaluation_backend', 'Data_curation/multi_agent_mpcs_v5_publish/hermes'])
def test_role_imports_share_implementation_and_behavior(role):
    code = '''
import concurrent.futures, json
from pathlib import Path
from agent.message_content import flatten_message_text
from agent.gemini_schema import sanitize_gemini_schema
from agent.iteration_budget import IterationBudget
from agent.turn_retry_state import TurnRetryState
from agent.retry_utils import jittered_backoff
from agent.tool_result_classification import file_mutation_result_landed
from agent.web_search_provider import WebSearchProvider
from plugins.web.search_server.provider import SearchServerProvider
from plugins.web.fetch_server.provider import FetchServerProvider
import agent.message_content as implementation
budget = IterationBudget(7)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    consumed = sum(pool.map(lambda _: budget.consume(), range(100)))
assert consumed == 7 and budget.remaining == 0
budget.refund(); assert budget.consume() and not budget.consume()
assert flatten_message_text([{'type':'image_url','text':'hidden'}, {'text':'visible'}, 'tail']) == 'visible\\ntail'
schema = {'type':'object', 'additionalProperties':False, 'properties':{'x':{'type':'integer', 'enum':[1,2]}}}
assert sanitize_gemini_schema(schema) == {'type':'object','properties':{'x':{'type':'integer'}}}
assert 'additionalProperties' in schema
assert not any(dict(TurnRetryState()).values())
assert jittered_backoff(2, base_delay=3, jitter_ratio=0) == 6
assert file_mutation_result_landed('patch', '{"success":true}')
assert not file_mutation_result_landed('patch', '{"success":true,"error":"failed"}')
assert issubclass(SearchServerProvider, WebSearchProvider)
assert issubclass(FetchServerProvider, WebSearchProvider)
print(json.dumps({'implementation':str(Path(implementation.__file__).resolve())}))
'''
    env = dict(os.environ, PYTHONPATH=str(ROOT / role))
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT / role, env=env,
                            text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout.splitlines()[-1])
    assert payload['implementation'] == str(ROOT / 'Shared/glasseek_shared/hermes/message_content.py')
