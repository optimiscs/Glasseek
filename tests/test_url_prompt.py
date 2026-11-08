import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'Shared'), str(ROOT / 'RL_training')]
from glasseek_shared.hermes.url_prompt import (
    DEFAULT_CONFIG, MARKER, append_system_prompt, apply_to_messages, prompt_module,
)
from glasseek_shared.hermes.url_registry import URLRegistry
from glasseek_training.web_tools import parse_tool_calls, search_pool_worker_env


def test_raw_and_disabled_prompt_leave_input_unchanged():
    messages = [{'role': 'system', 'content': '  Original\n'}, {'role': 'user', 'content': 'Question'}]
    assert apply_to_messages(messages, 'raw', '/missing/config') is messages
    assert apply_to_messages(messages, 'ref', 'off') is messages
    assert append_system_prompt('  Original\n', 'raw') == '  Original\n'


@pytest.mark.parametrize('mode', ['hybrid', 'ref', 'evidence'])
def test_module_is_static_idempotent_and_does_not_publish_future_ids(mode):
    messages = [{'role': 'system', 'content': 'Research instructions'}, {'role': 'user', 'content': 'Question'}]
    original = copy.deepcopy(messages)
    result = apply_to_messages(messages, mode, 'auto')
    assert messages == original
    assert result[0]['content'].startswith('Research instructions\n\n')
    assert result[0]['content'].count(MARKER) == 1
    assert apply_to_messages(result, mode, 'auto') == result
    registry = URLRegistry(mode)
    with pytest.raises(ValueError, match='not-yet-visible'):
        registry.resolve('U1')
    registry.observe('web_search', json.dumps({'url': 'https://example.org/real?signature=ABC'}))
    assert prompt_module(mode, 'auto')['text'] == prompt_module(mode, 'auto')['text']
    assert 'signature=ABC' not in result[0]['content']
    with pytest.raises(ValueError):
        apply_to_messages(result, mode, 'off')


def test_custom_config_is_replaceable_and_versioned(tmp_path):
    config = json.loads(DEFAULT_CONFIG.read_text())
    config['version'] = 'experiment-v2'
    config['common'] += '\nAdditional general instruction.'
    path = tmp_path / 'custom.json'
    path.write_text(json.dumps(config))
    custom = prompt_module('ref', str(path))
    assert custom['sha256'] != prompt_module('ref', 'auto')['sha256']
    assert custom['version'] == 'experiment-v2'
    result = apply_to_messages([{'role': 'user', 'content': 'Q'}], 'ref', str(path))
    assert result[0]['role'] == 'system'
    assert result[1] == {'role': 'user', 'content': 'Q'}
    with pytest.raises(ValueError, match='differs'):
        apply_to_messages(result, 'ref', 'auto')


def test_few_shots_parse_as_xml_and_resolve_real_registered_sources():
    calls = parse_tool_calls(prompt_module('evidence', 'auto')['text'])
    assert calls[0] == {'name': 'web_extract', 'arguments': {'urls': ['U1']}}
    for call in calls:
        assert '<url>' not in str(call['arguments'])
    registry = URLRegistry('evidence')
    for i in range(1, 7):
        registry.observe('web_search', json.dumps({'url': f'https://example.org/report/{i}?signed=A%2FB'}))
    registry.begin_turn()
    assert registry.resolve_args('web_extract', calls[0]['arguments'])['urls'] == ['https://example.org/report/1?signed=A%2FB']
    registry.observe('web_extract', json.dumps({'url': 'https://example.org/report/6?signed=A%2FB', 'content': 'x' * 6400}))
    registry.begin_turn()
    evidence_call = calls[-1]
    cached = json.loads(registry.cached_evidence(evidence_call['name'], evidence_call['arguments']))
    assert len(cached['results']) == 2
    assert '<url>D6@v1#E3</url>' in cached['results'][0]['content']


def test_tool_descriptions_use_the_same_pluggable_config(tmp_path, monkeypatch):
    from glasseek_shared.hermes.url_registry import annotate_schemas
    schemas = [{'function': {'name': 'web_extract', 'description': 'Original tool description'}}]
    monkeypatch.setenv('GLASSEEK_URL_PROMPT', 'auto')
    result = annotate_schemas(schemas, 'ref')
    assert 'urls=["U1"]' in result[0]['function']['description']
    assert 'without XML tags' in result[0]['function']['description']
    assert schemas[0]['function']['description'] == 'Original tool description'
    config = json.loads(DEFAULT_CONFIG.read_text())
    config['tool_descriptions']['web_extract'] = 'Custom reference instruction.'
    path = tmp_path / 'prompt.json'; path.write_text(json.dumps(config))
    monkeypatch.setenv('GLASSEEK_URL_PROMPT', str(path))
    assert 'Custom reference instruction.' in annotate_schemas(schemas, 'ref')[0]['function']['description']
    assert 'Custom reference instruction.' in prompt_module('ref')['text']
    monkeypatch.setenv('GLASSEEK_URL_PROMPT', 'off')
    assert annotate_schemas(schemas, 'ref')[0]['function']['description'] == 'Original tool description'
    assert annotate_schemas(schemas, 'raw') is schemas


def test_sft_and_rollout_share_prompt_and_worker_configuration(monkeypatch):
    from glasseek_training.prepare_data import convert_record
    monkeypatch.setenv('GLASSEEK_URL_PROMPT', 'auto')
    record = {'messages': [{'role': 'user', 'content': 'Question'}, {'role': 'assistant', 'content': 'Answer'}]}
    output, _ = convert_record(record, 'sft', 'ref')
    messages = json.loads(output['messages'])
    assert messages[0]['content'] == prompt_module('ref')['text']
    assert messages[-1] == record['messages'][-1]
    assert json.loads(output['url_prompt'])['sha256'] == prompt_module('ref')['sha256']
    monkeypatch.setenv('GLASSEEK_URL_MODE', 'ref')
    monkeypatch.setenv('GLASSEEK_URL_PROMPT', 'off')
    monkeypatch.setenv('GLASSEEK_SEARCH_POOL_ENDPOINT', 'http://example.invalid/search')
    assert search_pool_worker_env()['GLASSEEK_URL_PROMPT'] == 'off'
