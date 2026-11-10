import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from glasseek_shared.research import protocol, prompts, runtime
from glasseek_shared.hermes.iteration_budget import IterationBudget


def call(name='finish', payload=None, content=None, extra=False):
    calls = [NS(id='t1', function=NS(name=name, arguments=json.dumps(payload or {'answer': '42', 'evidences': [], 'confidence': 99})))]
    if extra:
        calls.append(NS(id='t2', function=NS(name='web_search', arguments='{}')))
    return NS(content=content, tool_calls=calls)


@pytest.fixture
def research(monkeypatch, tmp_path):
    monkeypatch.setenv('GLASSEEK_RESEARCH_PROTOCOL', 'arex-native')
    monkeypatch.setenv('GLASSEEK_RESEARCH_DATASET', 'suite_hle')
    monkeypatch.setenv('GLASSEEK_RESEARCH_RESULT', str(tmp_path/'result.json'))
    return NS(max_iterations=12, iteration_budget=IterationBudget(12))


def test_native_constraints_do_not_mutate_or_expand_tools(monkeypatch):
    data = {'tools': [{'type':'function','function':{'name':'web_search'}},
                      {'type':'function','function':{'name':'terminal'}}], 'tool_choice':'auto'}
    monkeypatch.delenv('GLASSEEK_RESEARCH_PROTOCOL', raising=False)
    assert protocol.native_kwargs(data) is data
    monkeypatch.setenv('GLASSEEK_RESEARCH_PROTOCOL', 'arex-native')
    actual = protocol.native_kwargs(data)
    assert [t['function']['name'] for t in actual['tools']] == ['web_search', 'finish']
    assert actual['tool_choice'] == 'required' and actual['parallel_tool_calls'] is False
    assert len(data['tools']) == 2 and data['tool_choice'] == 'auto'


@pytest.mark.parametrize('bad', [True, -1, 101, '95%', float('nan')])
def test_rejects_invalid_confidence(bad):
    with pytest.raises(ValueError):
        protocol.validate_finish(json.dumps({'answer':'a','evidences':[],'confidence':bad}), 'hle', [])


def test_evidence_requires_actual_observed_url_and_strict_fields():
    value = {'answer': 'a', 'evidences': [{'evidence':'fact','url':'https://example.org/page'}], 'confidence':95}
    with pytest.raises(ValueError, match='not present'):
        protocol.validate_finish(json.dumps(value), 'browsecomp', [{'role':'user', 'content':'https://example.org/page'}])
    assert protocol.validate_finish(json.dumps(value), 'browsecomp', [{'role':'tool','content':json.dumps({'url':'https://example.org/page'})}]) == value
    with pytest.raises(ValueError, match='duplicate'):
        protocol.validate_finish('{"answer":"a","answer":"b","evidences":[],"confidence":95}', 'hle', [])
    with pytest.raises(ValueError, match='source-backed'):
        protocol.validate_finish('{"answer":"a","evidences":[],"confidence":95}', 'drbench', [])


def test_mixed_finish_has_no_side_effect_and_matching_tool_errors(research):
    messages=[]
    action, answer, extra = runtime.handle_response(research, call(extra=True), messages, [], 1)
    assert action == 'continue' and answer is None and extra == 0
    assert [m['tool_call_id'] for m in messages if m['role']=='tool'] == ['t1','t2']
    assert not runtime.state(research)['candidates']


def test_text_submission_never_accepted_and_corrections_bounded(research):
    for i in range(3):
        action, answer, _ = runtime.handle_response(research, NS(content='Answer: 42',tool_calls=[]), [], [], i+1)
    assert action == 'failed' and answer is None
    assert runtime.state(research)['status'] == 'protocol_failed'


def test_finish_render_and_reports_preserved(research):
    messages=[]
    action, answer, _ = runtime.handle_response(research, call(), messages, [], 1)
    assert action == 'finished' and 'Answer: 42' in answer
    assert messages[-2]['role'] == 'tool' and messages[-1]['content'] == answer
    report = {'answer':'# Full report\n\n| A | B |\n|---|---|\n| x | y |\n', 'confidence':95, 'evidences':[]}
    for dataset in ['drbench', 'liveresearchbench', 'widesearch', 'gaia']:
        assert protocol.render_finish(report, dataset) == report['answer']


def test_low_confidence_review_does_not_read_gold_and_shares_budget(research):
    research._build_api_kwargs = lambda messages: {'messages':messages}
    review = call('review_candidate', {'accept':False,'issues':['missing support'],'next_steps':['read original source']})
    research._interruptible_api_call = Mock(return_value=NS(choices=[NS(message=review)], usage=None))
    candidate=call(payload={'answer':'42','evidences':[],'confidence':70})
    original=[{'role':'system','content':'old system'},{'role':'user','content':'the problem'}]
    action, _, extra = runtime.handle_response(research,candidate,[],original,1)
    assert action == 'continue' and extra == 1 and research.iteration_budget.used == 1
    sent=research._interruptible_api_call.call_args.args[0]
    assert sent['temperature'] == 0 and sent['tool_choice']['function']['name'] == 'review_candidate'
    assert 'old system' not in json.dumps(sent) and 'the problem' in json.dumps(sent)
    action, _, extra = runtime.handle_response(research,candidate,[],original,3)
    assert action == 'failed' and extra == 1
    assert len(runtime.state(research)['reviews']) == 2


def test_review_acceptance_conflicting_with_issues_fails(research):
    research._build_api_kwargs=lambda messages: {'messages':messages}
    review=call('review_candidate', {'accept':True,'issues':['unsupported answer'],'next_steps':[]})
    research._interruptible_api_call=Mock(return_value=NS(choices=[NS(message=review)],usage=None))
    event=runtime._review(research,[],{'answer':'a'},'verify_candidate')
    assert not event['feedback']['accept'] and 'conflicts' in event['error']


def test_prompt_pairs_no_second_protocol_or_short_report():
    for name in prompts.POLICIES:
        text=prompts.system_prompt(name)+prompts.user_prompt(name,'unique question')
        assert text.count('unique question') == 1
        for forbidden in ['<think>', '<tool_call>', 'google_scholar', 'update_context']:
            assert forbidden not in text
    assert 'FULL report' in prompts.system_prompt('drbench')
    assert 'shortest' in prompts.system_prompt('gaia')
    with pytest.raises(ValueError):prompts.system_prompt('browsecomp_plus')


def test_source_identity_survives_program_compression(research, monkeypatch):
    monkeypatch.setenv('GLASSEEK_RESEARCH_DATASET','suite_browsecomp')
    runtime.observe(research,[{'role':'tool','content':'https://example.org/read'}])
    candidate=call(payload={'answer':'entity','evidences':[{'evidence':'fact','url':'https://example.org/read'}],'confidence':99})
    action,_,_=runtime.handle_response(research,candidate,[],[],2)
    assert action=='finished'


def test_native_extract_description_matches_actual_no_summary_path(research):
    kwargs=protocol.native_kwargs({'tools':[{'type':'function','function':{'name':'web_extract',
        'description':'LLM summarized', 'parameters':{'properties':{'goal':{'type':'string','description':'summarize'}}}}}]})
    function=kwargs['tools'][0]['function']
    assert 'original passages' in function['description']
    assert 'summarize' not in function['parameters']['properties']['goal']['description']


@pytest.mark.parametrize('url', ['https://en.wikipedia.org/wiki/Function_(mathematics)',
    'https://example.org/a,b?signature=abc%2Bdef&key=1', 'https://example.org/path,'])
def test_original_url_identity_is_preserved_in_finish(url):
    observed=[{'role':'tool','content':json.dumps({'results':[{'url':url,'content':'source fact'}]})}]
    candidate={'answer':'a','confidence':95,'evidences':[{'evidence':'source fact','url':url}]}
    assert protocol.validate_finish(json.dumps(candidate),'browsecomp',observed)==candidate
    # A normalization must not silently bless a different URL.
    candidate['evidences'][0]['url']=url[:-1] if url.endswith((',', ')')) else url+'changed'
    with pytest.raises(ValueError,match='not present'):
        protocol.validate_finish(json.dumps(candidate),'browsecomp',observed)
