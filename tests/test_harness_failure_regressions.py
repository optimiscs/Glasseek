import json
import os
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Evaluation_backend'))
from plugins.browse_comp_guard import _filter_web_result,_on_pre_tool_call,_contains_question_leak
from tools.quest_page_retrieval import page_failure


def test_chinese_question_keeps_real_sources_but_blocks_its_solution():
    q='prometheus 的高流失率会造成什么影响，有什么系统的方案可以解决？各家云厂商有没有现有方案？'
    with patch.dict(os.environ,{'BROWSE_COMP_RUN_ID':'test','BROWSE_COMP_QUESTION':q,'BROWSE_COMP_QUESTION_MATCH_MODE':'evaluation'}):
        doc={'title':'Prometheus documentation','content':'Prometheus is an open-source monitoring system.'}
        assert _filter_web_result(json.dumps({'results':[doc]})) is None
        leak={'content':q+' 标准答案：提供测试参考解答。'}
        assert json.loads(_filter_web_result(json.dumps({'results':[leak]})))['success'] is False
        assert _on_pre_tool_call('web_extract',{'urls':['https://huggingface.co/datasets/gaia-benchmark/GAIA']})['action']=='block'
        assert _on_pre_tool_call('execute_code',{'code':"open('/etc/passwd').read()"})['action']=='block'


def test_agent_paper_short_matching_excerpt_is_not_independent_evidence():
    q='The Latin root of the Yola word "gimlie" shares a spelling with a Spanish word. What is the Google translation of the source title for the 1994 example sentence for that word in the Collins Spanish-to-English dictionary online?'
    assert _contains_question_leak({'title':'OWL: Optimized Workforce Learning for General Multi-Agent Assistance','description':'What is the Google translation of the source title for the 1994 example ...'},q)
    assert not _contains_question_leak({'title':'OWL: Optimized Workforce Learning','description':'This paper studies agent architecture and memory.'},q)


def test_short_named_topics_are_not_leaks_without_answer_context():
    assert not _contains_question_leak({'content':'Prometheus monitoring documentation'},'Prometheus')
    assert _contains_question_leak({'content':'Question: Explain entropy. Reference answer: ...'},'Explain entropy')


def test_error_pages_not_articles_about_error_pages():
    assert page_failure('Title: Verifying your browser | OpenReview\nPlease complete the verification above.')=='access_challenge'
    assert page_failure('Title: CAPTCHA design\nThis article explains CAPTCHA and access control.') is None


def test_verdict_parser_accepts_formatting_but_rejects_conflict():
    from scripts.evaluation_suite.score import parse_binary_verdict
    assert parse_binary_verdict('Explanation: clear\n**Correct:** YES')=='yes'
    assert parse_binary_verdict('correct:no\ncorrect: NO')=='no'
    for text in ['correct: yes\ncorrect: no','The answer is probably correct: yes','correct: maybe','correct: yes/no','correct: yes or no']:
        with pytest.raises(ValueError):parse_binary_verdict(text)


def test_real_run_spec_converts_through_strict_wire_before_host_metadata(tmp_path,monkeypatch):
    from browse_comp_eval import hermes_client as hc
    from browse_comp_eval.dataset import DatasetItem,RunSpec
    from browse_comp_eval.sandbox_protocol import SandboxRunResult
    from types import SimpleNamespace
    config=hc.HermesConfig(hermes_bin='unused',api_key='offline',base_url='http://localhost/v1',model='test',provider='custom',save_name='test',dataset='suite_hle',max_rounds=2,timeout_seconds=None,toolsets=[],skills=[])
    client=hc.HermesClient(config)
    for key in ['GLASSEEK_CONTEXT_MANAGER','GLASSEEK_MAX_TOOL_CALLS','GLASSEEK_RESUME_INTERRUPTED','GLASSEEK_RESEARCH_PROTOCOL']:monkeypatch.delenv(key,raising=False)
    monkeypatch.setenv('HERMES_HOME',str(tmp_path/'hermes'))
    monkeypatch.setenv('GLASSEEK_HARNESS_REVISION','tested')
    monkeypatch.setattr(client,'_setup_profile',lambda *a,**kw:None)
    monkeypatch.setattr(client,'_build_env',lambda *a,**kw:{})
    monkeypatch.setattr(client,'_build_chat_command',lambda *a:['unused'])
    monkeypatch.setattr(client,'_browser_cdp_url_for_spec',lambda *a,**kw:'')
    monkeypatch.setattr(hc.subprocess,'Popen',lambda *a,**kw:SimpleNamespace(returncode=0,pid=123,communicate=lambda **kw:('','')))
    monkeypatch.setattr(hc,'extract_session_id',lambda *a:'test-session')
    session={'session_id':'test-session','messages':[{'role':'user','content':'q'},
        {'role':'tool','content':'{"status":"tool_budget_denied","error":"limit"}'},
        {'role':'assistant','content':'unknown','finish_reason':'stop'}]}
    monkeypatch.setattr(hc,'load_session',lambda *a,**kw:session)
    spec=RunSpec('q','q__r1',1,DatasetItem('q','q','PRIVATE','research',1))
    wire=client.run_request(client.build_request(spec,1),tmp_path/'wire')
    assert isinstance(wire,SandboxRunResult)
    assert 'termination_reason' not in wire.to_dict() and 'answer' not in wire.to_dict()
    record=client.run(spec,tmp_path/'host',1)
    assert record['termination_reason']=='tool_budget_exhausted'
    assert record['harness_revision']=='tested' and record['answer']=='PRIVATE'
