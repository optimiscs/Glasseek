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
    for text in ['correct: yes\ncorrect: no','The answer is probably correct: yes','correct: maybe']:
        with pytest.raises(ValueError):parse_binary_verdict(text)
