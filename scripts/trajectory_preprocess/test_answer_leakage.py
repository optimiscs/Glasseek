import json
from pathlib import Path
import unittest
from safety_rules import Scanner
from glasseek_contract import build_contract
from tool_xml import serialize
from answer_leakage import external_signals

QUESTION='Which scientist discovered the unusual blue mineral near the abandoned northern mountain observatory during the winter expedition of 1923?'


class AnswerLeakageTests(unittest.TestCase):
    def setUp(self):
        self.contract=build_contract();self.policy=json.loads(Path(__file__).with_name('safety_policy.json').read_text())
        self.scanner=Scanner(self.policy,self.contract,b'synthetic-answer-leakage-test-key-0000')

    def item(self,text,query='mineral discovery history'):
        return {'messages':[{'role':'system','content':'Research task. Benchmark: BrowseComp. Do not read answer keys.'},
                {'role':'user','content':QUESTION},{'role':'assistant','content':serialize({'name':'web_search','arguments':{'query':query}})},
                {'role':'user','content':'<tool_response>'+text+'</tool_response>'},{'role':'assistant','content':'Answer'}],
                'tools':[{'type':'function','function':s} for s in self.contract['schemas'].values()]}

    def test_new_default_ignores_email_phone_and_task_benchmark(self):
        result=self.scanner.scan(self.item('contact@example.org 13800138000'))
        self.assertEqual(result['state'],'pass_rules')
        self.assertFalse(any(f['rule_id'].startswith(('privacy.email','privacy.phone','benchmark.')) for f in result['findings']))

    def test_current_question_plus_reference_answer_exposure(self):
        text=json.dumps({'results':[{'content':QUESTION+'\nReference answer: Synthetic scientist'}]})
        result=self.scanner.scan(self.item(text))
        found=[f for f in result['findings'] if f['rule_id']=='benchmark.current_question_answer']
        self.assertTrue(found);self.assertTrue(all(f['origin']=='external' for f in found))
        self.assertEqual(result['state'],'review') # A signal is not proof of assistant use.

    def test_query_echo_and_hidden_reference_do_not_prove_leak(self):
        text=json.dumps({'query':QUESTION,'metadata':{'reference_answer':'Synthetic scientist'},'results':[{'content':'Independent historical account.'}]})
        item=self.item(text);item['reference_answer']='Synthetic scientist'
        self.assertFalse(any(f['rule_id'].startswith('benchmark.') for f in self.scanner.scan(item)['findings']))

    def test_regular_huggingface_is_not_blocked_as_answer_bank(self):
        result=self.scanner.scan(self.item('Model documentation at https://huggingface.co/docs/transformers'))
        self.assertFalse(any(f['rule_id'].startswith('benchmark.') for f in result['findings']))

    def test_actual_benchmark_query_is_request_signal_only(self):
        result=self.scanner.scan(self.item('No results',query='BrowseComp answer key'))
        self.assertTrue(any(f['rule_id']=='benchmark.resource_requested' for f in result['findings']))
        self.assertFalse(any(f['rule_id']=='benchmark.current_question_answer' for f in result['findings']))

    def test_key_masking_and_url_slug_requires_review(self):
        key='sk-'+'z'*32
        self.assertEqual(self.scanner.scan(self.item(key))['state'],'quarantine')
        self.assertNotIn(key,self.scanner.scrub(key))
        slug='https://example.org/sk-ordinary-lowercase-news-article'
        result=self.scanner.scan(self.item(slug))
        self.assertEqual(result['state'],'review')
        self.assertFalse(any(f['disposition']=='quarantine' for f in result['findings']))

    def test_chinese_long_question_and_coverage_limit(self):
        question='请根据这次考察记录指出在一九二三年冬季前往北方山区废弃天文台附近首次发现蓝色矿物的科学家姓名。'
        rules=dict(external_signals(question+'\n标准答案：示例人名',question))
        self.assertIn('benchmark.current_question_answer',rules)
        self.assertIn('coverage.answer_overlap_limit',dict(external_signals('x'*100,question,10)))


if __name__=='__main__':unittest.main()
