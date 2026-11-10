"""Protocol regressions use synthetic requests, not dataset entities or answers."""
import copy
import itertools
import json
from pathlib import Path
import unittest

from ingest import normalize,harness_plan
from glasseek_contract import build_contract,materialize
from call_protocol import model_visible_chars
from migrate_searchswarm import review_messages
from glasseek_shared.extract_protocol import historical_results
from glasseek_shared.trajectory_release import validate_messages
from safety_rules import Scanner
from tool_xml import serialize


class HistoricalAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract=build_contract()
        cls.config={'search_name':'web_search','extract_name':'web_extract','extract_max_urls':5,'contract':cls.contract}
        cls.policy=json.loads(Path(__file__).with_name('safety_policy.json').read_text())
        cls.scanner=Scanner(cls.policy,cls.contract,b'synthetic-protocol-key-not-real-00000')

    def item(self,calls,responses):
        return normalize({'messages':[{'role':'user','content':'Which explorer found the ancient instrument near the abandoned southern observatory during the summer expedition of 1850?'},
            {'role':'assistant','tool_calls':calls},*responses,{'role':'assistant','content':'Answer'}]},'synthetic','fixture',0)

    def call(self,query,cid=None):
        return {'id':cid,'function':{'name':'search','arguments':{'query':query}}}

    def materialize(self,item):
        item['harness']=harness_plan(item,self.config)
        self.assertEqual(item['program_errors'],[]);self.assertEqual(item['harness']['errors'],[])
        materialize(item)
        return review_messages(item)

    def test_native_multi_query_batches_preserve_every_result_order(self):
        for order in itertools.permutations(['alpha','beta','gamma']):
            payload=[{'query':q,'result':[{'content':q.upper()}]} for q in order]
            response='<tool_response>\nTool: search\nResult: '+json.dumps(payload)+'</tool_response>'
            item=self.item([self.call(['alpha','beta','gamma'],'source')],[{'role':'user','content':response}])
            messages=self.materialize(item);step=item['steps'][0]
            self.assertEqual(len(step['responses']),1)
            self.assertEqual(step['responses'][0]['content'],response)
            self.assertEqual(step['responses'][0]['response_to'],'source')
            self.assertEqual(step['calls'][0]['arguments']['query'],['alpha','beta','gamma'])
            self.assertNotIn('target_call_response_alignment_required',item['harness']['flags'])
            validate_messages(messages,self.contract['schemas'])

    def test_text_search_renderer_is_query_keyed_not_order_keyed(self):
        queries=["an explorer's instrument",'另一个检索问题']
        for heading in ('','### '):
            content='\n\n'.join(heading+"A Example Engine search for '"+q+"' found 2 results:\n\n1. result "+str(i) for i,q in enumerate(reversed(queries)))
            item=self.item([self.call(queries,'batch')],[{'role':'tool','tool_call_id':'batch','content':content}])
            messages=self.materialize(item)
            self.assertEqual(item['steps'][0]['responses'][0]['content'],content)
            self.assertEqual(item['steps'][0]['calls'][0]['arguments']['query'],queries)
            validate_messages(messages,self.contract['schemas'])

    def test_historical_batch_return_is_not_forced_into_per_query_results(self):
        for payload in ([{'query':'a','result':'A'}],
                        [{'query':'a','result':'A'},{'query':'a','result':'other'}],
                        [{'query':'a','result':'A'},{'query':'b','result':'B'},{'query':'c','result':'C'}],
                        ['A','B']):
            item=self.item([self.call(['a','b'],'source')],[{'role':'tool','tool_call_id':'source','content':json.dumps(payload)}])
            messages=self.materialize(item)
            self.assertEqual(item['steps'][0]['responses'][0]['content'],json.dumps(payload))
            self.assertEqual(len(item['steps'][0]['responses']),1)
            validate_messages(messages,self.contract['schemas'])

    def test_grouped_responses_reverse_order_and_reminder(self):
        content='<tool_response>'+json.dumps({'query':'b','result':'B'})+'</tool_response>\n<tool_response>'+json.dumps({'query':'a','result':'A'})+'</tool_response>\nKeep dates in ISO format.'
        item=self.item([self.call('a'),self.call('b')],[{'role':'user','content':content}])
        messages=self.materialize(item);responses=item['steps'][0]['responses']
        self.assertEqual(len(responses),1)
        self.assertEqual([p['source_call_id'] for p in responses[0]['merged_response_parts']],['generated:m1:c1','generated:m1:c0'])
        original=item['steps'][0]['web_batch_provenance']['responses']
        self.assertEqual([r['source_envelope_index'] for r in original],[0,1])
        for response in original:
            self.assertEqual(content[response['source_content_start']:response['source_content_end']],response['content'])
        self.assertIn('Keep dates in ISO format.',item['context_messages'][0]['content'])
        validate_messages(messages,self.contract['schemas'])

    def test_grouped_equal_tools_without_identifiers_have_only_batch_association(self):
        text='<tool_response>A</tool_response>\n<tool_response>B</tool_response>'
        item=self.item([self.call('a'),self.call('b')],[{'role':'user','content':text}])
        self.assertEqual(item['program_errors'],[])
        self.assertEqual(len(item['steps'][0]['responses']),1)
        self.assertEqual(item['steps'][0]['call_links'][0]['association_scope'],'closed_round_call_batch')
        self.assertTrue(all('source_call_id' not in p for p in item['steps'][0]['responses'][0]['batch_response_parts']))
        validate_messages(self.materialize(item),self.contract['schemas'])

    def test_single_outstanding_call_does_not_require_an_unchanged_query_echo(self):
        for cid in (None,'native'):
            item=self.item([self.call('original request',cid)],[{'role':'tool','content':json.dumps({'query':'provider rewrite','result':'Evidence'})}])
            self.assertEqual(item['program_errors'],[])
            self.materialize(item)
            self.assertEqual(len(item['steps'][0]['responses']),1)

    def test_grouped_url_headers_identify_calls(self):
        urls=['https://example.org/one','https://example.org/two']
        calls=[{'function':{'name':'visit','arguments':{'url':u,'goal':'Read'}}} for u in urls]
        text='\n'.join('<tool_response>URL: '+u+'\nReport: recorded text</tool_response>' for u in reversed(urls))
        item=self.item(calls,[{'role':'user','content':text}]);self.materialize(item)
        response=item['steps'][0]['responses'][0]
        self.assertEqual(len(item['steps'][0]['calls']),1)
        self.assertEqual([r['source_call_id'] for r in response['merged_response_parts']],['generated:m1:c1','generated:m1:c0'])

    def test_headers_json_wrappers_and_single_url_keep_unknown_status(self):
        urls=['https://example.org/a','https://example.org/b']
        text='URL: '+urls[1]+'\nReport: B\nURL: '+urls[0]+'\nReport: A'
        result,reason=historical_results('<tool_response>Tool: visit_urls\nResult: '+text+'</tool_response>',urls)
        self.assertIsNone(reason);self.assertEqual([p['url'] for p in result['results']],list(reversed(urls)))
        self.assertTrue(all('error' not in p for p in result['results']))
        self.assertEqual(historical_results(text,['https://example.org/other'])[1],'extract_result_url_coverage_mismatch')
        self.assertIsNone(historical_results('unlabeled merged pages',urls)[0])
        result,reason=historical_results('Recorded page text',[urls[0]])
        self.assertIsNone(reason);self.assertEqual(result['results'][0]['content'],'Recorded page text')
        self.assertNotIn('error',result['results'][0])
        explicit={'results':[{'url':u,'content':'Page','error':None} for u in urls]}
        self.assertEqual(historical_results(json.dumps({'success':True,'data':explicit}),urls),(explicit,None))

    def test_historical_status_is_not_invented_or_a_default_format_gate(self):
        url='https://example.org/a'
        calls=[{'id':'x','function':{'name':'visit','arguments':{'url':url}}}]
        item=self.item(calls,[{'role':'tool','tool_call_id':'x','content':'Page text'}])
        messages=self.materialize(item)
        validate_messages(messages,self.contract['schemas'])
        with self.assertRaisesRegex(ValueError,'extract_page_status_unknown'):
            validate_messages(messages,self.contract['schemas'],require_page_status=True)
        self.assertNotIn('extract_page_status_unknown',item['harness']['flags'])

    def test_length_counts_rendered_content_once(self):
        item=self.item([self.call('a')],[{'role':'tool','content':'Evidence'}]);messages=self.materialize(item)
        count=model_visible_chars(item)
        self.assertEqual(count,sum(len(m['content']) for m in messages))
        item['provenance']['extra']='x'*300000
        item['steps'][0]['source_responses']=[{'content':'x'*400000}]
        self.assertEqual(model_visible_chars(item),count)

    def test_question_echo_and_natural_language_markers_do_not_trigger(self):
        item=self.item([self.call('a')],[{'role':'tool','content':'Evidence'}]);messages=self.materialize(item)
        question=item['question']
        for content in [json.dumps([{'query':question,'result':json.dumps([{'snippet':'Independent account'}])}]),
                        'Tool: search\nResult: '+json.dumps([{'query':question,'result':'Independent account'}]),
                        'Study protocol and ground truth generation. Help answer key questions about a project.']:
            candidate={'messages':copy.deepcopy(messages),'tools':[{'type':'function','function':s} for s in self.contract['schemas'].values()]}
            candidate['messages'][2]['content']='<tool_response>'+content+'</tool_response>'
            self.assertFalse(any(f['rule_id'].startswith('benchmark.') for f in self.scanner.scan(candidate)['findings']))
        candidate['messages'][2]['content']='<tool_response>'+question+'\nReference answer: a person</tool_response>'
        self.assertIn('benchmark.current_question_answer',{f['rule_id'] for f in self.scanner.scan(candidate)['findings']})

    def test_login_copy_is_not_a_credential_but_literal_values_are(self):
        for text in ['Sign in. Password: Register an account.','Password: Recover access now']:
            self.assertFalse(any(rule.startswith('credential.') for _,_,rule in self.scanner.sensitive_spans(text)))
        for text in ['password="opaque-actual-value"','password: opaque-actual-value','api_key=opaque-actual-value',
                     'Authorization: Bearer '+'z'*32,'https://example.org?access_token=opaque-value']:
            self.assertTrue(any(rule.startswith('credential.') for _,_,rule in self.scanner.sensitive_spans(text)),text)

    def test_answer_artifact_links_remain_visible_risks(self):
        from answer_leakage import external_signals
        question='Which numbered example was recorded during the survey at the ancient northern observatory in the summer of 1870?'
        for resource in ('[Answer Key](/archive/1870_Answer_Key)', 'https://example.org/answers/reference_answers.jsonl'):
            rules=dict(external_signals(question+'\n'+resource,question))
            self.assertIn('benchmark.current_question_answer',rules)
        self.assertNotIn('benchmark.reference',dict(external_signals('https://example.org/how-to-answer-key-questions',question)))

    def test_weak_bare_credentials_are_review_signals_not_ignored(self):
        for text in ('Meeting password: 123456 Telephonically available.', 'Resource password: exampleword Next field: document'):
            found=self.scanner.sensitive_spans(text)
            self.assertTrue(any(rule=='credential.unquoted_value' for _,_,rule in found))

    def test_basic_auth_requires_an_authorization_field(self):
        self.assertEqual(self.scanner.sensitive_spans('Basic conceptualization principles are discussed.'),[])
        self.assertTrue(self.scanner.sensitive_spans('Authorization: Basic '+'Y'*24))


if __name__=='__main__':unittest.main()
