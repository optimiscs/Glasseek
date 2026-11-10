"""Unified batch behavior, closed-turn proofs and fail-closed counterexamples."""
import copy
import json
from pathlib import Path
import unittest

from ingest import normalize,harness_plan,semantic_hash
from glasseek_contract import build_contract,materialize
from migrate_searchswarm import review_messages
from glasseek_shared.trajectory_release import validate_messages
from safety_context import build_context
from safety_rules import Scanner


class WebBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract=build_contract()
        cls.config={'search_name':'web_search','extract_name':'web_extract','extract_max_urls':5,'contract':cls.contract}

    def call(self,name,args,cid=None):return {'id':cid,'function':{'name':name,'arguments':args}}

    def normalize(self,calls,responses,tail=None,question='Find the publication date.'):
        raw={'messages':[{'role':'user','content':question},
                         {'role':'assistant','tool_calls':calls},*responses,
                         *(tail or [{'role':'assistant','content':'Recorded answer.'}])]}
        before=copy.deepcopy(raw);item=normalize(raw,'synthetic','fixture',0)
        self.assertEqual(raw,before)
        return item

    def render(self,item):
        item['harness']=harness_plan(item,self.config)
        self.assertEqual(item['program_errors'],[]);self.assertEqual(item['harness']['errors'],[])
        materialize(item);messages=review_messages(item)
        validate_messages(messages,self.contract['schemas'])
        self.assertEqual(build_context({'messages':messages})[2],[])
        return messages

    def test_search_native_responses_merge_in_response_order(self):
        calls=[self.call('search',{'query':'A'},'a'),self.call('web_search',{'query':'B','limit':5},'b')]
        item=self.normalize(calls,[{'role':'tool','tool_call_id':'b','content':'B result'},
                                   {'role':'tool','tool_call_id':'a','content':'A result'}])
        self.render(item);step=item['steps'][0]
        self.assertEqual(step['calls'],[{'name':'web_search','arguments':{'query':['A','B'],'limit':5}}])
        self.assertEqual(step['responses'][0]['content'],'B result\n\nA result')
        self.assertEqual(step['call_links'][0]['source_call_ids'],['a','b'])
        self.assertEqual([p['source_call_id'] for p in step['responses'][0]['merged_response_parts']],['b','a'])

    def test_closed_round_opaque_returns_use_batch_only_for_both_tools(self):
        for name,args in [('search',[{'query':'A'},{'query':'B'}]),
                          ('visit',[{'url':'https://example.org/a','goal':'Read'},{'url':'https://example.org/b','goal':'Read'}])]:
            for text in ('<tool_response>One opaque batch return.</tool_response>',
                         '<tool_response>First body</tool_response>\n<tool_response>Second body</tool_response>'):
                item=self.normalize([self.call(name,a) for a in args],[{'role':'user','content':text}])
                messages=self.render(item);step=item['steps'][0]
                self.assertEqual(len(step['calls']),1);self.assertNotIn('unassigned_responses',item)
                self.assertEqual(step['call_links'][0]['association_scope'],'closed_round_call_batch')
                response=next(m for m in messages if m.get('batch_response_parts'))
                self.assertTrue(all('source_call_id' not in p for p in response['batch_response_parts']))
                self.assertEqual(response['source_call_ids'],step['call_links'][0]['source_call_ids'])
                self.assertEqual(step['web_batch_provenance']['unassigned_responses'][0]['content'],text.split('</tool_response>')[0]+'</tool_response>')

    def test_known_missing_native_response_is_not_hidden_by_a_batch(self):
        calls=[self.call('search',{'query':'A'},'a'),self.call('search',{'query':'B'},'b')]
        item=self.normalize(calls,[{'role':'tool','tool_call_id':'a','content':'Only A'}])
        self.assertIn('missing_tool_response',item['program_errors'])
        self.assertNotIn('web_batch_provenance',item['steps'][0])

    def test_mixed_tools_different_limits_or_goals_are_not_batch_repaired(self):
        pairs=[ [('search',{'query':'A'}),('visit',{'url':'https://example.org/a'})],
                [('search',{'query':'A','limit':5}),('search',{'query':'B','limit':10})],
                [('visit',{'url':'https://example.org/a','goal':'A'}),('visit',{'url':'https://example.org/b','goal':'B'})]]
        for pair in pairs:
            item=self.normalize([self.call(n,a) for n,a in pair],[{'role':'user','content':'<tool_response>Opaque</tool_response>'}])
            self.assertIn('ambiguous_tool_response',item['program_errors'])
            self.assertNotIn('web_batch_provenance',item['steps'][0])

    def test_context_interruption_prevents_repair(self):
        calls=[self.call('search',{'query':'A'}),self.call('search',{'query':'B'})]
        responses=[{'role':'tool','content':'One body'}, {'role':'user','content':'A revised goal'}, {'role':'tool','content':'Another body'}]
        item=self.normalize(calls,responses)
        self.assertIn('ambiguous_tool_response',item['program_errors'])
        self.assertNotIn('web_batch_provenance',item['steps'][0])

    def test_explicit_identifier_conflict_cannot_be_batch_repaired(self):
        calls=[self.call('search',{'query':'A'},'a'),self.call('search',{'query':'B'},'b')]
        item=self.normalize(calls,[{'role':'tool','tool_call_id':'unknown','content':'Opaque'}])
        self.assertIn('unmatched_tool_response_id',item['program_errors'])
        self.assertNotIn('web_batch_provenance',item['steps'][0])

    def test_a_query_labeled_single_result_cannot_hide_an_unanswered_query(self):
        calls=[self.call('search',{'query':'A'}),self.call('search',{'query':'B'})]
        item=self.normalize(calls,[{'role':'tool','content':json.dumps({'query':'A','result':'Only A'})}])
        self.assertIn('missing_tool_response',item['program_errors'])
        self.assertNotIn('web_batch_provenance',item['steps'][0])

    def test_repaired_batch_does_not_poison_the_next_round(self):
        calls=[self.call('search',{'query':'A'}),self.call('search',{'query':'B'})]
        tail=[{'role':'assistant','tool_calls':[self.call('search',{'query':'A'})]},
              {'role':'tool','content':json.dumps({'query':'A','result':'Next round evidence'})},
              {'role':'assistant','content':'Recorded answer.'}]
        item=self.normalize(calls,[{'role':'tool','content':'Opaque first-round batch'}],tail)
        self.render(item)
        self.assertEqual([len(s['calls']) for s in item['steps']],[1,1,0])
        self.assertEqual(item['steps'][0]['responses'][0]['source_message_indices'],[2])
        self.assertEqual(item['steps'][1]['responses'][0]['source_message_index'],4)

    def test_later_round_response_is_never_pulled_into_an_earlier_batch(self):
        calls=[self.call('search',{'query':'A'}),self.call('search',{'query':'B'})]
        tail=[{'role':'assistant','content':'Intervening assistant.'},
              {'role':'tool','content':json.dumps({'query':'A','result':'Late response'})},
              {'role':'assistant','content':'Answer'}]
        item=self.normalize(calls,[{'role':'tool','content':'Opaque earlier return'}],tail)
        self.assertIn('ambiguous_tool_response',item['program_errors'])
        self.assertNotIn('web_batch_provenance',item['steps'][0])

    def test_search_six_queries_is_held_without_truncation(self):
        qs=[str(i) for i in range(6)]
        item=self.normalize([self.call('search',{'query':qs},'a')],[{'role':'tool','tool_call_id':'a','content':'Opaque'}])
        item['harness']=harness_plan(item,self.config)
        self.assertIn('search_query_batch_limit_requires_explicit_split',item['harness']['errors'])
        self.assertEqual(item['steps'][0]['calls'][0]['arguments']['query'],qs)

    def test_mixed_tools_keep_batches_and_use_explicit_header_types(self):
        calls=[self.call('search',{'query':['A','B']}),self.call('visit',{'url':['https://example.org/a','https://example.org/b'],'goal':'Read'})]
        text='<tool_response>A Example Engine search for \'A\' found 1 results:\nSearch evidence.</tool_response>\n<tool_response>URL: https://example.org/a\nReport: Recorded combined page report.</tool_response>'
        item=self.normalize(calls,[{'role':'user','content':text}])
        self.render(item)
        self.assertEqual(len(item['steps'][0]['calls']),2)
        self.assertEqual([r['source_envelope_index'] for r in item['steps'][0]['responses']],[0,1])
        self.assertEqual(item['steps'][0]['calls'][0]['arguments']['query'],['A','B'])
        self.assertEqual(item['steps'][0]['calls'][1]['arguments']['urls'],['https://example.org/a','https://example.org/b'])

    def test_batch_spans_and_membership_tampering_fail_closed(self):
        item=self.normalize([self.call('search',{'query':'A'}),self.call('search',{'query':'B'})],
                            [{'role':'user','content':'<tool_response>Opaque evidence</tool_response>'}])
        messages=self.render(item)
        for field in ('content','members'):
            forged=copy.deepcopy(messages);r=next(m for m in forged if m.get('batch_response_parts'))
            if field=='content':r['content']=r['content'].replace('evidence','tampered')
            else:r['source_call_ids']=['fabricated']
            with self.assertRaisesRegex(ValueError,'invalid_(batch_response_parts|search_return_audit)'):validate_messages(forged,self.contract['schemas'])

    def test_batch_query_echo_is_not_evidence_but_secrets_are_scanned(self):
        q='Which explorer discovered the ancient instrument near the southern observatory during the summer expedition of 1850?'
        def item(text):
            result=self.normalize([self.call('search',{'query':q}),self.call('search',{'query':q})],
                                  [{'role':'user','content':'<tool_response>'+text+'</tool_response>'}],question=q)
            messages=self.render(result)
            return {'messages':messages,'tools':[{'type':'function','function':s} for s in self.contract['schemas'].values()]}
        policy=json.loads(Path(__file__).with_name('safety_policy.json').read_text())
        scanner=Scanner(policy,self.contract,b'synthetic-batch-policy-key-not-real')
        text=json.dumps({'query':q,'result':'Independent evidence.'})
        benign=item(text)
        self.assertFalse(any(f['rule_id'].startswith('benchmark.') for f in scanner.scan(benign)['findings']))
        self.assertIn('credential.token',{f['rule_id'] for f in scanner.scan(item(text+'\n'+'sk-'+'x'*32))['findings']})


if __name__=='__main__':unittest.main()
