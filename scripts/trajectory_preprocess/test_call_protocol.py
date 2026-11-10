import copy
import json
from pathlib import Path
import tempfile
import unittest

from ingest import normalize, harness_plan
from glasseek_contract import build_contract, materialize
from migrate_searchswarm import review_messages
from call_protocol import semantic_messages
from safety_context import build_context
from glasseek_shared.extract_protocol import historical_results, runtime_results, validate_urls
from glasseek_shared.trajectory_release import (digest, payload, runtime_schemas,
    validate_messages, verify_release_file)
from publish_release import publish


class CallProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = build_contract()
        cls.config = {'search_name':'web_search','extract_name':'web_extract',
                      'extract_max_urls':5,'contract':cls.contract}

    def item(self, calls, responses):
        return normalize({'messages':[{'role':'user','content':'Question'},
            {'role':'assistant','tool_calls':calls}, *responses,
            {'role':'assistant','content':'Answer'}]}, 'test', 'fixture', 0)

    def call(self, cid='a', name='web_search', args=None):
        return {'id':cid,'function':{'name':name,'arguments':args or {'query':'test'}}}

    def render(self, item):
        item['harness'] = harness_plan(item,self.config)
        self.assertEqual(item['harness']['errors'],[])
        materialize(item)
        return review_messages(item)

    def test_reversed_response_ids_keep_identity_and_source_positions(self):
        item=self.item([self.call('a'),self.call('b',args={'query':'test','limit':6})],[
            {'role':'tool','tool_call_id':'b','content':'B'},
            {'role':'tool','tool_call_id':'a','content':'A'}])
        messages=self.render(item)
        self.assertEqual(item['program_errors'],[])
        self.assertEqual([r['response_to'] for r in item['steps'][0]['responses']],['b','a'])
        self.assertEqual([r['source_message_index'] for r in item['steps'][0]['responses']],[2,3])
        validate_messages(messages,runtime_schemas())
        self.assertEqual(build_context({'messages':messages})[2],[])

    def test_missing_duplicate_and_unknown_responses_fail_closed(self):
        base=[self.call('a'),self.call('b')]
        item=self.item(base,[{'role':'tool','tool_call_id':'a','content':'A'}])
        self.assertIn('missing_tool_response',item['program_errors'])
        for response_id in ('a','unknown'):
            item=self.item([self.call('a')],[{'role':'tool','tool_call_id':'a','content':'A'},
                {'role':'tool','tool_call_id':response_id,'content':'extra'}])
            self.assertIn('unmatched_tool_response_id',item['program_errors'])
            self.assertEqual(item['unassigned_responses'][0]['content'],'extra')

    def test_no_id_only_unique_response_is_mapped(self):
        item=self.item([self.call(None)],[{'role':'tool','content':'A'}])
        self.assertEqual(item['program_errors'],[])
        self.assertTrue(item['steps'][0]['responses'][0]['response_to'].startswith('generated:'))
        item=self.item([self.call(None),self.call(None,args={'query':'test','limit':6})],[{'role':'tool','content':'A and B'}])
        self.assertIn('ambiguous_tool_response',item['program_errors'])
        self.assertEqual(item['steps'][0]['responses'],[])
        self.assertEqual(item['unassigned_responses'][0]['content'],'A and B')

    def test_names_can_disambiguate_different_tools(self):
        item=self.item([self.call(None),self.call(None,'execute_code',{'code':'print(1)'})],[
            {'role':'tool','name':'execute_code','content':'1'},
            {'role':'tool','name':'web_search','content':'A'}])
        self.assertEqual(item['program_errors'],[])
        self.assertEqual([r['response_to'] for r in item['steps'][0]['responses']],
                         ['generated:m1:c1','generated:m1:c0'])

    def test_url_batch_and_error_envelope_preserved(self):
        urls=['https://example.org/a','https://example.org/b']
        response={'results':[{'url':urls[1],'content':'','error':'unavailable'},
                             {'url':urls[0],'content':'A','error':None}]}
        item=self.item([self.call('native','visit',{'url':urls,'goal':'Find dates'})],[
            {'role':'tool','tool_call_id':'native','content':json.dumps(response)}])
        messages=self.render(item)
        self.assertEqual(len(item['steps'][0]['calls']),1)
        self.assertEqual(item['steps'][0]['calls'][0]['arguments'],{'urls':urls,'goal':'Find dates'})
        self.assertEqual(messages[2]['role'],'user')
        self.assertEqual(json.loads(messages[2]['content'].strip()[len('<tool_response>'):-len('</tool_response>')]),response)
        validate_messages(messages,runtime_schemas())

    def test_batch_text_is_preserved_without_inventing_page_status(self):
        urls=['https://example.org/a','https://example.org/b']
        item=self.item([self.call('native','visit',{'url':urls})],[
            {'role':'tool','tool_call_id':'native','content':'Merged prose with no page boundaries'}])
        messages=self.render(item)
        self.assertEqual(messages[2]['content'],'<tool_response>\nMerged prose with no page boundaries\n</tool_response>')
        self.assertNotIn('extract_page_boundaries_unverified',item['harness']['flags'])
        validate_messages(messages,runtime_schemas())
        with self.assertRaisesRegex(ValueError,'boundaries'):
            validate_messages(messages,runtime_schemas(),require_page_status=True)
        original={'results':[{'url':u,'content':'text'} for u in urls]}
        converted,reason=historical_results(json.dumps(original),urls)
        self.assertIsNone(reason)
        self.assertEqual(converted,original)
        self.assertNotIn('error',converted['results'][0])

    def test_more_than_five_is_held_without_truncating(self):
        urls=['https://example.org/'+str(i) for i in range(6)]
        item=self.item([self.call('a','visit',{'url':urls})],[{'role':'tool','tool_call_id':'a','content':'text'}])
        plan=harness_plan(item,self.config)
        self.assertIn('extract_url_batch_limit_requires_explicit_split',plan['errors'])
        self.assertEqual(plan['calls'],[])
        self.assertEqual(item['steps'][0]['calls'][0]['arguments']['url'],urls)
        with self.assertRaises(ValueError):validate_urls(urls)

    def test_multi_query_search_keeps_one_opaque_batch_response(self):
        item=self.item([self.call('a','search',{'query':['x','y']})],[{'role':'tool','tool_call_id':'a','content':'Merged'}])
        messages=self.render(item)
        self.assertNotIn('target_call_response_alignment_required',item['harness']['flags'])
        self.assertEqual(item['steps'][0]['calls'][0]['arguments']['query'],['x','y'])
        self.assertEqual(len(item['steps'][0]['responses']),1)
        self.assertEqual(item['steps'][0]['responses'][0]['response_to'],'a')
        self.assertEqual(item['steps'][0]['responses'][0]['content'],'Merged')
        validate_messages(messages,runtime_schemas())

    def test_duplicate_json_keys_and_wrong_url_coverage_held(self):
        self.assertIsNone(historical_results('{"results":[],"results":[]}',[])[0])
        text=json.dumps({'results':[{'url':'https://b','content':'B','error':None}]})
        self.assertEqual(historical_results(text,['https://a'])[1],'extract_result_url_coverage_mismatch')

    def test_provider_order_missing_and_extra_urls(self):
        results=runtime_results(['a','b'],[{'url':'b','content':'B'}])
        self.assertEqual([r['url'] for r in results],['a','b'])
        self.assertIsNotNone(results[0]['error'])
        self.assertIsNone(results[1]['error'])
        with self.assertRaises(ValueError):runtime_results(['a'],[{'url':'other','content':'x'}])

    def test_dedup_ignores_id_spelling_but_keeps_binding(self):
        a=self.render(self.item([self.call('a')],[{'role':'tool','tool_call_id':'a','content':'A'}]))
        b=self.render(self.item([self.call('different')],[{'role':'tool','tool_call_id':'different','content':'A'}]))
        self.assertEqual(semantic_messages(a),semantic_messages(b))

    def test_release_and_loader_reject_tampering_and_missing_approval(self):
        item=self.item([self.call('a')],[{'role':'tool','tool_call_id':'a','content':'A'}])
        candidate={'id':'fixture','messages':self.render(item),
            'tools':[{'type':'function','function':s} for s in runtime_schemas().values()]}
        # Acceptance is bound after presentation normalization, as in production.
        candidate['messages'][-1]['content']='<answer>done</answer>'
        approval={'id':'fixture','content_sha256':digest(payload(candidate)),
                  'safety':'passed','quality':'passed','replay':'passed','review':'passed'}
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=d/'input.jsonl';reviews=d/'reviews.jsonl'
            source.write_text(json.dumps(candidate)+'\n');reviews.write_text(json.dumps(approval)+'\n')
            manifest=publish(source,reviews,d/'release')
            self.assertEqual(manifest['rows'],1)
            path=d/'release/train.parquet'
            verify_release_file(path)
            with path.open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'hash'):verify_release_file(path)
            reviews.write_text('')
            manifest=publish(source,reviews,d/'held')
            self.assertEqual(manifest['rows'],0)
            self.assertFalse((d/'held/train.parquet').exists())
            with self.assertRaisesRegex(ValueError,'acceptance'):verify_release_file(d/'held/train.parquet')

    def test_stale_approval_and_changed_schema_rejected(self):
        candidate={'id':'fixture','messages':[{'role':'user','content':'q'},{'role':'assistant','content':'<answer>a</answer>'}],
                   'tools':[{'type':'function','function':s} for s in runtime_schemas().values()]}
        approval={'id':'fixture','content_sha256':'stale','safety':'passed','quality':'passed','replay':'passed','review':'passed'}
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=d/'input';review=d/'approval'
            source.write_text(json.dumps(candidate)+'\n');review.write_text(json.dumps(approval)+'\n')
            manifest=publish(source,review,d/'stale')
            self.assertEqual(manifest['held_reasons'],{'stale_acceptance_content_hash':1})
            approval['content_sha256']=digest(payload(candidate));review.write_text(json.dumps(approval)+'\n')
            publish(source,review,d/'ok')
            with self.assertRaisesRegex(ValueError,'schema'):
                verify_release_file(d/'ok/train.parquet',{})

    def test_approval_cannot_bypass_text_normalization(self):
        candidate={'id':'fixture','messages':[{'role':'user','content':'q'},{'role':'assistant','content':'plain answer'}],
                   'tools':[{'type':'function','function':s} for s in runtime_schemas().values()]}
        approval={'id':'fixture','content_sha256':digest(payload(candidate)),
                  'safety':'passed','quality':'passed','replay':'passed','review':'passed'}
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=d/'input';review=d/'approval'
            source.write_text(json.dumps(candidate)+'\n');review.write_text(json.dumps(approval)+'\n')
            manifest=publish(source,review,d/'held')
            self.assertEqual(manifest['rows'],0)
            self.assertEqual(manifest['held_reasons'],{'training_text_normalization_required':1})

    def test_excluded_source_and_qwen_external_origin(self):
        with self.assertRaisesRegex(ValueError,'excluded_dataset_source'):
            normalize({'messages':[]},'OpenResearcher','fixture',0)
        item=self.item([self.call('a')],[{'role':'tool','tool_call_id':'a','content':'Evidence'}])
        messages=self.render(item)
        self.assertEqual(messages[2]['role'],'user')
        self.assertTrue(messages[2]['content'].startswith('<tool_response>'))
        self.assertEqual(messages[2]['response_to'],'a')
        views,_,issues=build_context({'messages':messages})
        self.assertEqual(issues,[])
        self.assertEqual(next(v.kind for v in views if v.pointer=='/messages/2/content'),'external')

    def test_qwen_suffix_reminder_is_preserved_as_task_context(self):
        content='<tool_response>Evidence</tool_response>\nUse metric units.'
        item=self.item([self.call('a')],[{'role':'user','tool_call_id':'a','content':content}])
        messages=self.render(item)
        self.assertEqual(messages[2]['content'],'<tool_response>Evidence</tool_response>')
        self.assertEqual(messages[3]['content'],'\nUse metric units.')
        self.assertEqual(messages[2]['source_message_index'],messages[3]['source_message_index'])
        validate_messages(messages,runtime_schemas())
        views,_,issues=build_context({'messages':messages})
        self.assertEqual(issues,[])
        self.assertEqual(next(v.kind for v in views if v.pointer=='/messages/3/content'),'task')


if __name__ == '__main__':unittest.main()
