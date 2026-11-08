import unittest
import json

import jsonschema

from export_review import adapt_assistant, convert, current_tools
from tool_xml import normalize_text, serialize


class ReviewExportTests(unittest.TestCase):
    def test_materialized_multicall_and_five_tool_contract(self):
        calls=[{'name':'web_search','arguments':{'query':'x','limit':5}},
               {'name':'execute_code','arguments':{'code':'print(4)'}}]
        text='\n'.join(serialize(c) for c in calls)
        self.assertEqual(adapt_assistant({'assistant':text,'calls':calls},calls),text)
        self.assertEqual(set(current_tools()),{'web_search','web_extract','execute_code','delegate_task','vision_analyze'})

    def test_batched_search_becomes_scalar_xml_calls(self):
        old={'name':'search','arguments':{'query':['alpha','beta']}}
        step={'assistant':'<think>Plan search.</think>\n'+serialize(old),'calls':[old]}
        new=[{'name':'web_search','arguments':{'query':q,'limit':5}} for q in ('alpha','beta')]
        text=adapt_assistant(step,new)
        self.assertEqual(normalize_text(text)[1],new)
        self.assertNotIn('<function=search>',text)

    def test_schema_uses_current_plural_extract_parameter(self):
        schemas=current_tools()
        jsonschema.validate({'urls':['https://example.invalid']},schemas['web_extract']['parameters'])
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate({'url':'https://example.invalid'},schemas['web_extract']['parameters'])

    def test_export_never_auto_approves(self):
        item={'id':'x','source':'test','question':'What is the recorded label?',
              'steps':[{'step_id':1,'assistant':'<answer>A</answer>','calls':[],'observation':''}],
              'harness':{'errors':[],'calls':[],'flags':[]}}
        row=convert({'sample':item,'result':{'offline_quality_candidate':True,'sample_hash':'test'}},current_tools())
        self.assertFalse(row['training_ready']);self.assertFalse(row['harness_replayed'])
        self.assertEqual(row['status'],'review_only')

    def test_counterfactual_cannot_be_positive_sft(self):
        item={'lineage':{'operation':'counterfactual'}}
        with self.assertRaisesRegex(ValueError,'Negative'):
            convert({'sample':item,'result':{'offline_quality_candidate':True}},current_tools())

    def test_user_reminder_preserved_before_its_response(self):
        item={'id':'x','source':'test','question':'Question',
              'steps':[{'step_id':1,'assistant':'Draft','calls':[],'observation':''},
                       {'step_id':2,'assistant':'Final','calls':[],'observation':''}],
              'context_messages':[{'role':'user','content':'Use XML','before_step':2}],
              'harness':{'errors':[],'calls':[],'flags':[]}}
        row=convert({'sample':item,'result':{'offline_quality_candidate':True,'sample_hash':'test'}},current_tools())
        messages=json.loads(row['messages'])
        self.assertEqual([m['content'] for m in messages][-3:],['Draft','Use XML','Final'])


if __name__=='__main__':unittest.main()
