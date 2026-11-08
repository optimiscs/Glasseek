import copy,json,unittest
from runtime_metadata import clean_response,clean_item,clean_messages
from ingest import normalize
from migration_render import target_messages
from tool_xml import serialize

class RuntimeMetadataTests(unittest.TestCase):
    def response(self,model='x',duration=9):
        return {'results':[{'task_index':0,'status':'completed','summary':'Evidence with model, tokens and duration_seconds. https://example.org',
                            'model':model,'tokens':{'input':100},'duration_seconds':duration,'api_calls':3,'tool_trace':[{'tool':'web_search'}],
                            'exit_reason':'completed','extra_business_field':{'model':'car','tokens':8}}], 'total_duration_seconds':duration,'unknown':42}
    def wrapped(self,p):return '<tool_response>\n'+json.dumps(p)+'\n</tool_response>\nPlease verify the citations.'
    def messages(self,p,name='delegate_task'):
        return [{'role':'user','content':'Question'},
                {'role':'assistant','content':'<think>Plan.</think>'+serialize({'name':name,'arguments':{'goal':'Find evidence'}})},
                {'role':'user','content':self.wrapped(p)},{'role':'assistant','content':'Final'}]
    def test_removes_only_telemetry_and_preserves_semantics(self):
        p=self.response();before=copy.deepcopy(p);text,event=clean_response(self.wrapped(p));body=json.loads(text.split('<tool_response>')[1].split('</tool_response>')[0])
        self.assertEqual(body,{'results':[{'task_index':0,'status':'completed','summary':p['results'][0]['summary'],'extra_business_field':{'model':'car','tokens':8}}],'unknown':42})
        self.assertEqual(p,before);self.assertTrue(text.endswith('Please verify the citations.'));self.assertEqual(event['removed_fields'],7)
        self.assertEqual(clean_response(text),(text,None))
    def test_errors_are_retained(self):
        p={'results':[{'task_index':1,'status':'failed','error':'network timeout','duration_seconds':30}]}
        text,event=clean_response(json.dumps(p));q=json.loads(text)
        self.assertEqual(q['results'][0],{'task_index':1,'status':'failed','error':'network timeout'})
    def test_legacy_escaped_envelope_and_quoted_xml(self):
        p=self.response();p['results'][0]['summary']='Quoted </tool_response> and model are evidence'
        text='<tool_response>\\n'+json.dumps(p)+'\\n</tool_response>'
        new,event=clean_response(text)
        self.assertTrue(new.startswith('<tool_response>\\n'));self.assertIn('Quoted </tool_response>',new);self.assertEqual(event['status'],'cleaned')
    def test_invalid_duplicate_and_unknown_json_are_not_modified(self):
        for text in ['{"results":[', '{"results":[],"results":[]}', '{"results":[{"model":"car"}],"duration_seconds":5}']:
            cleaned,event=clean_response(text);self.assertEqual(cleaned,text);self.assertEqual(event['status'],'skipped')
        text='{"model":"car","tokens":5}';self.assertEqual(clean_response(text),(text,None))
    def test_non_delegate_and_user_text_never_cleaned(self):
        for name in ('web_search','execute_code'):
            msgs=self.messages(self.response(),name);self.assertEqual(clean_messages(msgs),(msgs,[]))
        msgs=[{'role':'user','content':self.wrapped(self.response())}];self.assertEqual(clean_messages(msgs),(msgs,[]))
    def test_normalize_before_dedup_and_preserve_source_hash(self):
        a=normalize({'question':'Question','messages':self.messages(self.response('model_a',5))},'x','x',0)
        b=normalize({'question':'Question','messages':self.messages(self.response('model_b',55))},'x','x',1)
        self.assertEqual(a['content_hash'],b['content_hash']);self.assertNotEqual(a['provenance']['raw_sha256'],b['provenance']['raw_sha256'])
        self.assertTrue(a['runtime_metadata_cleaning']['events'])
    def test_old_candidate_export_is_cleaned_and_input_unchanged(self):
        item={'agent_role':'parent','messages':self.messages(self.response())};before=copy.deepcopy(item)
        messages,changes=target_messages(item)
        self.assertIn('delegate_runtime_metadata_removed',changes);self.assertEqual(item,before)
        self.assertNotIn('total_duration_seconds',messages[2]['content']);self.assertIn('Plan.',messages[1]['content'])
    def test_native_ids_and_ambiguous_calls(self):
        calls=[{'id':'a','type':'function','function':{'name':'delegate_task','arguments':'{"goal":"q"}'}},
               {'id':'b','type':'function','function':{'name':'web_search','arguments':'{"query":"q"}'}}]
        msgs=[{'role':'assistant','content':'','tool_calls':calls},
              {'role':'tool','tool_call_id':'b','content':json.dumps(self.response())},
              {'role':'tool','tool_call_id':'a','content':json.dumps(self.response())}]
        cleaned,events=clean_messages(msgs);self.assertEqual(cleaned[1],msgs[1]);self.assertEqual(len(events),1)
        msgs[-1].pop('tool_call_id');cleaned,events=clean_messages(msgs);self.assertEqual(cleaned,msgs);self.assertFalse(events)

if __name__=='__main__':unittest.main()
