import json
import unittest
from ingest import normalize, harness_plan
from message_adapter import AdapterUnsupported, task_text
from tool_xml import normalize_text
from dispositions import categorize

HARNESS={'search_name':'web_search','extract_name':'web_extract','extract_max_urls':5}


class MessageAdapterTests(unittest.TestCase):
    def item(self,messages):
        return normalize({'messages':messages},'synthetic','fixture',0)

    def test_structured_routing_and_channels_preserve_observation(self):
        text=lambda t:[{'type':'text','text':t}]
        item=self.item([
            {'role':'system','content':[{'type':'system_content','tools':{'browser':{}},'text':None}]},
            {'role':'user','content':text('Find the year')},
            {'role':'assistant','channel':'analysis','content':text('Plan a search')},
            {'role':'assistant','channel':'analysis','recipient':'browser.search','content':text('{"query":"example","topn":7}')},
            {'role':'tool','name':'browser.search','content':text('Evidence: 2020')},
            {'role':'assistant','channel':'final','content':text('2020')}])
        self.assertIn('<think>',item['steps'][0]['assistant'])
        self.assertEqual(item['steps'][1]['observation'],'Evidence: 2020\n')
        self.assertIn('<function=browser.search>',item['steps'][1]['assistant'])
        self.assertEqual(harness_plan(item,HARNESS)['calls'][0]['arguments'],{'query':'example','limit':7})
        self.assertEqual(harness_plan(item,HARNESS)['errors'],[])

    def test_reminders_and_changed_requests_remain_timed_context_not_evidence(self):
        item=self.item([{'role':'user','content':'Original question'},
                        {'role':'assistant','content':'Draft'},
                        {'role':'user','content':'Use XML. Also give the population.'},
                        {'role':'assistant','content':'Final'}])
        self.assertEqual(item['context_messages'][0]['before_step'],2)
        self.assertEqual(item['steps'][0]['observation'],'')
        self.assertIn('before assistant step 2',task_text(item))
        self.assertEqual(item['question'],'Original question')
        changed=json.loads(json.dumps(item));changed['context_messages'][0]['content']='Different constraint'
        self.assertNotEqual(normalize(changed,'synthetic','fixture',0)['content_hash'],item['content_hash'])

    def test_native_argument_strings_and_aliases(self):
        for name in ('Visit','visit_urls','WEB_EXTRACT','webextract'):
            item=self.item([{'role':'user','content':'q'},
                            {'role':'assistant','tool_calls':[{'function':{'name':name,'arguments':json.dumps({'url':'["https://example.org"]'})}}]},
                            {'role':'tool','content':'evidence'}, {'role':'assistant','content':'answer'}])
            self.assertEqual(harness_plan(item,HARNESS)['errors'],[])
            self.assertEqual(harness_plan(item,HARNESS)['calls'][0]['arguments'],{'urls':['https://example.org']})

    def test_browser_cursor_is_not_invented_as_url(self):
        for target in (3,'/tmp/source.xlsx'):
            item=self.item([{'role':'user','content':'q'},
                            {'role':'assistant','recipient':'browser.open','content':json.dumps({'id':target})},
                            {'role':'tool','content':'[1] Page (https://example.org)'},
                            {'role':'assistant','content':'answer'}])
            plan=harness_plan(item,HARNESS)
            self.assertIn('unsupported_tool:browser.open.cursor_or_local_resource',plan['errors'])
            self.assertEqual(plan['calls'],[])

    def test_direct_browser_url_maps_but_find_stays_unsupported(self):
        for name,args in [('browser.open',{'id':'https://example.org'}),('browser.find',{'pattern':'term'})]:
            item=self.item([{'role':'user','content':'q'},{'role':'assistant','recipient':name,'content':json.dumps(args)},
                            {'role':'tool','name':name,'content':'evidence'},{'role':'assistant','channel':'final','content':'answer'}])
            plan=harness_plan(item,HARNESS)
            if name.endswith('open'):self.assertEqual(plan['calls'][0]['name'],'web_extract')
            else:self.assertEqual(plan['errors'],['unsupported_tool:browser.find'])

    def test_multipart_code_is_preserved_not_treated_as_web_search(self):
        for tail in ('<code>print(2)</code>','{"code":"print(2)"}'):
            text,calls=normalize_text('<tool_call>{"name":"PythonInterpreter","arguments":{}}\n'+tail+'</tool_call>')
            self.assertEqual(calls[0]['arguments']['code'],'print(2)')
            self.assertIn('<parameter=code>',text)
            item=normalize({'question':'q','messages':[{'role':'assistant','content':text},{'role':'tool','content':'2'},
                                                      {'role':'assistant','content':'2'}]},'synthetic','fixture',0)
            self.assertEqual(harness_plan(item,HARNESS)['errors'],['unsupported_tool:PythonInterpreter'])

    def test_non_text_blocks_are_not_silently_dropped(self):
        with self.assertRaises(AdapterUnsupported):
            self.item([{'role':'user','content':[{'type':'image_url','image_url':'https://example.org/image'}]}])

    def test_tool_response_ids_are_matched_and_unknown_ids_flagged(self):
        msgs=[{'role':'user','content':'q'},
              {'role':'assistant','tool_calls':[{'id':'call1','function':{'name':'search','arguments':'{"query":"q"}'}}]},
              {'role':'assistant','channel':'analysis','content':'Waiting'},
              {'role':'tool','tool_call_id':'call1','content':'Evidence'},
              {'role':'assistant','content':'Answer'}]
        item=self.item(msgs)
        self.assertEqual(item['steps'][0]['observation'],'Evidence\n')
        self.assertEqual(item['steps'][1]['observation'],'')
        msgs[3]['tool_call_id']='missing'
        self.assertIn('unmatched_tool_response_id',self.item(msgs)['program_errors'])

    def test_empty_and_invalid_calls_still_fail(self):
        with self.assertRaisesRegex(ValueError,'Empty tool call'):
            normalize_text('<tool_call></tool_call>')
        item=self.item([{'role':'user','content':'q'},
                        {'role':'assistant','recipient':'browser.search','content':'{}'},
                        {'role':'tool','content':'error'},{'role':'assistant','content':'answer'}])
        self.assertIn('invalid_search_query',harness_plan(item,HARNESS)['errors'])
        self.assertEqual(categorize(['missing_final_answer']),['incomplete_episodes'])
        self.assertEqual(categorize(['unsupported_tool:browser.find']),['unsupported_tools'])

    def test_conflicting_alias_arguments_do_not_pass(self):
        item=self.item([{'role':'user','content':'q'},
                        {'role':'assistant','tool_calls':[{'name':'visit','arguments':{'url':'https://a.org','urls':['https://b.org']}}]},
                        {'role':'tool','content':'evidence'},{'role':'assistant','content':'answer'}])
        self.assertIn('conflicting_extract_urls',harness_plan(item,HARNESS)['errors'])


if __name__=='__main__':unittest.main()
