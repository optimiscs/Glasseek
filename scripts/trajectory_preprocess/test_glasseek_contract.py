import copy
import unittest
from glasseek_contract import build_contract, code_checks, validate_args, materialize, decode_typed
from ingest import normalize, harness_plan, semantic_hash
from tool_xml import normalize_text

class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract=build_contract()
        cls.config={'search_name':'web_search','extract_name':'web_extract','extract_max_urls':5,'contract':cls.contract}

    def test_contract_tracks_active_tools(self):
        self.assertEqual(set(self.contract['schemas']),{'web_search','web_extract','execute_code','delegate_task','vision_analyze'})
        self.assertEqual(self.contract['code_policy']['python_syntax'],[3,8])
        self.assertIn('execute_code',self.contract['delegation']['blocked_child_tools'])

    def test_pure_compute_and_external_state(self):
        self.assertEqual(code_checks('import math\nn=144\nprint(math.isqrt(n))'),[])
        self.assertEqual(code_checks('from collections import Counter\ndef f(x):\n return Counter(x)\nprint(f([1,1]))'),[])
        for code in ['import requests\nprint(requests.get("https://example.com"))','print(open("/etc/passwd").read())','print(previous_result)','match 1:\n case 1: print(1)']:
            self.assertTrue(code_checks(code),code)

    def test_pinned_scientific_dependencies_and_io_targets(self):
        from dispositions import categorize
        for code, module in [('import numpy as np\nprint(np.mean([1,2]))', 'numpy'),
                             ('from scipy.optimize import minimize', 'scipy'),
                             ('import pandas as pd', 'pandas'),
                             ('import matplotlib.pyplot as plt', 'matplotlib')]:
            errors = code_checks(code)
            self.assertEqual(errors, [])
            self.assertEqual(categorize(errors), [])
            self.assertEqual(validate_args('execute_code', {'code':code}, self.contract), [])
        self.assertEqual(code_checks('import textwrap\nprint(textwrap.dedent(" a"))'), [])
        for code in ['import pandas as pd\npd.read_csv("https://example.com/x")',
                     'from numpy import load as restore\nrestore("/home/private/x.npy")',
                     'import matplotlib.pyplot as plt\nplt.savefig("/home/private/x.png")',
                     'from sympy import sympify\nsympify("input")']:
            self.assertIn('tool_runtime:execute_code.dynamic_or_io', code_checks(code))

    def test_memory_and_private_temp_are_not_host_io(self):
        for code in ['import pandas as pd\nfrom io import StringIO\nprint(pd.read_csv(StringIO("a\\n1\\n")))',
                     'import numpy as np\nfrom io import BytesIO\nb=BytesIO()\nnp.save(b,np.array([1]))',
                     'class Item:\n def save(self): return 1\nprint(Item().save())',
                     'import matplotlib.pyplot as plt\nplt.plot([1,2])\nplt.savefig("/tmp/plot.png")',
                     'with open("/tmp/value", "w") as f: f.write("1")']:
            self.assertEqual(code_checks(code), [], code)
        self.assertTrue(code_checks('open("/tmp/../../etc/passwd").read()'))

    def test_delegate_runtime_constraints(self):
        check=lambda args:validate_args('delegate_task',args,self.contract)
        self.assertEqual(check({'goal':'Find evidence','toolsets':['web']}),[])
        self.assertTrue(check({'goal':'x','toolsets':['terminal']}))
        self.assertTrue(check({'goal':'x','toolsets':['code_execution']}))
        self.assertTrue(check({'goal':'x','role':'orchestrator'}))
        self.assertTrue(check({'tasks':[{'goal':'x'}]*4}))
        self.assertEqual(check({'tasks':[{'goal':'x'}],'background':True}),[])
        self.assertTrue(check({'tasks':[{'goal':'x'}]*2,'background':True}))
        self.assertTrue(check({}))

    def test_vision_assets_and_schema(self):
        check=lambda args:validate_args('vision_analyze',args,self.contract)
        self.assertEqual(check({'image_url':'https://example.com/a.png','question':'Describe'}),[])
        self.assertTrue(check({'image_url':'/tmp/a.png','question':'Describe'}))
        self.assertTrue(check({'image_url':'https://example.com/a.png','question':123}))
        self.assertTrue(validate_args('execute_code',{'code':'print(1)','timeout':9},self.contract))

    def test_typed_decoding(self):
        args=decode_typed({'tasks':'[{"goal":"x","toolsets":"[\\"web\\"]"}]','background':'false'},self.contract['schemas']['delegate_task']['parameters'])
        self.assertEqual(args,{'tasks':[{'goal':'x','toolsets':['web']}],'background':False})
        self.assertEqual(decode_typed('123',{'type':'string'}),'123')

    def item(self, calls):
        import json
        text='\n'.join('<tool_call>'+json.dumps(c)+'</tool_call>' for c in calls)
        return normalize({'question':'Question','messages':[{'role':'assistant','content':text},{'role':'tool','content':'Original evidence'},{'role':'user','content':'Please include units'},{'role':'assistant','content':'Answer'}]},'test','input',0)

    def test_ambiguous_multiple_calls_do_not_invent_response_associations(self):
        item=self.item([{'name':'search','arguments':{'query':['a','b']}},{'name':'PythonInterpreter','arguments':{'code':'print(2+2)'}}])
        before=copy.deepcopy(item)
        item['harness']=harness_plan(item,self.config)
        self.assertIn('missing_observation',item['harness']['errors'])
        self.assertIn('ambiguous_tool_response', item['program_errors'])
        self.assertEqual(item['unassigned_responses'][0]['content'], 'Original evidence')
        self.assertEqual(item['context_messages'],before['context_messages'])
        self.assertEqual([s['observation'] for s in item['steps']],[s['observation'] for s in before['steps']])

    def test_disabled_tools_and_network_code_are_held(self):
        for call in [{'name':'terminal','arguments':{'command':'pwd'}},{'name':'PythonInterpreter','arguments':{'code':'import requests'}}]:
            self.assertTrue(harness_plan(self.item([call]),self.config)['errors'])

    def test_extract_goals_survive_mapping_materialization_and_url_batches(self):
        for name,key in [('visit','goal'),('visit_urls','query'),('web_extract','goal')]:
            goal='Find dates, units & conflicting evidence <source>; 中文'
            item=self.item([{'name':name,'arguments':{'urls':['https://example.org/'+str(i) for i in range(5)],key:goal}}])
            original_observation=item['steps'][0]['observation']
            item['harness']=harness_plan(item,self.config)
            self.assertEqual(item['harness']['errors'],[])
            self.assertNotIn('extraction_semantics_require_replay',item['harness']['flags'])
            self.assertIn('goal_conditioned_extraction_requires_replay',item['harness']['flags'])
            materialize(item)
            self.assertEqual([c['arguments']['goal'] for c in item['steps'][0]['calls']],[goal])
            self.assertEqual(item['steps'][0]['observation'],original_observation)
            self.assertEqual(normalize_text(item['steps'][0]['assistant'])[1],item['steps'][0]['calls'])

    def test_extract_goal_conflicts_invalid_types_and_other_parameters(self):
        for extra,reason in [({'goal':['a']},'invalid_extract_goal'),({'query':None},'invalid_extract_goal'),
                             ({'goal':'a','query':'b'},'conflicting_extract_goals')]:
            plan=harness_plan(self.item([{'name':'visit','arguments':{'url':'https://example.org',**extra}}]),self.config)
            self.assertIn(reason,plan['errors'])
        plan=harness_plan(self.item([{'name':'visit','arguments':{'url':'https://example.org','goal':'a','query':'a','parse_type':'pdf'}}]),self.config)
        self.assertIn('unmapped_extract_argument:parse_type',plan['errors'])
        self.assertEqual(plan['calls'],[])
        plan=harness_plan(self.item([{'name':'web_extract','arguments':{'urls':['https://example.org']}}]),self.config)
        self.assertEqual(plan['errors'],[])
        self.assertNotIn('goal',plan['calls'][0]['arguments'])
        self.assertNotIn('goal_conditioned_extraction_requires_replay',plan['flags'])

    def test_unmapped_arguments_are_held_even_when_empty(self):
        for name,base,key in [('search',{'query':'evidence'},'source'),
                              ('visit',{'url':'https://example.org'},'parse_type'),
                              ('browser.open',{'id':'https://example.org'},'num_lines')]:
            for value in [None,'',False,0,'custom']:
                with self.subTest(name=name,value=value):
                    item=self.item([{'name':name,'arguments':{**base,key:value}}])
                    item['harness']=harness_plan(item,self.config);before=copy.deepcopy(item)
                    self.assertTrue(item['harness']['errors'])
                    self.assertEqual(item['harness']['calls'],[])
                    with self.assertRaisesRegex(ValueError,'held trajectory'):materialize(item)
                    self.assertEqual(item,before)

    def test_browser_direct_url_materialization_preserves_evidence_and_goal(self):
        args={'id':'https://example.org/path?a=1&b=2','goal':'Read <source> &amp; dates'}
        item=self.item([{'name':'browser.open','arguments':args}]);before=copy.deepcopy(item)
        item['harness']=harness_plan(item,self.config)
        self.assertEqual(item['harness']['errors'],[])
        materialize(item)
        self.assertEqual(item['steps'][0]['calls'],[{'name':'web_extract','arguments':{'urls':[args['id']],'goal':args['goal']}}])
        self.assertEqual(item['steps'][0]['observation'],before['steps'][0]['observation'])
        self.assertEqual(normalize_text(item['steps'][0]['assistant'])[1],item['steps'][0]['calls'])

    def test_browser_conflicting_urls_and_viewport_options_are_held(self):
        for extra in [{'url':'https://other.org'},{'urls':['https://other.org']},
                      {'cursor':1},{'line_start':2},{'line_end':5},{'session':'abc'},{'source':'web'}]:
            item=self.item([{'name':'browser.open','arguments':{'id':'https://example.org',**extra}}])
            plan=harness_plan(item,self.config)
            self.assertTrue(plan['errors']);self.assertEqual(plan['calls'],[])

if __name__=='__main__':unittest.main()
