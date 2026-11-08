import unittest
from pathlib import Path
from migrate_searchswarm import parse_reports,target_tasks,convert_parent,associate,normalize_target,review_messages,child_prompt_builder
from glasseek_contract import build_contract
from tool_xml import serialize

class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.contract=build_contract()
    def fixture(self,n=2):
        tasks=[{'goal':'Goal '+str(i),'prompt':'Full task '+str(i)} for i in range(n)]
        text='<tool_response>\n'+'\n\n---\n\n'.join('A sub-agent dispatched for goal "'+t['goal']+'" returned the following report:\n\nReport '+str(i)+' https://example.org/'+str(i) for i,t in enumerate(tasks))+'\n</tool_response>'
        row={'question':'Parent question','messages':[{'role':'user','content':'Parent question'},
            {'role':'assistant','content':'<think>Parent thinking</think>'+serialize({'name':'call_sub_agent','arguments':{'prompts':tasks}})},
            {'role':'user','content':text},{'role':'assistant','content':'Final'}],
             'subagents':[{'question':t['prompt'],'messages':[{'role':'user','content':t['prompt']},{'role':'assistant','content':'<think>Child private thinking</think>Report '+str(i)+' https://example.org/'+str(i)}]} for i,t in enumerate(tasks)]}
        return row
    def test_fields_reports_and_child_isolation(self):
        row=self.fixture();bid,m,ev,err=convert_parent(row,0,Path('/source'),self.contract)
        self.assertEqual(err,[]);self.assertEqual(ev[0]['tasks'][0],{'goal':'Goal 0','context':'Full task 0'})
        self.assertNotIn('Child private thinking',str(m));self.assertIn('Parent thinking',m[1]['content'])
        self.assertNotIn('duration_seconds',m[2]['content']);self.assertNotIn('"status"',m[2]['content'])
        edges,_=associate(row['subagents'],ev,bid)
        self.assertTrue(all(e['child_id'] and e['report_matches_child_final'] for e in edges))
        item=normalize_target(m,row['question'],'test',0,'/source',bid,bid,'parent',self.contract)
        self.assertEqual(item['migration_errors'],[])
        roles=review_messages(item)
        self.assertEqual([m['role'] for m in roles],['user','assistant','user','assistant'])
    def test_cap_is_not_silently_split(self):
        _,m,ev,_=convert_parent(self.fixture(4),0,Path('/source'),self.contract)
        self.assertEqual(len(ev),1);self.assertEqual(len(ev[0]['tasks']),4)
        self.assertIn('tool_runtime:delegate_task.batch_limit',ev[0]['runtime_issues'])
    def test_duplicate_labels_fail_closed(self):
        row=self.fixture();row['messages'][2]['content']=row['messages'][2]['content'].replace('Goal 1','Goal 0')
        _,_,_,err=convert_parent(row,0,Path('/source'),self.contract)
        self.assertTrue(err)
    def test_missing_child_keeps_historical_report(self):
        row=self.fixture();bid,m,ev,_=convert_parent(row,0,Path('/source'),self.contract)
        edges,_=associate([],ev,bid)
        self.assertTrue(all(e['child_id'] is None for e in edges));self.assertIn('Report 0',m[2]['content'])
    def test_ambiguous_prompt_requires_report_disambiguation(self):
        row=self.fixture(1);bid,_,ev,_=convert_parent(row,0,Path('/source'),self.contract)
        child=dict(row['subagents'][0]);child['messages']=[{'role':'assistant','content':'Different report'}]
        edges,_=associate([child,*row['subagents']],ev,bid)
        self.assertEqual(edges[0]['child_index'],1)
    def test_child_compute_is_blocked(self):
        ms=[{'role':'assistant','content':serialize({'name':'execute_code','arguments':{'code':'print(2)'}})},
            {'role':'user','content':'<tool_response>2</tool_response>'},{'role':'assistant','content':'2'}]
        item=normalize_target(ms,'q','test',0,'/source','x','b','child',self.contract)
        self.assertIn('child_tool_blocked:execute_code',item['migration_errors'])
    def test_real_child_prompt_placement(self):
        text=child_prompt_builder()('short goal','full prompt',max_spawn_depth=1,child_depth=1,timeout_seconds=900)
        self.assertIn('YOUR TASK:\nshort goal',text);self.assertIn('CONTEXT:\nfull prompt',text)

    def test_report_wrapper_and_trailing_reminder(self):
        row=self.fixture(1)
        row['messages'][2]['content']+='\nPlease give your final answer now.'
        row['subagents'][0]['messages'][-1]['content']='<report>Report 0 https://example.org/0</report>'
        bid,m,ev,err=convert_parent(row,0,Path('/source'),self.contract)
        self.assertFalse(err);self.assertTrue(m[2]['content'].endswith('Please give your final answer now.'))
        edges,_=associate(row['subagents'],ev,bid)
        self.assertTrue(edges[0]['report_matches_child_final'])

if __name__=='__main__':unittest.main()
