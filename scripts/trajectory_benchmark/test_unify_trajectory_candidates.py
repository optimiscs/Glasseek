import copy
import json
from pathlib import Path
import tempfile
import unittest

from unify_trajectory_candidates import build_release, records
from migrate_searchswarm import review_messages


class UnifiedReleaseTest(unittest.TestCase):
    def test_union_dedup_retains_goals_aliases_review_flags_and_parent_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);full=root/'full';migration=root/'migration'
            full.mkdir();migration.mkdir()
            schema={'name':'web_extract','parameters':{'type':'object','properties':{'urls':{'type':'array'},'goal':{'type':'string'}}}}
            contract={'sha256':'fixture','schemas':{'web_extract':schema},'delegation':{'blocked_child_tools':[]}}
            (full/'manifest.json').write_text(json.dumps({'config':{'harness':{'contract':contract}}}))
            (migration/'manifest.json').write_text(json.dumps({'contract':contract}))
            def item(ident,question,group,role='parent',source='SearchSwarm'):
                r={'id':ident,'source':source,'question':question,'question_hash':group,'split_group_id':group,
                   'agent_role':role,'bundle_id':group,'training_ready':False,'harness':{'errors':[],'flags':[]},
                   'provenance':{'path':'fixture','row_index':0},'steps':[{'step_id':1,'assistant':
                   '<tool_call><function=web_extract><parameter=urls>["https://example.org"]</parameter><parameter=goal>Find dates</parameter></function></tool_call>',
                   'observation':'<tool_response>Source evidence</tool_response>'},
                   {'step_id':2,'assistant':'Answer','observation':''}]}
                r['messages']=review_messages(r)
                return r
            main=item('full','Parent A','main-group',source='Glasseek')
            pa=item('parent-a','Parent A','group-a');pb=item('parent-b','Parent B','group-b')
            ca=item('child-a','Shared task','group-a','child');ca['parent_id']='parent-a'
            cb=item('child-b','Shared task','group-b','child');cb['parent_id']='parent-b'
            cb['review_flags']=['extra_review_from_duplicate']
            for path,items in [(full/'clean.jsonl',[main]),(migration/'parent_candidates.jsonl',[pa,pb]),(migration/'child_candidates.jsonl',[ca,cb])]:
                path.write_text(''.join(json.dumps(r)+'\n' for r in items))
            edges=[{'parent_id':'parent-a','child_id':'child-a','report_matches_child_final':True,'split_group_id':'group-a'},
                   {'parent_id':'parent-b','child_id':'child-b','report_matches_child_final':False,'split_group_id':'group-b'}]
            (migration/'links.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in edges))
            out=root/'unified';result=build_release(full,migration,out)
            self.assertEqual(result['input_candidates'],5)
            self.assertEqual(result['candidates'],3)
            self.assertEqual(result['union_exact_duplicates_removed'],2)
            merged=list(records(out/'candidates.jsonl'))
            self.assertEqual(len({r['split_group_id'] for r in merged}),1)
            child=next(r for r in merged if r['agent_role']=='child')
            self.assertEqual(child['canonical_parent_id'],'full')
            self.assertIn('linked_report_final_mismatch',child['review_flags'])
            self.assertIn('extra_review_from_duplicate',child['review_flags'])
            self.assertTrue(all('Find dates' in json.dumps(r['messages']) for r in merged))
            self.assertEqual(len(list(records(out/'origins.jsonl'))),5)
            links=list(records(out/'delegation_links.jsonl'))
            self.assertEqual(links[0]['canonical_parent_id'],'full')
            self.assertEqual(links[1]['canonical_child_id'],'child-a')
            self.assertTrue(all(r['both_endpoints_are_candidates'] for r in links))


if __name__=='__main__':unittest.main()
