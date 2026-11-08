import copy
import json
from pathlib import Path
import unittest
import run_v2 as v

class PipelineTests(unittest.TestCase):
    def fixture(self):
        item={'id':'test','source':'test','program_errors':[],'adapter_flags':['legacy'],
              'steps':[{'step_id':1,'observation':'The event occurred in 1991.','assistant':'Checking the year.'},
                       {'step_id':2,'observation':'The event occurred in 1991.','assistant':'Confirmed.'},
                       {'step_id':3,'observation':'','assistant':'The answer is 1991.'}]}
        inventory={'constraints':[{'id':'q1'}],'claims':[{'id':'a1'}]}
        audit={'items':[{'id':i,'status':'supported','invalid_evidence':[]} for i in ['q1','a1']],
               'final_matches_reference':True,'unresolved_conflicts':[]}
        behavior={'steps':[{'step_id':i,'events':[]} for i in [1,2,3]]}
        return item,inventory,audit,behavior

    def test_decoded_evidence_quotes_preserve_escapes(self):
        passages=['The event occurred in 1991.\nA second sentence.', 'A quoted "title" with 中文 text.\nEnd.']
        raw='Evidence in page: '+repr(passages)+'\n\nSummary: unrelated summary'
        for passage in passages:
            found=v.locate_quote(raw,passage)
            self.assertIsNotNone(found)
            self.assertTrue(found['source'].startswith('evidence_list['))

    def test_no_semantic_fuzzy_match(self):
        self.assertIsNone(v.locate_quote('The year is 1991.','The year is 1992.'))
        self.assertIsNone(v.locate_quote('He did not sign.','He did sign.'))
        self.assertIsNone(v.locate_quote('Evidence in page: ["alpha", "beta"]\n\nSummary: none','alpha beta'))
        self.assertIsNone(v.locate_quote('text',''))

    def test_invalid_quote_downgrades_support(self):
        item,inv,audit,b=self.fixture()
        for e in audit['items']: e['evidence']=[{'step_id':1,'quote':'invented'}]
        validated=v.validate_audit(audit,item,inv)
        self.assertTrue(all(e['status']=='insufficient' for e in validated['items']))

    def test_deterministic_score_and_no_repeat_bonus(self):
        item,inv,audit,b=self.fixture()
        for s in b['steps'][:2]:s['events']=[{'tag':'discover_supported','target_id':'q1','quote':'The event occurred in 1991.','reason':'evidence'}]
        r=v.compute_result(item,inv,audit,b)
        self.assertEqual([s['score'] for s in r['steps']],[1.8,1.0,2.0])
        self.assertTrue(r['offline_quality_candidate'])
        self.assertFalse(r['training_ready'])

    def test_conflict_blocks_candidate_and_final_bonus(self):
        item,inv,audit,b=self.fixture()
        audit['unresolved_conflicts']=[{'item_ids':['q1'],'reason':'two dates'}]
        r=v.compute_result(item,inv,audit,b)
        self.assertFalse(r['offline_quality_candidate']); self.assertEqual(r['steps'][-1]['score'],1)

    def test_unsupported_final_claim_blocks(self):
        item,inv,audit,b=self.fixture();audit['items'][1]['status']='insufficient'
        r=v.compute_result(item,inv,audit,b)
        self.assertIn('claim:a1:insufficient',r['blockers']);self.assertEqual(r['steps'][-1]['score'],1)

    def test_unlocated_behavior_cannot_gain_points(self):
        item,inv,audit,b=self.fixture()
        b['steps'][0]['events']=[{'tag':'discover_supported','target_id':'q1','quote':'unknown text','reason':'claim'}]
        r=v.compute_result(item,inv,audit,b)
        self.assertEqual(r['steps'][0]['score'],1); self.assertTrue(r['rejected_behavior_events'])

    def test_missing_steps_and_numeric_scores_rejected(self):
        item,inv,audit,b=self.fixture()
        with self.assertRaises(ValueError):v.validate_behaviors({'steps':[]},item)
        b['steps'][0]['score']=1.8
        with self.assertRaises(ValueError):v.validate_behaviors(b,item)

if __name__=='__main__':unittest.main()
