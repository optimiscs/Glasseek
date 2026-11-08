import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trajectory_review.core import RUBRIC, RUBRIC_HASH, connect, score, compare, submit


class HumanReviewTests(unittest.TestCase):
    def good(self):return {r['id']:r['good'] for r in RUBRIC}
    def evidence(self):return {r['id']:[0] for r in RUBRIC}
    def test_no_default_or_partial_pass(self):
        with self.assertRaises(ValueError):score({}, {}, 2)
        result=score(self.good(),self.evidence(),2)
        self.assertEqual(result['score'],100);self.assertEqual(result['state'],'pass_review');self.assertFalse(result['training_ready'])

    def test_critical_failure_overrides_score_and_unknown_gives_range(self):
        answers=self.good();answers['secret_sent']='yes';answers['secret_exposed']='yes'
        result=score(answers,self.evidence(),2)
        self.assertEqual(result['state'],'reject')
        answers=self.good();answers['support']='unknown'
        result=score(answers,self.evidence(),2)
        self.assertIsNone(result['score']);self.assertEqual((result['score_min'],result['score_max']),(90,100))
        self.assertEqual(result['state'],'review')

    def test_evidence_and_parent_consistency_required(self):
        with self.assertRaises(ValueError):score(self.good(),{},2)
        e=self.evidence();e['support']=[99]
        with self.assertRaises(ValueError):score(self.good(),e,2)
        a=self.good();a['leak_used']='yes'
        with self.assertRaises(ValueError):score(a,self.evidence(),2)

    def test_unknown_is_not_counted_as_agreement(self):
        auto={'answers':{'leak_read':'no','tool_schema':'no'}}
        compared=compare(self.good(),auto)
        self.assertEqual((compared['agree'],compared['disagree'],compared['uncomparable']),(1,1,10))
        self.assertEqual(compared['agreement_rate'],.5)

    def test_hash_binding_idempotence_and_append_only_reviews(self):
        with tempfile.TemporaryDirectory() as d:
            db=connect(Path(d)/'test.sqlite')
            db.execute('INSERT INTO cases VALUES(?,?,?,?,?,?,?)',('case','Synthetic','content-hash','policy-hash',json.dumps({'messages':[{'origin':'tool_response'},{}]}),json.dumps({'answers':self.good()}),'{}'));db.commit()
            request={'case_id':'case','content_hash':'content-hash','policy_hash':'policy-hash','rubric_hash':RUBRIC_HASH,
                     'reviewer':'test-reviewer','submission_id':'test_submission_00001','answers':self.good(),'evidence':self.evidence(),
                     'confirmed_complete':True,'note':'Synthetic only'}
            event=submit(db,request);self.assertEqual(event['result']['score'],100)
            self.assertEqual(event,submit(db,request));self.assertEqual(db.execute('SELECT COUNT(*) FROM reviews').fetchone()[0],1)
            bad=copy.deepcopy(request);bad['score']=100
            with self.assertRaises(ValueError):submit(db,bad)
            for key in ('content_hash','policy_hash','rubric_hash'):
                bad=copy.deepcopy(request);bad[key]='stale'
                with self.assertRaises(ValueError):submit(db,bad)
            other=copy.deepcopy(request);other['submission_id']='test_submission_00002';other['note']='api_key=sk-'+'x'*32
            result=submit(db,other);self.assertNotIn('x'*32,json.dumps(result))
            self.assertEqual(db.execute('SELECT COUNT(*) FROM reviews').fetchone()[0],2);db.close()


if __name__=='__main__':unittest.main()
