import copy
import unittest

import policy as p


class EvidenceIDPolicyTests(unittest.TestCase):
    def setUp(self):
        self.item={'id':'synthetic','source':'test','question':'Which year was Registry A established?',
                   'reference_answer':'1907','program_errors':[],'adapter_flags':[],
                   'steps':[{'step_id':1,'assistant':'Inspect the register.','calls':[],
                             'observation':'The [register](https://example.invalid/a) states: “Registry A was established in 1907.”'},
                            {'step_id':2,'assistant':'Registry A was established in 1907.','calls':[],'observation':''}]}
        sources=p.inventory_sources(self.item)
        self.inv=p.validate_inventory({'format_requirements':[{'id':'f1','type':'single_year','requirement_kind':'answer_type','text':'year','source_span_id':sources['question'][0]['span_id']}],
                     'constraints':[{'id':'q1','text':'Registry A establishment year is 1907','source_span_id':sources['question'][0]['span_id']}],
                     'claims':[{'id':'a1','text':'Registry A was established in 1907','source_span_id':sources['answer'][0]['span_id']}]},self.item)
        self.span=p.evidence_catalog(self.item)[0]['span_id']
        self.audit=p.validate_audit({'final_matches_reference':None,'items':[
            {'id':k,'status':'supported','reason':'register','evidence':[{'span_id':self.span}],'derivation':None} for k in ('q1','a1')]},self.item,self.inv)
        self.review=p.validate_review({'items':[{'id':k,'verdict':'supported','reason':'register directly states it',
                    'requires_computation':False,'derivation_relevant':False,'edges':[{'evidence_id':k+':e1','relation':'supports','reason':'direct statement','same_scope':True,'scope_reason':'same register and event'}]} for k in ('q1','a1')]},self.audit)
        self.behavior={'steps':[{'step_id':1,'events':[]},{'step_id':2,'events':[]}]}

    def test_program_restores_exact_markdown_and_unicode(self):
        quote=self.audit['items'][0]['evidence'][0]['quote']
        self.assertEqual(quote,self.item['steps'][0]['observation'])
        self.assertFalse(self.audit['items'][0]['invalid_evidence'])

    def test_later_user_requirements_reach_inventory_with_chronology(self):
        item=copy.deepcopy(self.item)
        item['context_messages']=[{'role':'user','content':'Return only the year.', 'before_step':2}]
        sources=p.inventory_sources(item)
        self.assertIn('before assistant step 2',sources['question'][0]['text'])
        self.assertIn('Return only the year.',p.inventory_prompt(item))
        checked=p.validate_inventory(copy.deepcopy(self.inv),item)
        self.assertIn('Return only the year.',checked['constraints'][0]['question_quote'])
        self.assertFalse(any('Return only the year.' in row['text'] for row in p.evidence_catalog(item)))

    def test_unknown_evidence_never_becomes_support(self):
        with self.assertRaisesRegex(ValueError,'unknown evidence'):
            p.validate_audit({'items':[{'evidence':[{'span_id':'invented'}]}]},self.item,self.inv)

    def test_inventory_cannot_cite_answer_as_question(self):
        inv=copy.deepcopy(self.inv);inv['constraints'][0]['source_span_id']=p.inventory_sources(self.item)['answer'][0]['span_id']
        with self.assertRaises(ValueError):p.validate_inventory(inv,self.item)

    def test_wrong_scope_is_not_a_contradiction(self):
        review=copy.deepcopy(self.review)
        for e in review['items']:
            e['verdict']='contradicted';e['edges'][0].update(relation='contradicts',same_scope=False,scope_reason='different entity')
        review=p.validate_review(review,self.audit)
        self.assertTrue(all(e['verdict']=='insufficient' for e in review['items']))

    def test_answer_type_does_not_forbid_explanation(self):
        result=p.compute_result(self.item,self.inv,self.audit,self.review,self.behavior)
        self.assertTrue(result['offline_quality_candidate'])
        self.assertTrue(result['format_checks'][0]['passed'])

    def test_explicit_presentation_still_enforced(self):
        inv=copy.deepcopy(self.inv);inv['format_requirements'][0]['requirement_kind']='presentation'
        result=p.compute_result(self.item,inv,self.audit,self.review,self.behavior)
        self.assertFalse(result['offline_quality_candidate'])
        self.assertEqual(result['quality_issues'][0]['type'],'format_failure')

    def test_optional_invalid_arithmetic_does_not_override_direct_evidence(self):
        audit=copy.deepcopy(self.audit)
        audit['items'][0]['checked_derivation']={'valid':False,'reason':'unused bad proof'}
        result=p.compute_result(self.item,self.inv,audit,self.review,self.behavior)
        self.assertTrue(result['offline_quality_candidate'])
        self.assertTrue(any(r['type']=='unused_invalid_proof_removed_from_directly_supported_fact' for r in result['repaired_annotations']))

    def test_negative_behavior_is_not_lost_in_aggregate(self):
        item=copy.deepcopy(self.item);item['steps'][0]['assistant']='Reject the register statement that Registry A was established in 1907.'
        behavior=copy.deepcopy(self.behavior)
        behavior['steps'][0]['events']=[{'tag':'reject_supported','target_id':'a1','quote':item['steps'][0]['assistant'],'reason':'explicit rejection of a supported fact'}]
        result=p.compute_result(item,self.inv,self.audit,self.review,behavior)
        self.assertIn('negative_behavior_requires_review',result['blockers'])
        self.assertFalse(result['offline_quality_candidate'])

    def test_positive_credit_requires_target_evidence(self):
        behavior=copy.deepcopy(self.behavior)
        behavior['steps'][0]['events']=[{'tag':'discover_supported','target_id':'not_in_audit','span_id':self.span,'reason':'bad target'}]
        result=p.validate_behaviors(behavior,self.item,p.reviewed_audit(self.audit,self.review))
        self.assertEqual(result['steps'][0]['events'][0]['tag'],'reasonable_exploration')
        self.assertEqual(result['steps'][0]['events'][0]['quote'],'')

    def test_string_span_id_is_same_as_object_reference(self):
        value={'final_matches_reference':None,'items':[{'id':k,'status':'supported','reason':'source','evidence':[self.span],'derivation':None} for k in ('q1','a1')]}
        audit=p.validate_audit(value,self.item,self.inv)
        self.assertEqual(audit['items'][0]['evidence'][0]['quote'],self.item['steps'][0]['observation'])

    def test_unlocated_exclusion_receives_no_positive_credit(self):
        behavior=copy.deepcopy(self.behavior)
        behavior['steps'][0]['events']=[{'tag':'exclude_wrong','target_id':'candidate:X','span_id':'B1:0','reason':'unverified label'}]
        result=p.validate_behaviors(behavior,self.item,p.reviewed_audit(self.audit,self.review))
        self.assertEqual(result['steps'][0]['events'][0]['tag'],'reasonable_exploration')

    def test_invalid_negative_annotation_still_blocks(self):
        behavior=copy.deepcopy(self.behavior)
        behavior['steps'][0]['events']=[{'tag':'reject_supported','target_id':'a1','span_id':'B2:0','reason':'wrong step'}]
        with self.assertRaisesRegex(ValueError,'same assistant step'):
            p.validate_behaviors(behavior,self.item,p.reviewed_audit(self.audit,self.review))


if __name__=='__main__':unittest.main()
