import copy
import unittest
import run_v3 as v

class V3Tests(unittest.TestCase):
    def item(self):
        return {'id':'x','source':'test','question':'What is the single year?', 'reference_answer':'1934',
                'adapter_flags':[],'program_errors':[],
                'steps':[{'step_id':1,'assistant':'Search','observation':'The event took place in 1934. The partitions began in 1772 and ended in 1795.'},
                         {'step_id':2,'assistant':'Check again','observation':'Some other result.'},
                         {'step_id':3,'assistant':'<think>private</think><answer>1934</answer>','observation':''}]}

    def test_output_year_requires_no_external_evidence(self):
        inv={'format_requirements':[{'id':'f1','type':'single_year','text':'single year','question_quote':'single year'}]}
        self.assertTrue(v.format_checks(self.item(),inv)[0]['passed'])

    def test_relocate_wrong_step(self):
        found=v.locate_across(self.item(),{'step_id':2,'quote':'The event took place in 1934.'})
        self.assertEqual(found['step_id'],1);self.assertTrue(found['relocated']);self.assertEqual(found['requested_step_id'],2)

    def test_no_assistant_as_evidence(self):
        self.assertIsNone(v.locate_across(self.item(),{'step_id':3,'quote':'private'}))

    def test_arithmetic_supported(self):
        p={'op':'subtract','left':1795,'right':1772,'result':23}
        e=[{'quote':'The partitions began in 1772 and ended in 1795.'}]
        self.assertTrue(v.check_derivation(p,e,'The interval spans 23 years')['valid'])
        p['result']=24
        self.assertFalse(v.check_derivation(p,e,'The interval spans 24 years')['valid'])

    def test_calendar_interval_partitions_and_boundaries(self):
        # Cross-domain numeric coverage, not the four pilot answers.
        for century in (2,7,13,26,49):
            for count in (1,2,4,5,10,20,25,50,100):
                for index in range(1,count+1):
                    start=(century-1)*100+1+(index-1)*(100//count)
                    end=start+100//count-1
                    for year in (start-1,start,end,end+1):
                        expected=start<=year<=end
                        p={'op':'calendar_interval','left':year,'right':century,
                           'partition_count':count,'partition_index':index,'result':expected}
                        result=v.check_derivation(p,[{'quote':f'Observed year {year}.'}],f'In part {index}/{count} of century {century}')
                        self.assertTrue(result['valid'],(p,result))
                        self.assertEqual(result['result'],expected)

    def test_bad_calendar_parameters_and_unsupported_proof(self):
        proof={'op':'calendar_interval','left':1234,'right':13,'result':True}
        evidence=[{'quote':'Dated 1234.'}]
        for bad in ({'partition_count':3},{'partition_count':2,'partition_index':0},
                    {'right':14},{'op':'guess_from_memory'}):
            self.assertFalse(v.check_derivation({**proof,**bad},evidence,'During century 13')['valid'])

    def test_number_normalization_has_no_selected_centuries(self):
        for text,number in [('seventh',7),('twenty-third',23),('forty ninth',49),('ninety-ninth',99),('-12',-12)]:
            self.assertEqual(v.numbers(text),{number})
        self.assertEqual(v.numbers('2.75'),set())

    def test_arithmetic_is_not_tied_to_example_years(self):
        for left in (7,42,811,2345):
            for right in (2,13,81):
                for op,result in [('add',left+right),('subtract',left-right)]:
                    proof={'op':op,'left':left,'right':right,'result':result}
                    evidence=[{'quote':f'Values {left} and {right}.'}]
                    self.assertTrue(v.check_derivation(proof,evidence,f'Result {result}')['valid'])
                    self.assertFalse(v.check_derivation({**proof,'result':result+1},evidence,f'Result {result+1}')['valid'])

    def test_invented_operand_not_allowed(self):
        self.assertFalse(v.check_derivation({'op':'subtract','left':1795,'right':1772,'result':23},[{'quote':'Only 1795 known.'}],'23 years')['valid'])

    def test_blind_reviewer_gate_no_contradiction_from_context(self):
        audit={'items':[{'id':'a1','evidence':[{'evidence_id':'e1','quote':'Entity X has attribute A.'}],'checked_derivation':None}]}
        review={'items':[{'id':'a1','verdict':'contradicted','edges':[{'evidence_id':'e1','relation':'context','reason':'Attribute A says nothing about attribute B.'}],'requires_computation':False,'derivation_relevant':False,'reason':'Memory is not evidence'}]}
        self.assertEqual(v.validate_review(review,audit)['items'][0]['verdict'],'insufficient')

    def test_duplicate_identical_support_edges_collapsed(self):
        audit={'items':[{'id':'a1','evidence':[{'evidence_id':'e1','quote':'1991'}],'checked_derivation':None}]}
        edge={'evidence_id':'e1','relation':'supports','reason':'source'}
        review={'items':[{'id':'a1','verdict':'supported','edges':[dict(edge),dict(edge)],'requires_computation':False,'derivation_relevant':False}]}
        checked=v.validate_review(review,audit)['items'][0]
        self.assertEqual(len(checked['edges']),1);self.assertEqual(checked['duplicate_edges_collapsed'],1)
        edge2={**edge,'relation':'contradicts'}
        review={'items':[{'id':'a1','verdict':'conflict','edges':[edge,edge2],'requires_computation':False,'derivation_relevant':False}]}
        with self.assertRaises(ValueError):v.validate_review(review,audit)

    def test_computation_cannot_be_silently_replaced_with_model_memory(self):
        audit={'items':[{'id':'a1','evidence':[{'evidence_id':'e1','quote':'Value 64.'}],'checked_derivation':None}]}
        review={'items':[{'id':'a1','verdict':'supported','edges':[{'evidence_id':'e1','relation':'supports','reason':'Computed result'}],
                         'requires_computation':True,'derivation_relevant':False}]}
        checked=v.validate_review(review,audit)['items'][0]
        self.assertEqual(checked['verdict'],'insufficient')
        self.assertEqual(checked['annotation_issue'],'missing_valid_computation_proof')

    def test_conflict_requires_opposing_evidence(self):
        audit={'items':[{'id':'a1','evidence':[{'evidence_id':'e1','quote':'2019'}],'checked_derivation':None}]}
        review={'items':[{'id':'a1','verdict':'conflict','edges':[{'evidence_id':'e1','relation':'contradicts','reason':'different date'}],'requires_computation':False,'derivation_relevant':False}]}
        self.assertEqual(v.validate_review(review,audit)['items'][0]['verdict'],'insufficient')

    def test_earlier_quote_becomes_repeat_not_new_reward(self):
        b={'steps':[{'step_id':2,'events':[{'tag':'discover_supported','target_id':'q1','quote':'The event took place in 1934.','reason':'evidence'}]}]}
        fixed,notes,issues=v.repair_behaviors(self.item(),{'items':[]},b)
        self.assertEqual(fixed['steps'][0]['events'][0]['tag'],'repeat');self.assertTrue(notes);self.assertFalse(issues)

    def test_future_quote_never_supports_earlier_step(self):
        b={'steps':[{'step_id':1,'events':[{'tag':'discover_supported','target_id':'q1','quote':'Some other result.','reason':'evidence'}]}]}
        fixed,notes,issues=v.repair_behaviors(self.item(),{'items':[]},b)
        self.assertFalse(fixed['steps'][0]['events']);self.assertTrue(issues)

    def test_open_ended_task_does_not_require_reference_answer(self):
        item={'id':'open','source':'synthetic','question':'Describe X','reference_answer':'',
              'program_errors':[],'adapter_flags':[],
              'steps':[{'step_id':1,'assistant':'X is an object.','observation':'X is an object.'}]}
        inv={'constraints':[{'id':'q1','text':'Describe X'}], 'claims':[{'id':'a1','text':'X is an object'}], 'format_requirements':[]}
        entries=[{'id':ident,'status':'supported','reason':'located source', 'evidence':[],
                  'invalid_evidence':[],'checked_derivation':None} for ident in ('q1','a1')]
        audit={'items':entries,'final_matches_reference':None}
        review={'items':[{'id':ident,'verdict':'supported','reason':'source supports'} for ident in ('q1','a1')]}
        result=v.compute_result(item,inv,audit,review,{'steps':[{'step_id':1,'events':[]}]})
        self.assertEqual(result['reference_comparison'],'not_available')
        self.assertFalse(result['evidence_gaps']);self.assertTrue(result['offline_quality_candidate'])
        self.assertIsNone(result['audit']['final_matches_reference'])

    def test_insufficient_does_not_mean_false_penalty(self):
        b={'steps':[{'step_id':3,'events':[{'tag':'unsupported_assertion','target_id':'a1','quote':'1934','reason':'not proven'}]}]}
        fixed,notes,issues=v.repair_behaviors(self.item(),{'items':[{'id':'a1','status':'insufficient'}]},b)
        self.assertFalse(fixed['steps'][0]['events']);self.assertTrue(notes)

if __name__=='__main__':unittest.main()
