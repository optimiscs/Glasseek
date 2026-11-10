import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from glasseek_shared.curation.supervision import build_plan,validate_plan
from glasseek_shared.trajectory_release import payload,digest
from eligibility import assess,observation_matches
from segment_audit import causal_prefix,partition
from pipeline import annotate_one
from client import Client
import claim_audit


def fixture():
    item={'id':'x','source':'fixture','question':'What?', 'program_errors':[], 'harness':{},
          'steps':[{'step_id':1,'assistant':'search','calls':[{'name':'web_search','arguments':{}}],
                    'observation':'Recorded result','source_message_index':1},
                   {'step_id':2,'assistant':'Answer','calls':[],'observation':'','source_message_index':3}]}
    canonical={'id':'x','source':'fixture','messages':[{'role':'user','content':'What?'},
        {'role':'assistant','content':'search','source_message_index':1},
        {'role':'user','content':'Recorded result'},
        {'role':'assistant','content':'Answer','source_message_index':3}]}
    scanner=SimpleNamespace(scan=lambda row:{'state':'pass_rules','coverage_complete':True})
    return item,canonical,scanner


def test_observation_binding_allows_only_known_envelopes_and_separators():
    responses=[{'content':'A\n  B'},{'content':'C'}]
    assert observation_matches('<tool_response>\nA\n  B\n</tool_response>\n\n<tool_response>C</tool_response>',responses)
    assert not observation_matches('<tool_response>A B</tool_response>\nC',responses)
    assert not observation_matches('A\n  B\nC\nInvented extra evidence',responses)
    merged={'content':'A\n\nB','merged_response_parts':[{'start':0,'end':1,'source_call_id':'a'},
                                                        {'start':3,'end':4,'source_call_id':'b'}]}
    assert observation_matches('<tool_response>A</tool_response>\n<tool_response>B</tool_response>',[merged])
    merged['merged_response_parts'][1]['start']=2
    # Including a separator in a part does not change its actual body.
    assert observation_matches('<tool_response>A</tool_response>\n<tool_response>B</tool_response>',[merged])
    merged['merged_response_parts'][1]['start']=0
    with pytest.raises(ValueError):observation_matches('A\nB',[merged])


def test_safety_precedes_llm_and_cannot_be_masked():
    item,canonical,scanner=fixture()
    scanner.scan=lambda row:{'state':'quarantine','coverage_complete':True}
    client=SimpleNamespace(canonical={'x':canonical},scanner=scanner,policy_hash='p')
    result=annotate_one(item,client)
    assert result['status']=='routed' and result['decision']=='quarantine'
    assert result['training_ready'] is False and 'supervision' not in result


@pytest.mark.parametrize('answer',['...','<answer>...</answer>','<think>unfinished','TODO',''])
def test_placeholder_or_unfinished_is_not_annotation_failure(answer):
    item,canonical,scanner=fixture()
    item['steps'][-1]['assistant']=answer;canonical['messages'][-1]['content']=answer
    assert assess(item,canonical,scanner)['status']=='incomplete'


def test_zero_call_routes_and_summary_with_real_observation_survives():
    item,canonical,scanner=fixture()
    item['question']='RESEARCH STATE SUMMARY: an unverified prior claim'
    assert assess(item,canonical,scanner)['status']=='eligible'
    item['steps']=item['steps'][-1:];canonical['messages']=canonical['messages'][-1:]
    assert assess(item,canonical,scanner)['status']=='evidence_insufficient'
    item['question']='Compute an answer'
    assert assess(item,canonical,scanner)['status']=='manual_review'


def test_causal_prompt_excludes_same_step_hidden_and_future_evidence():
    item,_,_=fixture()
    item['steps'][0]['response_parts']=[{'text':'SECRET CURRENT RETURN'}]
    item['steps'][1]['assistant']='SECRET FUTURE ANSWER'
    item['context_messages']=[{'before_step':2,'content':'SECRET LATER USER'}]
    item['instruction_messages']=[{'before_step':1,'content':'visible now'}]
    text=json.dumps(causal_prefix(item,1))
    assert 'SECRET' not in text and 'Recorded result' not in text and 'visible now' in text


def test_fixed_boundary_keys_cannot_drop_reorder_or_duplicate_steps():
    item,_,_=fixture()
    class FakeClient:
        def call(self,stage,ident,prompt,system,validator,*,schema):
            assert schema['required']==['1','2']
            assert schema['properties']['1']['properties']['start_new_segment']['const'] is True
            return validator({'1':{'start_new_segment':True,'type':'candidate_search','purpose':'Find answer'},
                              '2':{'start_new_segment':False,'type':'answer_synthesis','purpose':'Use result'}})
    assert partition(item,FakeClient())==[{'id':'segment_1','start_step':1,'end_step':2,
                                         'type':'candidate_search','purpose':'Find answer'}]


def test_supervision_keeps_context_and_binds_release_hash():
    _,canonical,_=fixture()
    messages=canonical['messages']; original=copy.deepcopy(messages)
    plan=build_plan(messages,{1:{'decision':'context_only','segment_id':'s1'},3:{'decision':'learn','segment_id':'s2'}})
    assert messages==original
    validate_plan(plan,messages,require_learn=True)
    canonical.update(tools=[],supervision=plan)
    before=digest(payload(canonical))
    plan['messages'][1]['decision']='learn'
    assert digest(payload(canonical))!=before
    messages[-1]['content']='changed'
    with pytest.raises(ValueError,match='content_mismatch'):validate_plan(plan,messages)


def test_nonassistant_and_all_zero_and_pending_supervision_rejected():
    _,canonical,_=fixture();messages=canonical['messages']
    with pytest.raises(ValueError):build_plan(messages,{0:{'decision':'learn','segment_id':'s'}})
    for decision in ('context_only','review'):
        plan=build_plan(messages,{3:{'decision':decision,'segment_id':'s'}})
        with pytest.raises(ValueError):validate_plan(plan,messages,require_learn=True)


def test_local_only_refuses_external_before_transport(tmp_path):
    provider={'url':'https://example.invalid/v1','model':'x'}
    client=Client({'local_only':True,'providers':{'default':provider},'execution':{'workers':1}},tmp_path,'p')
    with pytest.raises(ValueError,match='Local-only'):client.request(provider,'POST','/chat/completions')


def test_program_owned_inventory_ids_and_evidence_schema():
    item,_,_=fixture()
    class FakeClient:
        def call(self,stage,ident,prompt,system,validator,*,schema):
            assert schema['additionalProperties'] is False
            assert 'id' not in schema['properties']['claims']['items']['properties']
            return validator({'constraints':[{'text':'What?','source_span_id':'Q:0'}],
                              'claims':[{'text':'Answer','source_span_id':'A:0'}],'task_requirements':[],'format_requirements':[]})
    inv=claim_audit.inventory(item,FakeClient())
    assert inv['constraints'][0]['id']=='q1' and inv['claims'][0]['id']=='a1'


def test_unknown_evidence_and_unbacked_support_still_rejected():
    import policy
    item,_,_=fixture()
    inv={'constraints':[{'id':'q1','text':'What?'}],'claims':[]}
    with pytest.raises(ValueError,match='unknown evidence'):
        policy.validate_audit({'items':[{'id':'q1','status':'supported','evidence':[{'span_id':'invented'}]}]},item,inv)
    audit={'items':[{'id':'q1','evidence':[],'checked_derivation':None}]}
    reviewed=policy.validate_review({'items':[{'id':'q1','verdict':'supported','edges':[],
        'requires_computation':False,'derivation_relevant':False,'reason':'guess'}]},audit)
    assert reviewed['items'][0]['verdict']=='insufficient'
    assert reviewed['items'][0]['model_reason']=='guess'
    assert reviewed['items'][0]['reason']=='No evidence support edge or valid derivation'
    assert reviewed['items'][0]['decision_origin']=='program_gate'


def test_task_instructions_remain_requirements_not_world_facts():
    import policy
    item,_,_=fixture();item['question']='Identify this lexical unit.'
    raw={'constraints':[],'claims':[{'id':'a1','text':'Answer','source_span_id':'A:0'}],
         'task_requirements':[{'id':'r1','text':'Identify the unit','source_span_id':'Q:0'}],'format_requirements':[]}
    checked=policy.validate_inventory(copy.deepcopy(raw),item)
    assert not checked['constraints'] and checked['task_requirements'][0]['question_quote']==item['question']
    raw['task_requirements'][0]['source_span_id']='A:0'
    with pytest.raises(ValueError):policy.validate_inventory(raw,item)


def test_initial_supported_verdict_without_evidence_is_annotation_error():
    item,_,_=fixture()
    inv={'constraints':[],'claims':[{'id':'a1','text':'Answer'}],'format_requirements':[]}
    class FakeClient:
        def call(self,stage,ident,prompt,system,validator,*,schema):
            return validator({'a1':{'status':'supported','reason':'guess','evidence':[],'derivation':None}})
    with pytest.raises(ValueError,match='no located evidence'):claim_audit.audit(item,inv,FakeClient())


def test_label_multiplicity_and_conversion_collision_are_not_search_redundancy():
    import policy
    item,_,_=fixture();item['adapter_flags']=[]
    inv={'constraints':[],'claims':[{'id':'a1','text':'Answer'}]}
    audit={'items':[{'id':'a1','status':'supported','invalid_evidence':[]}],
           'unresolved_conflicts':[],'final_matches_reference':True}
    first={'tag':'reasonable_exploration','target_id':'a1','reason':'explore','quote':''}
    converted={**first,'proposed_event':{'tag':'discover_supported','target_id':'a1'}}
    behavior={'steps':[{'step_id':1,'events':[first,copy.deepcopy(first),converted]}, {'step_id':2,'events':[]}]}
    result=policy.score_behavior_events(item,inv,audit,behavior)
    assert result['offline_quality_candidate'] and not result['rejected_behavior_events']
    assert len(result['steps'][0]['events'])==1
    assert {m['kind'] for m in result['merged_behavior_annotations']}=={'model_label_multiplicity','program_conversion_collision'}
    assert all(not m['trajectory_redundancy_inferred'] and m['extra_score']==0 for m in result['merged_behavior_annotations'])


def test_merging_never_bypasses_invalid_witness_or_negative_behavior_gate():
    import policy
    item,_,_=fixture();item['adapter_flags']=[]
    inv={'constraints':[],'claims':[{'id':'a1','text':'Answer'}]}
    audit={'items':[{'id':'a1','status':'contradicted','invalid_evidence':[]}],
           'unresolved_conflicts':[],'final_matches_reference':True}
    event={'tag':'unsupported_assertion','target_id':'a1','reason':'contrary evidence','quote':'Answer'}
    behavior={'steps':[{'step_id':1,'events':[]},{'step_id':2,'events':[event,dict(event),{**event,'quote':'invented'}]}]}
    result=policy.score_behavior_events(item,inv,audit,behavior)
    assert not result['offline_quality_candidate']
    assert result['rejected_behavior_events'][0]['cause']=='assertion_not_located'
    assert 'negative_behavior_requires_review' in result['blockers']


def test_semantic_prompt_keeps_full_observation_without_metadata_copies():
    from segment_audit import visible_step
    item,_,_=fixture();step=item['steps'][0]
    step['responses']=[{'content':step['observation']}]
    visible=visible_step(step)
    assert visible['observation']==step['observation'] and visible['assistant']==step['assistant']
    assert 'responses' not in visible
