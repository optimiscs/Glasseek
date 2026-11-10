import copy
from normalize_training_text import normalize
from glasseek_shared.trajectory_release import payload,digest


def fixture():
 return {'id':'x','source':'test','tools':[], 'messages':[
  {'role':'system','content':'Keep functions.search here'},
  {'role':'user','content':'Question'},
  {'role':'assistant','content':'<think>Use functions.search now.</think>\n<tool_call>\n<function=web_search>\n<parameter=query>\n["q"]\n</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>', 'call_links':[{'call_id':'c','name':'web_search','arguments':{'query':['q'],'limit':5}}]},
  {'role':'user','content':'<tool_response>\nTool: search\nResult: functions.search source quote\n</tool_response>','response_to':'c','tool_call_id':'c'},
  {'role':'assistant','content':'<think>Done</think>\n<answer>The answer</answer>'}], 'training_ready':False,'review_flags':[]}


def test_normalize_preserves_evidence_and_is_idempotent():
 r=fixture();before=copy.deepcopy(r);n=normalize(r)
 assert r==before
 assert n['messages'][0]==r['messages'][0]
 assert '<think>Use web_search now.</think>' in n['messages'][2]['content']
 assert n['messages'][2]['call_links']==r['messages'][2]['call_links']
 assert 'Tool: web_search\nResult: functions.search source quote' in n['messages'][3]['content']
 assert n['messages'][-1]['content']==r['messages'][-1]['content']
 assert not n['text_normalization']['holds']
 assert digest(payload(normalize(n)))==digest(payload(n))


def test_state_is_marked_not_held_and_wrapper_is_bounded():
 r=fixture();r['messages'][1]['content']='You are running one browse-comparison evaluation task.\nTask type: complex_search\nQuestion:\nReal question\nRESEARCH STATE SUMMARY (prev_state)\ntrusted claims'
 n=normalize(r)
 assert n['messages'][1]['content']=='Real question\nRESEARCH STATE SUMMARY (prev_state)\ntrusted claims'
 assert n['task_context_kind']=='research_continuation'
 assert 'historical_research_state_requires_provenance' not in n['text_normalization']['holds']


def test_ambiguous_spans_and_associations_are_held():
 r=fixture();r['messages'][2]['content']=r['messages'][2]['content'].replace('Use functions.search now.','Quote `functions.search`')
 r['messages'][3]['response_to']='wrong'
 n=normalize(r)
 assert '`functions.search`' in n['messages'][2]['content']
 assert 'legacy_tool_reference_ambiguous' in n['text_normalization']['holds']
 assert 'response_label_association_mismatch' in n['text_normalization']['holds']


def test_changed_content_invalidates_old_acceptance():
 r=fixture();r.update(safety={'state':'pass_rules'},acceptance={'quality':'passed'},content_sha256='old',payload_sha256='old')
 n=normalize(r)
 assert not n['training_ready'] and not n['harness_replayed']
 assert 'safety' not in n and 'acceptance' not in n and 'content_sha256' not in n
 assert 'normalized_content_requires_revalidation' in n['review_flags']


def test_response_hash_rebound_without_changing_raw_evidence():
 import hashlib
 r=fixture();m=r['messages'][3]
 m['search_return_audit']={'rendered_response_sha256':hashlib.sha256(m['content'].encode()).hexdigest(),'raw_response_sha256':'original'}
 n=normalize(r);a=n['messages'][3]['search_return_audit']
 assert a['raw_response_sha256']=='original'
 assert a['rendered_response_sha256']==hashlib.sha256(n['messages'][3]['content'].encode()).hexdigest()
 assert 'stale_search_return_audit' not in n['text_normalization']['holds']


def test_user_required_answer_format_is_not_held():
 r=fixture();r['messages'][1]['content']='Return your answer in <answer> tags.'
 n=normalize(r)
 assert n['messages'][-1]['content'].count('<answer>')==1
 assert not n['text_normalization']['holds']


def test_explanation_becomes_answer_and_remains_idempotent():
 r=fixture();r['messages'][-1]['content']='<think>Done</think>\n<explanation>Evidence</explanation>\nResult'
 n=normalize(r)
 assert n['messages'][-1]['content']=='<think>Done</think><answer>\nEvidence\nResult</answer>'
 assert digest(payload(normalize(n)))==digest(payload(n))
 assert not n['text_normalization']['holds']


def test_multiple_and_orphan_answer_tags_preserve_body():
 for content in ('<answer>A</answer>\n<explanation>B</explanation>\nC', 'A\nB</answer>', 'plain text'):
  r=fixture();r['messages'][-1]['content']=content
  n=normalize(r);text=n['messages'][-1]['content']
  import re
  assert text.count('<answer>')==text.count('</answer>')==1
  assert text[8:-9]==re.sub(r'</?(?:answer|explanation)>','',content)
  assert not n['text_normalization']['holds']
  assert digest(payload(normalize(n)))==digest(payload(n))


def test_apostrophes_and_sentence_period_are_not_quote_or_namespace():
 r=fixture();r['messages'][2]['content']=r['messages'][2]['content'].replace('Use functions.search now.', "We haven't searched. We have a tool: functions.search. We'll use the functions.search tool.")
 n=normalize(r)
 assert 'functions.search' not in n['messages'][2]['content']
 assert not n['text_normalization']['holds']


def test_final_answer_prefix_removed_only_from_final_body():
 r=fixture();r['messages'][-1]['content']='<think>FINAL ANSWER: plan</think>\nFINAL ANSWER: Result'
 n=normalize(r)
 assert n['messages'][-1]['content']=='<think>FINAL ANSWER: plan</think><answer>\nResult</answer>'
 assert not n['text_normalization']['holds']


def test_required_final_answer_prefix_is_retained_without_format_conflict():
 r=fixture();r['messages'][1]['content']='Use FINAL ANSWER: <answer>.'
 r['messages'][-1]['content']='FINAL ANSWER: The British Invasion'
 n=normalize(r)
 assert n['messages'][-1]['content']=='<answer>FINAL ANSWER: The British Invasion</answer>'
 assert not n['text_normalization']['holds']


def test_empty_think_is_repaired_but_real_missing_boundary_stays_untouched():
 r=fixture();tail=r['messages'][2]['content'].split('</think>')[1]
 r['messages'][2]['content']='<think>'+tail
 n=normalize(r)
 assert n['messages'][2]['content']==tail
 assert not n['text_normalization']['holds']
 r['messages'][2]['content']='<think>We have a tool: functions.search.'+tail
 n=normalize(r)
 assert 'unbalanced_think_boundary' in n['text_normalization']['holds']
 assert n['messages'][2]['content']==r['messages'][2]['content']


def test_empty_think_repair_requires_exact_valid_calls_and_no_other_text():
 for invalid_kind in ('extra_prose','metadata_mismatch','missing_return'):
  r=fixture();r['messages'][2]['content']='<think>'+r['messages'][2]['content'].split('</think>')[1]
  if invalid_kind=='extra_prose':r['messages'][2]['content']+='real reasoning'
  elif invalid_kind=='metadata_mismatch':r['messages'][2]['call_links'][0]['arguments']['limit']=4
  else:r['messages'].pop(3)
  n=normalize(r)
  assert n['messages'][2]['content']==r['messages'][2]['content']
  assert 'unbalanced_think_boundary' in n['text_normalization']['holds']


def test_duplicate_think_removes_only_redundant_opener_before_alias_migration():
 r=fixture();r['messages'][2]['content']='<think>\n'+r['messages'][2]['content']
 n=normalize(r)
 assert n['messages'][2]['content']=='\n'+fixture()['messages'][2]['content'].replace('functions.search','web_search')
 assert not n['text_normalization']['holds']
 assert {'duplicate_think_opening','assistant_tool_reference'} <= {e['rule'] for e in n['text_normalization']['events']}
 r['messages'][2]['content']='<think>Real first thought.\n'+fixture()['messages'][2]['content']
 n=normalize(r)
 assert n['messages'][2]['content']==r['messages'][2]['content']
 assert 'unbalanced_think_boundary' in n['text_normalization']['holds']


def test_exact_duplicate_rebinds_mask_and_plan_without_changing_decisions():
 from glasseek_shared.curation.supervision import build_plan,validate_plan
 r=fixture();r['messages'].insert(2,{'role':'user','content':'Solve the following task: Question'})
 r['message_loss_mask']=[0,0,0,0,0,1]
 r['supervision']=build_plan(r['messages'],{3:{'decision':'context_only'},5:{'decision':'learn','segment_id':'final'}})
 before=copy.deepcopy(r);n=normalize(r)
 assert r==before
 assert len(n['messages'])==5 and n['message_loss_mask']==[0,0,0,0,1]
 assert n['text_normalization']['message_index_map']==[0,1,3,4,5]
 validate_plan(n['supervision'],n['messages'],require_learn=True)
 assert n['supervision']['messages'][2]['decision']=='context_only'
 assert n['supervision']['messages'][4]['decision']=='learn'
 assert n['supervision']['messages'][4]['segment_id']=='final'
 assert n['messages'][2]['call_links']==r['messages'][3]['call_links']
 assert not n['text_normalization']['holds']
 assert digest(payload(normalize(n)))==digest(payload(n))


def test_near_duplicate_and_genuine_followup_are_not_dropped():
 for content in ('Solve the following task: Question.','Question','Solve the following task: Question\nUse citations.'):
  r=fixture();r['messages'].insert(2,{'role':'user','content':content})
  n=normalize(r)
  assert len(n['messages'])==len(r['messages'])
  assert n['messages'][2]['content']==content


def test_stale_or_invalid_supervision_is_not_laundered_by_rebinding():
 from glasseek_shared.curation.supervision import build_plan
 r=fixture();r['supervision']=build_plan(r['messages'],{4:{'decision':'learn','segment_id':'final'}})
 r['messages'][1]['content']='Changed question'
 r['message_loss_mask']=[0,1,1,0,1]  # User supervision is forbidden.
 n=normalize(r)
 assert 'supervision_requires_rebinding' in n['text_normalization']['holds']
 assert 'invalid_message_loss_mask' in n['text_normalization']['holds']
 assert n['supervision']==r['supervision']


def test_prediction_heading_only_removed_when_leading_and_unrequested():
 for request,remove in [('Question',True),('Keep the ### Prediction heading.',False)]:
  r=fixture();r['messages'][1]['content']=request
  r['messages'][-1]['content']='<answer>\n### Prediction\nFirst\n\nSecond\n### Prediction\nLater heading</answer>'
  n=normalize(r)
  expected='<answer>\n'+('' if remove else '### Prediction\n')+'First\n\nSecond\n### Prediction\nLater heading</answer>'
  assert n['messages'][-1]['content']==expected
  assert not n['text_normalization']['holds']


def test_alias_and_final_prefix_do_not_rewrite_quoted_or_code_content():
 r=fixture();body='Use functions.search.\n> functions.visit is a source quote\n```python\nfunctions.search()\n```\n"functions.visit_urls"'
 r['messages'][2]['content']=r['messages'][2]['content'].replace('Use functions.search now.',body)
 r['messages'][-1]['content']='```text\nFINAL ANSWER: literal\n```\n> FINAL ANSWER: quoted\nFINAL ANSWER: Result'
 n=normalize(r)
 assert '<think>Use web_search.' in n['messages'][2]['content']
 assert body.split('\n',1)[1] in n['messages'][2]['content']
 assert '```text\nFINAL ANSWER: literal\n```\n> FINAL ANSWER: quoted\nResult' in n['messages'][-1]['content']


def test_quoted_think_tags_cannot_capture_a_partial_reasoning_segment():
 r=fixture()
 body='Use functions.search. Literal "<think>functions.visit</think>". Then functions.visit_urls.'
 r['messages'][2]['content']=r['messages'][2]['content'].replace('Use functions.search now.',body)
 n=normalize(r)
 expected='Use web_search. Literal "<think>functions.visit</think>". Then web_extract.'
 assert n['messages'][2]['content']==r['messages'][2]['content'].replace(body,expected)
 assert 'legacy_tool_reference_ambiguous' in n['text_normalization']['holds']


def test_child_report_restored_without_parent_answer_or_body_changes():
 for content in ('<think>Done</think>\nReport\nReferences: [1]', '<think>Done</think>\n<answer>Report</answer>\nReferences: [1]'):
  r=fixture();r['agent_role']='child';r['parent_id']='parent';r['messages'][1]['content']='Return <report>...</report> with references.'
  r['messages'][-1]['content']=content
  n=normalize(r);text=n['messages'][-1]['content']
  assert text=='<think>Done</think><report>\nReport\nReferences: [1]</report>'
  assert '<answer>' not in text
  assert not n['text_normalization']['holds']
  assert digest(payload(normalize(n)))==digest(payload(n))
 r=fixture();r['agent_role']='child';r['messages'][-1]['content']='Report without a requested envelope'
 assert normalize(r)['messages'][-1]['content']==r['messages'][-1]['content']


def test_native_citation_without_mapping_is_held_without_guessing_url():
 r=fixture();r['messages'][-1]['content']='<answer>Supported assertion 【8†L1-L4】</answer>'
 n=normalize(r)
 assert n['messages'][-1]['content']==r['messages'][-1]['content']
 assert 'citation_source_mapping_unverified' in n['text_normalization']['holds']
 assert normalize(n)['text_normalization']['holds']==n['text_normalization']['holds']
 r['messages'][-1]['content']='<answer>Literal syntax: `【8†L1-L4】`</answer>'
 assert 'citation_source_mapping_unverified' not in normalize(r)['text_normalization']['holds']


def test_missing_requested_prefix_is_not_fabricated():
 r=fixture();r['messages'][1]['content']='Use FINAL ANSWER: <answer>.'
 n=normalize(r)
 assert 'required_final_answer_prefix_missing' in n['text_normalization']['holds']
 assert 'FINAL ANSWER:' not in n['messages'][-1]['content']


def test_stale_response_hash_is_not_repaired_into_an_accepted_audit():
 r=fixture();r['messages'][3]['search_return_audit']={'rendered_response_sha256':'stale','raw_response_sha256':'original'}
 n=normalize(r)
 assert n['messages'][3]==r['messages'][3]
 assert 'stale_search_return_audit' in n['text_normalization']['holds']


def test_normalization_cannot_clear_an_existing_safety_hold():
 r=fixture();r['safety']={'state':'review_required'};r['review_flags']=['sensitive_content']
 n=normalize(r)
 assert 'upstream_safety_not_passed' in n['text_normalization']['holds']
 assert 'sensitive_content' in n['review_flags']
 assert 'upstream_safety_not_passed' in normalize(n)['text_normalization']['holds']
