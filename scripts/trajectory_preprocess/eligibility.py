"""Reuse the existing safety and protocol gates before any LLM request."""
import re
from client import digest
import policy
from task_routing import route


def observation_matches(observation, responses, *, outer=True):
    """Allow only adapter envelopes/separators; preserve every response body byte."""
    from glasseek_shared.extract_protocol import merged_response_slices,batch_response_slices
    expanded=[]
    for response in responses:
        text=response['content']
        if response.get('merged_response_parts'):
            offsets=merged_response_slices(text,response['merged_response_parts'])
        elif response.get('batch_response_parts'):
            offsets=batch_response_slices(text,response['batch_response_parts'])
        else: offsets=[(0,len(text))]
        expanded.extend({'content':text[a:b]} for a,b in offsets)
    responses=expanded
    text=observation.strip()
    if outer and text.startswith('<tool_response>') and text.endswith('</tool_response>'):
        if observation_matches(text[len('<tool_response>'):-len('</tool_response>')],responses,outer=False):return True
    remaining=observation
    for response in responses:
        body=response['content'].strip()
        if body.startswith('<tool_response>') and body.endswith('</tool_response>'):
            body=body[len('<tool_response>'):-len('</tool_response>')].strip()
        remaining=remaining.lstrip()
        framed=remaining.startswith('<tool_response>')
        if framed: remaining=remaining[len('<tool_response>'):].lstrip()
        if not remaining.startswith(body):return False
        remaining=remaining[len(body):].lstrip()
        if framed:
            if not remaining.startswith('</tool_response>'):return False
            remaining=remaining[len('</tool_response>'):]
    return not remaining.strip()


def assess(item, canonical, scanner):
    routing = route(item, canonical)
    result = {'routing':routing,'status':'eligible','reasons':[], 'training_ready':False}
    def hold(status, reason):
        result.update(status=status); result['reasons'].append(reason)
        return result
    if canonical.get('id') != item['id'] or canonical.get('source') != item['source']:
        return hold('repair_pending','canonical_identity_mismatch')
    # Match actual assistant text and original message positions, not just IDs.
    from migration_render import target_messages
    rendered,_ = target_messages({'agent_role':canonical.get('agent_role','parent'),'messages':[
        {'role':'assistant','content':s['assistant'],'source_message_index':s.get('source_message_index')} for s in item['steps']]})
    original = {(m.get('source_message_index'),m['content']) for m in rendered}
    exported = {(m.get('source_message_index'),m['content']) for m in canonical['messages'] if m['role']=='assistant'}
    if original != exported:
        return hold('repair_pending','canonical_assistant_content_mismatch')
    if item.get('normalized_schema_version'):
        messages=canonical['messages']
        # Every judge-visible observation must be the exact recorded response,
        # not an unscanned parallel copy. The adapter adds one newline per return.
        for step in item['steps']:
            from call_protocol import render_step
            responses=step.get('responses',[])
            if not observation_matches(step['observation'],responses):
                return hold('repair_pending','normalized_observation_mismatch')
            for response in render_step(step)[1:]:
                matches=[m for m in messages if m.get('origin')=='tool_response' and
                         m.get('response_to')==response.get('response_to') and m['content']==response['content']]
                if len(matches)!=1: return hold('repair_pending','canonical_observation_mismatch')
        if not any(m['content']==item['question'] and m['role']=='user' for m in messages):
            return hold('repair_pending','canonical_question_mismatch')
        for context in item.get('context_messages',[])+item.get('instruction_messages',[]):
            if not any(m['content']==context['content'] and m['role']==context['role'] for m in messages):
                return hold('repair_pending','canonical_instruction_mismatch')
    safety = scanner.scan(canonical)
    result['safety'] = safety
    result['canonical_sha256'] = digest(canonical)
    if safety['state'] == 'quarantine': return hold('quarantine','existing_safety_gate')
    if safety['state'] != 'pass_rules' or not safety['coverage_complete']:
        return hold('manual_review','existing_safety_review')
    if item.get('program_errors') or item.get('harness',{}).get('errors'):
        return hold('repair_pending','existing_structure_or_protocol_gate')
    if not item.get('question','').strip() or not item['steps']:
        return hold('incomplete','missing_question_or_steps')
    if any(s.get('calls') and not s.get('observation','').strip() for s in item['steps']):
        return hold('repair_pending','missing_tool_observation')
    last = item['steps'][-1]
    if last.get('calls') or last['assistant'].count('<think>') != last['assistant'].count('</think>'):
        return hold('incomplete','unfinished_final_answer')
    body = policy.answer_body(item)
    if not body or re.fullmatch(r'(?:[.\s…]+|TODO|TBD|PLACEHOLDER|待补充)',body,re.I):
        return hold('incomplete','placeholder_final_answer')
    if routing['trajectory_form'] == 'summary_continuation' and not any(s.get('observation','').strip() for s in item['steps']):
        return hold('evidence_insufficient','summary_is_not_original_tool_evidence')
    if routing['trajectory_form'] == 'direct_answer':
        return hold('manual_review','direct_answer_requires_separate_verifiable_proof')
    if not policy.evidence_catalog(item): return hold('evidence_insufficient','no_traceable_observation')
    return result
