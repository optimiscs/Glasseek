"""Call/response provenance is separate from model-visible XML syntax."""
import copy
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Shared'))

VERSION = 'glasseek-call-response-v1'


def semantic_messages(messages):
    """Exclude arbitrary IDs/positions from dedup; retain response association meaning."""
    ids = {link['call_id']: [mi, ci] for mi, m in enumerate(messages)
           for ci, link in enumerate(m.get('call_links', [])) if link.get('call_id')}
    output = []
    for message in messages:
        value = {k: message[k] for k in ('role', 'content')}
        if message.get('response_to'):
            value['response_call_position'] = ids.get(message['response_to'], 'unresolved')
        output.append(value)
    return output


def ensure_links(item):
    """Legacy canonical steps are only recoverable when one call has one observation."""
    for step in item['steps']:
        if 'call_links' in step and 'responses' in step:
            continue
        step['call_links'], step['responses'] = [], []
        if not step['calls']:
            if step['observation'].strip():
                item['program_errors'].append('orphan_tool_observation')
            continue
        if len(step['calls']) != 1 or not step['observation'].strip():
            item['program_errors'].append('ambiguous_legacy_call_response' if len(step['calls']) > 1 else 'missing_tool_response')
            continue
        cid = f"generated:step{step['step_id']}:c0"
        step['call_links'] = [{'call_id': cid, 'source_message_index': step.get('source_message_index'),
                               'source_step_id': step['step_id'], 'source_call_index': 0,
                               'id_origin': 'unique_legacy_step_association'}]
        step['responses'] = [{'response_to': cid, 'content': step['observation'],
                              'source_role': 'unknown', 'source_message_index': None,
                              'source_step_id': step['step_id']}]
    item['call_protocol_version'] = VERSION
    item['program_errors'] = sorted(set(item['program_errors']))


def materialize_links(step, mapped_by_call, reserved_ids=None):
    """Map calls without splitting historical query/page return payloads."""
    links=copy.deepcopy(step.get('call_links',[]));responses=copy.deepcopy(step.get('responses',[]))
    original_links,original_responses=copy.deepcopy(links),copy.deepcopy(responses)
    source_order={r['response_to']:i for i,r in enumerate(responses)}
    target_links,target_responses,flags=[],[],[]
    if len(links)!=len(mapped_by_call):
        flags.append('target_call_response_alignment_required')
    else:
        for link,calls in zip(links,mapped_by_call):
            if len(calls)!=1:
                flags.append('target_call_response_alignment_required');continue
            matching=[r for r in responses if r['response_to']==link['call_id']]
            if len(matching)!=1:
                flags.append('target_call_response_alignment_required');continue
            response=matching[0]
            link.update(copy.deepcopy(calls[0]));target_links.append(link)
            if calls[0]['name'] in ('web_search','web_extract'):
                response['web_response_scope']='call_batch'
                response['page_boundary_status']='not_required_for_batch'
            target_responses.append(response)
    if flags:
        step['source_call_links']=original_links;step['source_responses']=original_responses
    target_responses.sort(key=lambda r:source_order[r['response_to']])
    step['call_links'],step['responses']=target_links,target_responses
    step['association_status']='review_required' if 'target_call_response_alignment_required' in flags else 'linked'
    return flags


def render_step(step):
    """Qwen training uses user + tool_response; provenance stays in metadata."""
    rendered_links=[]
    for link in step.get('call_links',[]):
        call=link if 'name' in link and 'arguments' in link else step['calls'][link.get('source_call_index',0)]
        rendered_links.append({**copy.deepcopy(call),**copy.deepcopy(link)})
    assistant = {'role': 'assistant', 'content': step['assistant'],
                 'source_message_index': step.get('source_message_index'),
                 'call_links':rendered_links}
    result = [assistant]
    if step.get('responses'):
        for response in step['responses']:
            text = response['content']
            metadata=copy.deepcopy(response)
            if not text.lstrip().startswith('<tool_response>'):
                prefix='<tool_response>\n'
                text = prefix + text + '\n</tool_response>'
                for field in ('merged_response_parts','batch_response_parts'):
                    for part in metadata.get(field,[]):
                        part['start']+=len(prefix);part['end']+=len(prefix)
            result.append({**metadata, 'role': 'user', 'content': text,
                           'origin': 'tool_response',
                           'tool_call_id': response['response_to']})
            if 'search_return_audit' in result[-1]:
                result[-1]['search_return_audit']['rendered_response_sha256']=hashlib.sha256(text.encode()).hexdigest()
    elif step['observation']:
        # Readable review archive, explicitly ineligible for a protocol release.
        text = step['observation'].rstrip('\n')
        if not text.lstrip().startswith('<tool_response>'):
            text = '<tool_response>\n' + text + '\n</tool_response>'
        result.append({'role': 'user', 'content': text, 'origin': 'tool_response',
                       'response_to': None, 'association_status': 'review_required'})
    return result


def model_visible_chars(item):
    """Characters in actual rendered message contents, once per visible message.

    This is a character budget, not a tokenizer limit. Provenance, alternate raw
    copies, call-link metadata and serialization escaping never consume it.
    """
    return (len(item['question']) + sum(len(m['content']) for key in ('instruction_messages','context_messages')
             for m in item.get(key, [])) + sum(len(m['content']) for step in item['steps'] for m in render_step(step)))
