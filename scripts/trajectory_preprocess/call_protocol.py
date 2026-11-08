"""Call/response provenance is separate from model-visible XML syntax."""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Shared'))
try:
    from extract_protocol import historical_results  # frozen run snapshot
except ImportError:
    from glasseek_shared.extract_protocol import historical_results

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


def materialize_links(step, mapped_by_call):
    """Retain IDs for one-to-one mappings; never copy a shared observation to new calls."""
    links = copy.deepcopy(step.get('call_links', []))
    responses = copy.deepcopy(step.get('responses', []))
    flags = []
    if len(links) != len(mapped_by_call) or any(len(calls) != 1 for calls in mapped_by_call):
        # Retain source provenance, explicitly invalidate the target association.
        step['source_call_links'] = links
        step['source_responses'] = responses
        step['call_links'] = []
        step['responses'] = []
        step['association_status'] = 'review_required'
        return ['target_call_response_alignment_required']
    for link, calls in zip(links, mapped_by_call):
        call = calls[0]
        link.update(copy.deepcopy(call))
        if call['name'] != 'web_extract':
            continue
        matching = [r for r in responses if r['response_to'] == link['call_id']]
        if len(matching) != 1:
            flags.append('target_call_response_alignment_required')
            continue
        response = matching[0]
        payload, reason = historical_results(response['content'], call['arguments']['urls'])
        if reason:
            response['page_boundary_status'] = reason
            flags.append(reason)
        else:
            response['original_content'] = response['content']
            response['content'] = json.dumps(payload, ensure_ascii=False)
            response['page_boundary_status'] = 'verified_explicit_json'
            if any('error' not in page for page in payload['results']):
                flags.append('extract_page_status_unknown')
    step['call_links'], step['responses'] = links, responses
    step['association_status'] = 'linked'
    return flags


def render_step(step):
    """Qwen training uses user + tool_response; provenance stays in metadata."""
    assistant = {'role': 'assistant', 'content': step['assistant'],
                 'source_message_index': step.get('source_message_index'),
                 'call_links': [{**copy.deepcopy(call), **copy.deepcopy(link)}
                                for call, link in zip(step.get('calls', []), step.get('call_links', []))]}
    result = [assistant]
    if step.get('responses'):
        for response in step['responses']:
            text = response['content']
            if not text.lstrip().startswith('<tool_response>'):
                text = '<tool_response>\n' + text + '\n</tool_response>'
            result.append({**copy.deepcopy(response), 'role': 'user', 'content': text,
                           'origin': 'tool_response',
                           'tool_call_id': response['response_to']})
    elif step['observation']:
        # Readable review archive, explicitly ineligible for a protocol release.
        text = step['observation'].rstrip('\n')
        if not text.lstrip().startswith('<tool_response>'):
            text = '<tool_response>\n' + text + '\n</tool_response>'
        result.append({'role': 'user', 'content': text, 'origin': 'tool_response',
                       'response_to': None, 'association_status': 'review_required'})
    return result
