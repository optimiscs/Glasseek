"""Deterministic scan views: role is not a substitute for content provenance."""
from dataclasses import dataclass
import json

from tool_xml import normalize_text
from glasseek_shared.extract_protocol import merged_response_slices, batch_response_slices
from glasseek_shared.hermes.qwen_xml_protocol import has_batch_wrapper


@dataclass
class View:
    pointer: str
    text: str
    kind: str
    message_index: int = -1
    derived: bool = False


def leaves(value, pointer):
    if isinstance(value, str):
        yield pointer, value
    elif isinstance(value, dict):
        for key, child in value.items():
            escaped = str(key).replace('~', '~0').replace('/', '~1')
            yield pointer + '/' + escaped + '/@key', str(key)
            yield from leaves(child, pointer + '/' + escaped)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from leaves(child, pointer + '/' + str(i))
    elif isinstance(value,(int,float)) and not isinstance(value,bool):
        yield pointer,str(value)


def build_context(item):
    views, actions, issues = [], [], []
    messages = item.get('messages')
    if isinstance(messages, str):
        try: messages = json.loads(messages)
        except ValueError: messages = None
    if not isinstance(messages, list) or not messages:
        return views, actions, ['messages_unparseable']
    pending = False; outstanding = set(); seen_ids = set()
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            issues.append('invalid_message'); continue
        role = message.get('role'); content = message.get('content')
        if role not in ('system', 'developer', 'user', 'assistant', 'tool'):
            issues.append('unsupported_role')
        if not isinstance(content, str):
            issues.append('non_text_content')
            content = ''
        external = role == 'tool' or (role == 'user' and content.lstrip().startswith('<tool_response>'))
        kind = 'external' if external else ('assistant' if role == 'assistant' else 'task' if role == 'user' else 'source_instruction')
        spans=[];part_field='merged_response_parts'
        if external and 'merged_response_parts' in message:
            try:spans=merged_response_slices(content,message['merged_response_parts'])
            except ValueError:issues.append('invalid_merged_response_parts')
            else:kind='external_batch'
        if external and 'batch_response_parts' in message:
            part_field='batch_response_parts'
            if 'merged_response_parts' in message:issues.append('invalid_batch_response_parts')
            try:spans=batch_response_slices(content,message['batch_response_parts'])
            except ValueError:issues.append('invalid_batch_response_parts')
            else:kind='external_batch'
        views.append(View(f'/messages/{i}/content', content, kind, i))
        for part,(start,end) in enumerate(spans):
            views.append(View(f'/messages/{i}/{part_field}/{part}/content',content[start:end],'external_part',i,True))
        if role == 'assistant':
            if has_batch_wrapper(content):issues.append('noncanonical_tool_batch_wrapper')
            if pending or outstanding: issues.append('missing_tool_response')
            pending = False; outstanding = set()
            try:
                _, calls = normalize_text(content, message.get('tool_calls'))
            except (ValueError, TypeError, KeyError, AttributeError):
                issues.append('tool_call_unparseable'); calls = []
            for j, call in enumerate(calls):
                action = {'message_index': i, 'call_index': j, **call}
                actions.append(action)
                for pointer, text in leaves(call['arguments'], f'/messages/{i}/parsed_calls/{j}/arguments'):
                    views.append(View(pointer, text, 'argument', i, True))
            pending = bool(calls)
            links = message.get('call_links')
            if links is not None and len(links) != len(calls):
                issues.append('tool_call_metadata_mismatch')
            for call in (message.get('tool_calls') or []) + (links or []):
                cid = call.get('id', call.get('call_id'))
                if cid:
                    if cid in seen_ids: issues.append('duplicate_tool_call_id')
                    seen_ids.add(cid); outstanding.add(cid)
        elif external:
            cid = message.get('tool_call_id', message.get('response_to'))
            if cid:
                if cid not in outstanding: issues.append('unmatched_tool_response_id')
                outstanding.discard(cid); pending = bool(outstanding)
            elif not pending:
                issues.append('orphan_tool_response')
            else:
                # Canonical XML stores a grouped observation for the whole call batch.
                if outstanding: issues.append('native_response_id_missing')
                pending = False; outstanding = set()
        for pointer, text in leaves({k:v for k,v in message.items() if k != 'content'}, f'/messages/{i}'):
            views.append(View(pointer, text, 'metadata', i))
    if pending or outstanding: issues.append('missing_tool_response')
    for pointer, text in leaves({k:v for k,v in item.items() if k != 'messages'}, ''):
        views.append(View(pointer, text, 'metadata'))
    return views, actions, sorted(set(issues))
