"""Format-based message adaptation. No dataset names, task answers or entity rules."""
import json
from tool_xml import normalize_text, serialize


class AdapterUnsupported(ValueError):
    pass


def content_text(content, role):
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise AdapterUnsupported('Unsupported content container')
    parts = []
    for block in content:
        if not isinstance(block, dict):
            raise AdapterUnsupported('Unsupported content block')
        text = block.get('text')
        kind = block.get('type')
        if kind not in (None, 'text', 'input_text', 'output_text') and not (role in ('system','developer') and kind == 'system_content'):
            raise AdapterUnsupported('Non-text content block: ' + str(kind))
        # Arrow schemas materialize irrelevant union fields as null. Never discard
        # actual non-text payloads such as images or audio to make a row pass.
        extras = {k: v for k, v in block.items() if k not in ('text', 'type') and v is not None}
        if extras and role not in ('system', 'developer'):
            raise AdapterUnsupported('Non-text payload in content block')
        if text is not None:
            if not isinstance(text, str):
                raise AdapterUnsupported('Non-string text block')
            parts.append(text)
    return '\n'.join(parts)


def conversation(row):
    messages = row.get('trajectory', row.get('messages', row.get('conversations')))
    if not isinstance(messages, list):
        raise AdapterUnsupported('Unsupported message schema')
    meta = row.get('meta') or {}
    question = row.get('question', meta.get('question', ''))
    reference = row.get('answer', meta.get('answer', ''))
    steps, contexts, instructions, errors = [], [], [], []
    call_ids, answered_ids, pending = {}, set(), []
    unassigned = []
    question_index = None
    reserved_ids = {c.get('id', c.get('call_id')) for m in messages if isinstance(m, dict)
                    for c in (m.get('tool_calls') or []) if isinstance(c, dict)}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError('Invalid message object')
        role = message.get('role', message.get('from'))
        role = {'human': 'user', 'gpt': 'assistant', 'function': 'tool'}.get(role, role)
        raw_content = message.get('content', message.get('value', ''))
        text = content_text(raw_content, role)
        if role in ('system', 'developer'):
            instructions.append({'role': role, 'content': text, 'before_step': len(steps)+1,
                                 'source_message_index': index,
                                 'structured_metadata': raw_content if isinstance(raw_content, list) else None})
        elif role == 'assistant':
            recipient = message.get('recipient')
            native = message.get('tool_calls')
            original_payload = None
            if recipient not in (None, '', 'all', 'assistant'):
                if native:
                    raise AdapterUnsupported('Both recipient routing and native tool_calls present')
                try:
                    args = json.loads(text)
                except json.JSONDecodeError:
                    # Non-JSON code routed to a code interpreter remains an
                    # unsupported tool, rather than a falsely empty final answer.
                    if str(recipient).casefold() not in ('python', 'pythoninterpreter', 'execute_code'):
                        raise
                    args = {'code': text}
                if not isinstance(args, dict):
                    raise ValueError('Routed tool payload must be an object')
                original_payload = text
                text, calls = normalize_text('', [{'name': recipient, 'arguments': args}])
            else:
                if message.get('channel') == 'analysis':
                    if '<think>' in text or '</think>' in text:
                        raise AdapterUnsupported('Reasoning channel contains think delimiters')
                    text = '<think>\n' + text + '\n</think>'
                text, calls = normalize_text(text, native)
            step = {'step_id': len(steps)+1, 'assistant': text, 'calls': calls, 'observation': '',
                    'call_links': [], 'responses': []}
            if message.get('channel') is not None:
                step['channel'] = message['channel']
            if original_payload is not None:
                step['source_tool_payload'] = original_payload
            # Source indices are positional metadata, not used in semantic hashes.
            step['source_message_index'] = index
            steps.append(step)
            for ci, call in enumerate(calls):
                c = native[ci] if native else {}
                cid = c.get('id', c.get('call_id'))
                link = {'call_id': cid, 'source_message_index': index, 'source_call_index': ci,
                        'source_role': message.get('role', message.get('from')), 'source_channel': message.get('channel'),
                        'id_origin': 'native' if cid else 'pending_unique_association'}
                step['call_links'].append(link)
                pending.append((step, ci, link))
                if cid:
                    if cid in call_ids:
                        errors.append('duplicate_tool_call_id')
                    call_ids[cid] = (step, ci, link)
        elif role == 'tool' or (role == 'user' and text.lstrip().startswith('<tool_response>')):
            # Qwen may put a genuine user reminder after the response envelope.
            # Its explicit closing delimiter gives a deterministic boundary;
            # preserve the suffix as task context, not retrieved evidence.
            response_end = len(text)
            if role == 'user' and text.count('<tool_response>') == 1 and text.count('</tool_response>') == 1:
                response_end = text.index('</tool_response>') + len('</tool_response>')
                suffix = text[response_end:]
                if suffix.strip():
                    contexts.append({'role': 'user', 'content': suffix, 'before_step': len(steps)+1,
                                     'source_message_index': index, 'source_content_start': response_end,
                                     'source_content_end': len(text)})
                text = text[:response_end]
            cid = message.get('tool_call_id', message.get('response_to'))
            response = {'content': text, 'response_to': cid, 'source_role': message.get('role', message.get('from')),
                        'source_message_index': index, 'source_channel': message.get('channel'),
                        'source_content_start': 0, 'source_content_end': response_end}
            name = message.get('name')
            if cid:
                found = call_ids.get(cid)
                if found is None or cid in answered_ids:
                    errors.append('unmatched_tool_response_id')
                    unassigned.append(response)
                    continue
            else:
                candidates = [(s, ci, link) for s, ci, link in pending
                              if (not link['call_id'] or link['call_id'] not in answered_ids)
                              and (not name or name.casefold() == s['calls'][ci]['name'].casefold())]
                if len(candidates) != 1:
                    errors.append('ambiguous_tool_response' if candidates else 'orphan_tool_observation')
                    unassigned.append(response)
                    continue
                found = candidates[0]
            target, ci, link = found
            if name and name.casefold() != target['calls'][ci]['name'].casefold():
                errors.append('tool_response_name_mismatch')
            if not link['call_id']:
                generated = f"generated:m{link['source_message_index']}:c{ci}"
                while generated in reserved_ids:
                    generated += ':generated'
                reserved_ids.add(generated)
                link['call_id'] = generated
                link['id_origin'] = 'unique_response_association'
                call_ids[generated] = found
            response['response_to'] = link['call_id']
            answered_ids.add(link['call_id'])
            target['responses'].append(response)
            target['observation'] += text + '\n'
        elif role == 'user':
            if not steps and not question:
                question = text
                question_index = index
            elif not steps and text == question:
                question_index = index
            elif steps or text != question:
                contexts.append({'role': 'user', 'content': text, 'before_step': len(steps)+1,
                                 'source_message_index': index})
        else:
            raise AdapterUnsupported('Unsupported message role: ' + str(role))
    if any(not link['call_id'] or link['call_id'] not in answered_ids for _, _, link in pending):
        errors.append('missing_tool_response')
    item = {'question': question, 'reference_answer': reference, 'steps': steps,
            'program_errors': sorted(set(errors))}
    if question_index is not None:
        item['question_source_message_index'] = question_index
    if unassigned:
        item['unassigned_responses'] = unassigned
    if contexts:
        item['context_messages'] = contexts
        if contexts[-1]['before_step'] > len(steps):
            item['program_errors'].append('unanswered_user_message')
    if instructions:
        item['instruction_messages'] = instructions
    return item


def task_text(item):
    """Judge-visible requirements with chronology; never evidence observations."""
    text = item['question']
    for message in sorted(item.get('context_messages', []) + item.get('instruction_messages', []),
                          key=lambda m: (m['before_step'], m.get('source_message_index', -1))):
        if message['content']:
            text += f'\n\n[Recorded {message["role"]} instruction before assistant step {message["before_step"]}]\n' + message['content']
    return text
