"""Format-based message adaptation. No dataset names, task answers or entity rules."""
import json
from tool_xml import normalize_text, serialize
from glasseek_shared.historical_tools import response_fragments, response_hints, compatible, unique_matching, canonical_tool
from glasseek_shared.hermes.qwen_xml_protocol import strip_batch_wrappers,VERSION as FORMAT_VERSION


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
            framed_text,framing_events=strip_batch_wrappers(text)
            source_assistant=text;text=framed_text
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
            if framing_events:
                step['tool_format_provenance']={'version':FORMAT_VERSION,'original_content':source_assistant,'events':framing_events}
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
            responses=[]
            try:
                fragments, suffix, suffix_start = response_fragments(text) if text.lstrip().startswith('<tool_response>') else ([(0,len(text),text)],'',len(text))
                if not fragments:
                    raise ValueError('response_envelope_missing')
                if suffix.strip():
                    contexts.append({'role':'user','content':suffix,'before_step':len(steps)+1,
                                     'source_message_index':index,'source_content_start':suffix_start,
                                     'source_content_end':len(text)})
                available = [(s,ci,link) for s,ci,link in pending
                             if not link['call_id'] or link['call_id'] not in answered_ids]
                edges, responses = [], []
                for fi,(start,end,body) in enumerate(fragments):
                    hints = response_hints(body)
                    outer_id = message.get('tool_call_id',message.get('response_to'))
                    if outer_id and hints.get('id') and outer_id!=hints['id']:
                        raise ValueError('conflicting_response_identifier')
                    cid = hints.get('id') or (outer_id if len(fragments)==1 else None)
                    name = message.get('name') or hints.get('name')
                    if name: hints['name'] = name
                    response = {'content':body,'response_to':cid,'source_response_id':cid,'source_role':message.get('role',message.get('from')),
                                'source_message_index':index,'source_channel':message.get('channel'),
                                'source_content_start':start,'source_content_end':end,'source_envelope_index':fi}
                    responses.append(response)
                    if cid and (cid not in call_ids or cid in answered_ids):
                        errors.append('unmatched_tool_response_id');edges.append([]);continue
                    # Native IDs and a single outstanding request establish the
                    # binding independently of provider query rewriting/redirects.
                    typed=[candidate for candidate in available if name and compatible(candidate[0]['calls'][candidate[1]],{'name':name})]
                    identity_unique=bool(cid) or len(available)==1 or len(typed)==1
                    binding_hints={'name':name} if identity_unique and name else {} if identity_unique else hints
                    eligible = [j for j,(s,ci,link) in enumerate(available)
                                if (not cid or cid==link['call_id']) and compatible(s['calls'][ci],binding_hints)]
                    if cid and not eligible and cid in call_ids:
                        errors.append('tool_response_name_mismatch')
                    edges.append(eligible)
                mapping = unique_matching(edges,available)
            except ValueError:
                errors.append('ambiguous_tool_response' if pending else 'orphan_tool_observation')
                unassigned.extend(responses or [{'content':text,'source_message_index':index,'response_to':None}])
                continue
            for response, choice in zip(responses,mapping):
                target,ci,link = available[choice]
                if not link['call_id']:
                    generated = f"generated:m{link['source_message_index']}:c{ci}"
                    while generated in reserved_ids: generated += ':generated'
                    reserved_ids.add(generated);link['call_id']=generated
                    link['id_origin']='unique_response_association';call_ids[generated]=(target,ci,link)
                response['response_to']=link['call_id'];answered_ids.add(link['call_id'])
                target['responses'].append(response);target['observation']+=response['content']+'\n'
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
    batch_flags=[]
    if unassigned:
        from web_batching import closed_round_batches
        unassigned,batch_flags=closed_round_batches(steps,unassigned,contexts,instructions,reserved_ids,answered_ids)
        if not unassigned and batch_flags:
            errors=[e for e in errors if e!='ambiguous_tool_response']
    if any(not link['call_id'] or link['call_id'] not in answered_ids for _, _, link in pending):
        errors.append('missing_tool_response')
    item = {'question': question, 'reference_answer': reference, 'steps': steps,
            'program_errors': sorted(set(errors)),'adapter_flags':batch_flags}
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
