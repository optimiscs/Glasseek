"""Opt-in Hermes interception: native submission, bounded correction and review."""
import json
import os
from pathlib import Path
import time
from .prompts import REVIEW_SYSTEM
from .protocol import REVIEW_TOOL, validate_finish, render_finish, strict_json, write_record, observed_urls


def _tool_message(message):
    calls = []
    for call in message.tool_calls or []:
        calls.append({'id': call.id, 'type': 'function', 'function': {
            'name': call.function.name, 'arguments': call.function.arguments}})
    result = {'role': 'assistant', 'content': message.content or '', 'tool_calls': calls}
    reasoning = getattr(message, 'reasoning_content', None)
    if reasoning:
        result['reasoning'] = reasoning
    return result


def _reply(messages, message, text):
    if message.tool_calls:
        messages.append(_tool_message(message))
        for call in message.tool_calls:
            messages.append({'role': 'tool', 'name': call.function.name, 'tool_call_id': call.id, 'content': text})
    else:
        messages.extend([{'role': 'assistant', 'content': message.content or ''},
                         {'role': 'user', 'content': text}])


def state(agent):
    if not hasattr(agent, '_research_state'):
        result = os.environ.get('GLASSEEK_RESEARCH_RESULT')
        if os.environ.get('GLASSEEK_RESUME_PIN_SESSION') == '1' and result and Path(result).exists():
            saved = json.loads(Path(result).read_text())
            if saved.get('protocol') != 'arex-native':
                raise ValueError('Cannot resume a different research protocol')
            agent._research_state = saved
            saved.update(status='incomplete', resumed=True)
            return saved
        agent._research_state = {'protocol': 'arex-native', 'status': 'incomplete',
            'protocol_errors': [], 'candidates': [], 'reviews': []}
    return agent._research_state


def observe(agent, messages):
    record = state(agent)
    record['observed_urls'] = sorted(set(record.get('observed_urls', [])) | observed_urls(messages))


def _review(agent, api_messages, candidate, mode):
    # API messages already contain only inference inputs and observations. Never
    # read release references, evaluator packets or gold answers here.
    review_messages = [{'role': 'system', 'content': REVIEW_SYSTEM}]
    review_messages.extend(m for m in api_messages if m.get('role') != 'system')
    review_messages.append({'role': 'user', 'content': json.dumps({
        'review_mode': mode, 'candidate': candidate}, ensure_ascii=False)})
    kwargs = agent._build_api_kwargs(review_messages)
    kwargs.update(tools=[REVIEW_TOOL], tool_choice={'type': 'function', 'function': {'name': 'review_candidate'}},
                  parallel_tool_calls=False, temperature=0)
    # Main-model reasoning remains enabled. Reviewer output shares the original
    # output-token allowance and existing network/interruption controls.
    start = time.monotonic()
    event = {'input': review_messages, 'candidate': candidate, 'mode': mode}
    try:
        response = agent._interruptible_api_call(kwargs)
        usage = getattr(response, 'usage', None)
        event['usage'] = usage.model_dump() if usage and hasattr(usage, 'model_dump') else None
        message = response.choices[0].message
        event['output'] = message.model_dump() if hasattr(message, 'model_dump') else str(message)
        calls = message.tool_calls or []
        if len(calls) != 1 or calls[0].function.name != 'review_candidate':
            raise ValueError('review must return one review_candidate call')
        result = strict_json(calls[0].function.arguments)
        if not isinstance(result, dict) or set(result) != {'accept', 'issues', 'next_steps'} or type(result['accept']) is not bool:
            raise ValueError('invalid review fields')
        if any(not isinstance(result[k], list) or any(not isinstance(v, str) for v in result[k]) for k in ('issues', 'next_steps')):
            raise ValueError('invalid review lists')
        if result['accept'] and result['issues']:
            raise ValueError('review acceptance conflicts with unresolved issues')
        event['feedback'] = result
    except Exception as exc:
        event['error'] = type(exc).__name__ + ': ' + str(exc)[:300]
        event['feedback'] = {'accept': False, 'issues': ['Review failed; candidate was not accepted.'],
                             'next_steps': ['Verify evidence and constraints before resubmitting.']}
    event['seconds'] = time.monotonic() - start
    return event


def handle_response(agent, message, messages, api_messages, api_calls):
    """Return (action, final_text, extra_api_calls). Normal tools return 'execute'."""
    record = state(agent)
    calls = message.tool_calls or []
    allowed = {'web_search', 'web_extract', 'execute_code', 'vision_analyze', 'finish'}
    error = None
    if len(calls) != 1:
        error = 'Exactly one native tool call is required. finish must be called alone; ordinary text is not a submission.'
    elif calls[0].function.name not in allowed:
        error = 'Unsupported tool. Use only the provided native schemas; no alias repair is performed.'
    elif calls[0].function.name != 'finish':
        return 'execute', None, 0
    else:
        try:
            observe(agent, messages)
            catalog = {'role': 'tool', 'content': json.dumps(record.get('observed_urls', []))}
            candidate = validate_finish(calls[0].function.arguments, os.environ['GLASSEEK_RESEARCH_DATASET'], [catalog])
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
    if error:
        record['protocol_errors'].append({'api_call': api_calls, 'reason': error,
                                          'response': _tool_message(message)})
        _reply(messages, message, 'Protocol error: ' + error + ' Correct the next call; do not repeat ambiguous output.')
        limit = int(os.environ.get('GLASSEEK_RESEARCH_CORRECTIONS', '3'))
        if len(record['protocol_errors']) >= limit:
            record.update(status='protocol_failed', reason='correction budget exhausted')
            write_record(record)
            return 'failed', None, 0
        write_record(record)
        return 'continue', None, 0

    record['candidates'].append({'api_call': api_calls, 'submission': candidate})
    threshold = float(os.environ.get('GLASSEEK_RESEARCH_ACCEPT_CONFIDENCE', '95'))
    medium = float(os.environ.get('GLASSEEK_RESEARCH_MEDIUM_CONFIDENCE', '90'))
    reviews = int(os.environ.get('GLASSEEK_RESEARCH_MAX_REVIEWS', '2'))
    extra = 0
    accepted = candidate['confidence'] >= threshold
    if not accepted and len(record['reviews']) < reviews and api_calls < agent.max_iterations and agent.iteration_budget.consume():
        event = _review(agent, api_messages, candidate, 'verify_candidate' if candidate['confidence'] >= medium else 'challenge_candidate')
        extra = 1
        record['reviews'].append(event)
        accepted = event['feedback']['accept']
        if not accepted and len(record['reviews']) < reviews and api_calls + extra < agent.max_iterations:
            _reply(messages, message, 'Candidate not accepted. Review feedback: ' + json.dumps(event['feedback'], ensure_ascii=False))
            write_record(record)
            return 'continue', None, extra
    if not accepted:
        _reply(messages, message, 'Candidate saved but not accepted; review or iteration budget exhausted.')
        record.update(status='incomplete', reason='candidate unresolved within review/iteration budget')
        write_record(record)
        return 'failed', None, extra
    final = render_finish(candidate, os.environ['GLASSEEK_RESEARCH_DATASET'])
    _reply(messages, message, '{"accepted":true,"meaning":"submission accepted; correctness is independently graded"}')
    messages.append({'role': 'assistant', 'content': final})
    record.update(status='submitted', submission=candidate, rendered_answer=final,
                  acceptance='confidence_stop' if candidate['confidence'] >= threshold else 'evidence_review',
                  research_api_calls=api_calls + extra)
    write_record(record)
    return 'finished', final, extra
