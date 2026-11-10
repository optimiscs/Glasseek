"""Hermes completion boundary for the pinned CLM loop.

The upstream checkout stays pristine. Only its no-action branch is replaced in
the task's private module; context editing, token gates, rollback and checkpoint
code execute unchanged. A missing source anchor fails closed after an upgrade.
"""
import hashlib
import re

PROTOCOL = 'hermes-assistant-v1'
ACTION_INSTRUCTIONS = '''Use the available tools while researching. Use `bash` to edit your context
mirror or run a web command. When the task is complete, respond directly with
the final answer in an assistant message with no tool calls, following the
task's answer format. Reasoning alone is not a final answer. Tool-call turns
continue the task; a complete final answer ends it.'''

_ORIGINAL_ACTION_INSTRUCTIONS = '''Your response must include a THOUGHT section before your action where you
explain your reasoning. After the THOUGHT, you must call the `bash` tool
with EXACTLY ONE bash command (multiple commands chained with `&&` or `||`
count as a single action).

Failure to follow these rules — calling no tool, calling a tool other than
`bash`, or omitting the THOUGHT — will cause your response to be rejected.'''

_ORIGINAL_NO_ACTION = '''                if action["type"] == "no_tool_call":
                    msg.pop("tool_calls", None)
                    messages.append(msg)
                    agent_step_metrics.append(turn_metrics)
                    messages.append(
                        {"role": "user", "content": "(No tool call in your last turn. When "
                         "ready, issue your next `bash` command to continue the task.)"}
                    )
                    timing_log.append({"step": step + 1, "llm_s": round(llm_time, 1), "empty_turn": True})
                    step += 1  # an empty (no-tool-call) turn counts (as before)
                    continue
'''
_HERMES_NO_ACTION = '''                if action["type"] == "no_tool_call":
                    complete = _hermes_completion(self, msg, response.choices[0].finish_reason, messages)
                    agent_step_metrics.append(turn_metrics)
                    timing_log.append({"step": step + 1, "llm_s": round(llm_time, 1),
                                       "final_answer": complete, "empty_turn": not complete})
                    step += 1
                    if complete:
                        break
                    continue
'''


def patch_loop(source):
    if source.count(_ORIGINAL_NO_ACTION) != 1:
        raise RuntimeError('Pinned CLM no-action branch changed; review Hermes completion integration')
    replacements = {
        '            while step < self.max_steps and total_iters < max_total_iters:':
            '            while _glasseek_continue_run(self, step, total_iters, max_total_iters):',
        '                # Record the context actually sent this step (after gate/nudge).':
            '                _glasseek_prepare_turn(self, messages)\n\n'
            '                # Record the context actually sent this step (after gate/nudge).',
        '                if action["type"] == "context_status":':
            '                if not _glasseek_admit_tool_turn(self, msg, messages):\n'
            '                    agent_step_metrics.append(turn_metrics)\n'
            '                    break\n\n'
            '                if action["type"] == "context_status":',
    }
    source = source.replace(_ORIGINAL_NO_ACTION, _HERMES_NO_ACTION)
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise RuntimeError('Pinned CLM tool-round boundary changed; review integration')
        source = source.replace(before, after)
    return source


def prompt_template(original):
    if original.count(_ORIGINAL_ACTION_INSTRUCTIONS) != 1:
        raise RuntimeError('Pinned CLM action instructions changed')
    return original.replace(_ORIGINAL_ACTION_INSTRUCTIONS, ACTION_INSTRUCTIONS).replace('{{finish_instructions}}', '')


def final_text(message, finish_reason=None):
    if message.get('tool_calls') or message.get('tool_calls_raw'):
        return ''
    if finish_reason and str(finish_reason).lower() not in ('stop', 'end_turn', 'done', 'complete', 'completed'):
        return ''
    text = message.get('content') or ''
    if not isinstance(text, str):
        return ''
    if '<think>' in text and '</think>' not in text:
        return ''
    text = text.rsplit('</think>', 1)[-1].strip()
    lower=text.lower()
    raw_call=re.match(r'^\s*<\s*(?:tool_call\b|function\s*=|parameter\s*=)',text,re.I)
    if not text or lower == '(empty)' or raw_call or ('<tool_call' in lower and '</tool_call>' in lower):
        return ''
    return text


def complete_or_continue(agent, message, finish_reason, messages):
    """Keep the actual model message; never promote reasoning to an answer."""
    messages.append(message)
    answer = final_text(message, finish_reason)
    if answer:
        agent.glasseek_completion = {'protocol': PROTOCOL, 'stop_reason': 'answer', 'answer': answer}
        return True
    calls = message.get('tool_calls') or []
    if calls:
        # Invalid/non-bash calls rejected by upstream still require paired results.
        for call in calls:
            messages.append({'role': 'tool', 'tool_call_id': call['id'],
                'content': 'Invalid tool call. Use the advertised bash schema, or provide a final assistant answer without tool calls.'})
    else:
        messages.append({'role': 'user', 'content':
            'No complete final answer was received. Continue with a valid tool call, or provide the final answer directly without tool calls. Reasoning alone and truncated output do not count as an answer.'})
    return False


def install(agent, module):
    """Adapt finish wording only; retain upstream budget decisions and limits."""
    import types
    from .tool_rounds import continue_run, prepare_turn, admit_tool_turn
    module._glasseek_continue_run = continue_run
    module._glasseek_prepare_turn = prepare_turn
    module._glasseek_admit_tool_turn = admit_tool_turn
    module._hermes_completion = complete_or_continue
    agent.glasseek_completion = {'protocol': PROTOCOL, 'stop_reason': 'no_final_answer', 'answer': ''}
    original_nudge = agent._finalize_nudge

    def nudge(self, step):
        budget = getattr(self, 'glasseek_tool_round_budget', None)
        if budget is not None:
            left = budget.limit - budget.used
            if 0 < left <= 3:
                return {'role': 'user', 'content':
                    f'Only {left} tool rounds remain ({budget.used}/{budget.limit} used, including context edits). '
                    'Conclude research and give your final answer directly without tool calls when ready.'}
            return None
        message = original_nudge(step)
        if message:
            head = message['content'].split(' Stop exploring now:', 1)[0]
            return {'role': 'user', 'content': head +
                ' Stop exploring now and give your final answer directly as an assistant message without tool calls, in the format requested by the task.'}
        return None

    def final_budget_message(self, tokens):
        return (f'[SYSTEM NOTICE] You have reached your context budget ({tokens}/{self.budget} tokens). '
                'This is your FINAL turn. Give your final answer directly as an assistant message without tool calls, '
                'in the format requested by the task. After this turn the session ends and your answer is evaluated as-is.')

    agent._finalize_nudge = types.MethodType(nudge, agent)
    agent._budget.final_message = types.MethodType(final_budget_message, agent._budget)
    # Upstream warns about malformed echo submissions; echo is just an ordinary
    # shell command in Hermes, so it must not advertise a second finish protocol.
    agent._ctx.append_submit_warning = lambda command, content: content
    original_replay = agent._resume.replay

    def replay(self, messages, **kwargs):
        result = original_replay(messages, **kwargs)
        if self.enabled and result and str(result[-1].get('content') or '').startswith('SESSION RESUMED.'):
            original_note=result[-1]['content']
            note=('SESSION RESUMED. The checkpoint workspace and available conversation history have been restored to '
                  +agent.glasseek_paths['workspace']+'. Background processes from the previous session have not survived. '
                  'Verify the effects of any interrupted action before continuing. Preserve existing evidence and task files. '
                  'When you have sufficient evidence, give the final answer directly as an assistant message without tool calls, '
                  'in the format requested by the task.')
            if self.context_missing:note+=' Previous conversation history was unavailable; inspect the restored evidence before relying on it.'
            if 'its first ' in original_note:note+=' Earlier conversation turns were dropped to fit the replay budget.'
            result[-1]={**result[-1],'content':note}
        return result

    agent._resume.replay = types.MethodType(replay, agent._resume)
    agent.glasseek_completion_patch_sha256 = hashlib.sha256((_ORIGINAL_NO_ACTION + _HERMES_NO_ACTION).encode()).hexdigest()
