"""Opt-in cumulative tool-round limit, independent of CLM compaction steps."""
import json
from pathlib import Path


def recorded_tool_rounds(connection):
    """Count native history once, including turns hidden by context compaction."""
    seen = set()
    used = 0
    rows = connection.execute("SELECT id,tool_calls FROM messages WHERE role='assistant' "
                              "AND COALESCE(inherited_from_previous_session,0)=0 ORDER BY id")
    for mid, serialized in rows:
        calls = json.loads(serialized or '[]')
        if not calls:
            continue
        key = calls[0].get('id') or ('message', mid)
        if key not in seen:
            seen.add(key)
            used += 1
    return used


class ToolRoundBudget:
    def __init__(self, path, limit, initial_used=0):
        self.path = Path(path)
        self.limit = int(limit)
        if self.limit < 1 or initial_used < 0:
            raise ValueError('Tool-round limit must be positive and usage nonnegative')
        prior = json.loads(self.path.read_text()) if self.path.exists() else {}
        if prior and prior['limit'] != self.limit:
            raise ValueError('Tool-round limit changed on resume')
        self.used = max(int(initial_used), int(prior.get('used', 0)))
        self.final_attempted = False
        self.announced = False
        self.save()

    @property
    def exhausted(self):
        return self.used >= self.limit

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'limit': self.limit, 'used': self.used,
                                   'unit': 'assistant tool rounds, including context edits'}))
        tmp.replace(self.path)

    def reserve(self):
        if self.exhausted:
            return False
        self.used += 1
        # Reserve before executing: a cancelled action must not regain its budget.
        self.save()
        return True


def continue_run(agent, step, total_iters, max_total_iters):
    budget = getattr(agent, 'glasseek_tool_round_budget', None)
    if budget is not None:
        return not budget.final_attempted
    return step < agent.max_steps and total_iters < max_total_iters


def prepare_turn(agent, messages):
    budget = getattr(agent, 'glasseek_tool_round_budget', None)
    if budget is not None and not budget.announced and not budget.exhausted:
        budget.announced = True
        messages.append({'role': 'user', 'content':
            f'Tool-round budget: {budget.used}/{budget.limit} already used across this task and any resumes; '
            f'{budget.limit-budget.used} remain. Each tool-call turn, including context-editing bash, '
            'uses one round. Finish with a direct assistant answer before exhausting the budget.'})
    if budget is not None and budget.exhausted:
        budget.final_attempted = True
        messages.append({'role': 'user', 'content':
            f'The cumulative tool-round limit ({budget.used}/{budget.limit}) has been reached. '
            'No more tools may execute. Give your final answer now directly in an assistant '
            'message without tool calls, using the requested answer format and existing evidence.'})


def admit_tool_turn(agent, message, messages):
    budget = getattr(agent, 'glasseek_tool_round_budget', None)
    if budget is None or budget.reserve():
        return True
    # Keep sampled output intact; reject execution rather than rewriting the action.
    messages.append(message)
    for call in message.get('tool_calls') or []:
        messages.append({'role': 'tool', 'tool_call_id': call['id'],
                         'content': 'Tool round limit reached; this action was not executed.'})
    agent.glasseek_completion['stop_reason'] = 'tool_round_limit'
    return False
