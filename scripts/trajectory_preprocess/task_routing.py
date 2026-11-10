"""Deterministic routing; ambiguous task semantics remain explicitly unknown."""
import re


def route(item, canonical):
    steps = item['steps']
    calls = [c for s in steps for c in s.get('calls',[])]
    summary = bool(re.search(r'RESEARCH STATE SUMMARY',item['question'],re.I))
    names = {c['name'] for c in calls}
    goal = 'calculation' if names == {'execute_code'} else 'unknown'
    if names & {'web_search','web_extract'} and 'execute_code' in names: goal = 'mixed'
    return {'task_goal':goal, 'trajectory_form':'summary_continuation' if summary else 'tool_trace' if calls else 'direct_answer',
            'agent_topology':'child' if canonical.get('agent_role')=='child' else 'parent' if 'delegate_task' in names else 'single'}
