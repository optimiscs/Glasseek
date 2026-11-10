"""Strict offline release gate using the unchanged runtime XML protocol."""
from glasseek_shared.trajectory_release import validate_messages
from audit_xml import syntax_check


def check(item, contract):
    allowed = {name: schema for name, schema in contract['schemas'].items()
               if item.get('agent_role') != 'child'
               or name not in contract['delegation']['blocked_child_tools']}
    errors = []
    metrics = {}
    try:
        validate_messages(item['messages'], allowed)
        metrics = syntax_check(item['messages'])
    except ValueError as exc:
        errors.append('protocol.' + str(exc))
    blocked = {'target_call_response_alignment_required',
               'unverified_call_response_association', 'tool_prompt_protocol_review_required',
               'linked_report_final_mismatch'}
    errors.extend('protocol.' + flag for flag in blocked.intersection(item.get('review_flags', [])))
    errors.extend('mapping.' + rule for rule in item.get('migration_rules', [])
                  if rule.startswith('argument.') or rule == 'unsupported_tool')
    return sorted(set(errors)), metrics
