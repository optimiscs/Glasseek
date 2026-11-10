"""Whole-message supervision: no deletion, XML splitting, or model-written masks."""
import hashlib
import json

VERSION = 'assistant-message-supervision-v1'


def message_hash(messages):
    return hashlib.sha256(json.dumps(messages,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def validate_plan(plan, messages, *, require_learn=False):
    if plan.get('version') != VERSION or plan.get('messages_sha256') != message_hash(messages):
        raise ValueError('supervision_content_mismatch')
    rows = plan.get('messages', [])
    if len(rows) != len(messages): raise ValueError('supervision_coverage_mismatch')
    for i, (row, message) in enumerate(zip(rows,messages)):
        if row.get('index') != i or row.get('decision') not in ('learn','context_only','review'):
            raise ValueError('invalid_supervision_decision')
        if message['role'] != 'assistant' and row['decision'] != 'context_only':
            raise ValueError('non_assistant_supervision_forbidden')
        if row['decision'] == 'learn' and not row.get('segment_id'):
            raise ValueError('supervision_missing_segment')
    if require_learn and (any(r['decision']=='review' for r in rows) or not any(r['decision']=='learn' for r in rows)):
        raise ValueError('supervision_unresolved_or_empty')
    return plan


def build_plan(messages, decisions):
    rows = [{'index':i, 'decision':'context_only', 'segment_id':None} for i in range(len(messages))]
    for i, value in decisions.items():
        if type(i) is not int or not 0 <= i < len(messages) or messages[i]['role'] != 'assistant':
            raise ValueError('supervision_invalid_assistant_index')
        rows[i].update(value)
    plan = {'version':VERSION,'messages_sha256':message_hash(messages),'messages':rows}
    return validate_plan(plan,messages)
