"""Conservative post-judgment candidate normalization; never grants acceptance.

Only presentation and exact aliases are rewritten. Calls, evidence bodies, source
indices and prior supervision decisions are retained. Ambiguous boundaries and
unmapped native citations stay quarantined for source-level repair.
"""
import argparse
import collections
import copy
import hashlib
import json
import re
from pathlib import Path

from glasseek_shared.curation.supervision import message_hash, validate_plan
from glasseek_shared.trajectory_release import digest, payload, validate_messages, runtime_schemas

VERSION = 'training-text-v5'
ALIASES = {'functions.search': 'web_search', 'functions.visit': 'web_extract',
           'functions.visit_urls': 'web_extract'}
RETURNS = {'search': 'web_search', 'visit_urls': 'web_extract'}
PROTECTED = re.compile(
    r'```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)|`[^`\n]*(?:`|(?=\n|\Z))'
    r'|"[^"\n]*"|(?<!\w)\'[^\'\n]*\'(?!\w)|(?m:^[ \t]*>[^\n]*)')
NATIVE_CITATION = re.compile(r'【\d+†L\d+(?:-L?\d+)?】')
ANSWER_TAG = re.compile(r'</?(?:answer|explanation)>')
FINAL_PREFIX = re.compile(r'(?im)^(?P<space>[ \t]*)(?:\*\*)?FINAL ANSWER\s*[:：](?:\*\*)?[ \t]*')
# These are this normalizer's obsolete flags, not safety/protocol dispositions.
OBSOLETE_FLAGS = {
    'historical_research_state_requires_provenance', 'explicit_answer_format_requires_review',
    'final_answer_envelope_ambiguous', 'explanation_envelope_ambiguous',
    'explicit_final_answer_format_conflict', 'empty_think_before_tool_call',
}


def is_observation(message):
    return message['role'] == 'tool' or message['content'].lstrip().startswith('<tool_response>')


def _unprotected_sub(pattern, replacement, text):
    """Match against the original text so splitting cannot create line starts."""
    spans = [(m.start(), m.end()) for m in PROTECTED.finditer(text)]
    def replace(match):
        if any(a < match.end() and match.start() < b for a, b in spans):
            return match.group()
        return replacement(match) if callable(replacement) else match.expand(replacement)
    return re.sub(pattern, replace, text)


def _thought_tags(text):
    # Tags quoted in code or passed as tool arguments are not control markup.
    protected = [(m.start(), m.end()) for m in PROTECTED.finditer(text)]
    protected.extend((m.start(), m.end()) for m in re.finditer(r'<tool_call>[\s\S]*?</tool_call>', text))
    return [m for m in re.finditer(r'</?think>', text)
            if not any(a <= m.start() < b for a, b in protected)]


def _thought_well_formed(text):
    opened = False
    for tag in _thought_tags(text):
        if (tag.group() == '<think>') == opened:
            return False
        opened = tag.group() == '<think>'
    return not opened


def _rename_thought_aliases(text):
    # The caller has checked the boundaries. Scan control tokens rather than a
    # non-greedy regex: a quoted </think> must not end the actual reasoning span.
    tags = _thought_tags(text)
    for start, end in reversed(list(zip(tags[::2], tags[1::2]))):
        body = text[start.end():end.start()]
        for old, new in ALIASES.items():
            body = _unprotected_sub(r'(?<![\w.])' + re.escape(old) + r'(?!\w|\.\w)', new, body)
        text = text[:start.end()] + body + text[end.start():]
    return text


def _repair_thought(text, linked, protocol_ok):
    if _thought_well_formed(text):
        return text, None
    opener = re.match(r'\s*(<think>)\s*(?=<tool_call>)', text)
    if (opener and linked and protocol_ok
            and text.count('<think>') == 1 and '</think>' not in text
            and re.fullmatch(r'(?:<tool_call>[\s\S]*?</tool_call>\s*)+', text[opener.end():])):
        return text[:opener.start(1)] + text[opener.end(1):], 'empty_think_residue'
    duplicate = re.match(r'\s*(<think>)\s*(?=<think>)', text)
    if (duplicate and linked and protocol_ok
            and text.count('<think>') == 2 and text.count('</think>') == 1):
        fixed = text[:duplicate.start(1)] + text[duplicate.end(1):]
        if (_thought_well_formed(fixed)
                and re.fullmatch(r'\s*<think>[\s\S]*?</think>\s*(?:<tool_call>[\s\S]*?</tool_call>\s*)+', fixed)):
            return fixed, 'duplicate_think_opening'
    return text, None


def _rebind_supervision(original, out, kept, events, holds):
    """Reindex existing decisions; never turn previously masked tokens into targets."""
    if 'message_loss_mask' in original:
        mask = original['message_loss_mask']
        if (not isinstance(mask, list) or len(mask) != len(original['messages'])
                or any(v not in (0, 1) or (v and m['role'] != 'assistant')
                       for v, m in zip(mask, original['messages']))):
            holds.append('invalid_message_loss_mask')
        else:
            out['message_loss_mask'] = [mask[i] for i in kept]
            if kept != list(range(len(original['messages']))):
                events.append({'rule': 'message_loss_mask_rebound'})
    if original.get('supervision') is not None:
        try:
            plan = original['supervision']
            plan = json.loads(plan) if isinstance(plan, str) else plan
            validate_plan(plan, original['messages'])
            rebound = copy.deepcopy(plan)
            rebound['messages'] = [dict(plan['messages'][old], index=new) for new, old in enumerate(kept)]
            rebound['messages_sha256'] = message_hash(out['messages'])
            validate_plan(rebound, out['messages'])
            out['supervision'] = rebound
            if rebound != plan:
                events.append({'rule': 'supervision_rebound'})
        except (ValueError, TypeError, KeyError, AttributeError):
            holds.append('supervision_requires_rebinding')


def _normalize_final(out, instructions, unsafe_messages, events, holds):
    messages = out['messages']
    if not messages or messages[-1]['role'] != 'assistant':
        return
    mi = len(messages) - 1
    if mi in unsafe_messages:
        return
    message = messages[-1]
    text = message['content']
    prefix = re.match(r'\s*<think>.*?</think>', text, re.S)
    offset = prefix.end() if prefix else 0
    head, visible = text[:offset], text[offset:]
    protected = list(PROTECTED.finditer(visible))
    citations = [m.group() for m in NATIVE_CITATION.finditer(visible)
                 if not any(p.start() <= m.start() < p.end() for p in protected)]
    # The canonical protocol currently has no verified document-ID/line mapping.
    # URLs or source_message_index alone cannot resolve native line citations.
    if citations:
        holds.append('citation_source_mapping_unverified')
    if any(re.search(r'</?(?:answer|explanation|report)>', p.group()) for p in protected):
        holds.append('literal_answer_tag_requires_review')
        return
    if any(t in visible for t in ('<tool_call>', '<think>', '</think>')):
        holds.append('final_answer_boundary_ambiguous')
        return
    child = out.get('agent_role') == 'child'
    required_report = child and any('<report>' in t for t in instructions)
    if child and not required_report:
        return  # The child contract is not the parent's answer contract.
    target = 'report' if required_report else 'answer'
    tags = re.compile(r'</?(?:answer|explanation|report)>') if required_report else ANSWER_TAG
    envelope = re.fullmatch(r'(\s*<' + target + r'>)([\s\S]*?)(</' + target + r'>\s*)', visible)
    canonical = envelope is not None and not tags.search(envelope[2])
    body = envelope[2] if canonical else tags.sub('', visible)
    # A requested heading/prefix is part of the answer body, not disposable UI.
    if not child and not any('### Prediction' in t for t in instructions):
        heading = re.match(r'\A(\s*)### Prediction[ \t]*\r?\n', body)
        if heading:
            body = heading[1] + body[heading.end():]
            events.append({'rule': 'prediction_heading', 'message': mi})
    required_prefix = any(re.search(r'FINAL ANSWER\s*[:：]', t, re.I) for t in instructions)
    if not child and required_prefix:
        if not FINAL_PREFIX.search(body):
            holds.append('required_final_answer_prefix_missing')
    elif not child:
        stripped = _unprotected_sub(FINAL_PREFIX, r'\g<space>', body)
        if stripped != body:
            events.append({'rule': 'final_answer_prefix', 'message': mi})
        body = stripped
    if canonical:
        message['content'] = head + envelope[1] + body + envelope[3]
    else:
        message['content'] = head + '<' + target + '>' + body + '</' + target + '>'
        events.append({'rule': 'required_report_envelope' if required_report else 'single_answer_envelope',
                       'message': mi})


def normalize(row, *, tool_schemas=None):
    out = copy.deepcopy(row)
    events, holds, calls = [], [], {}
    if ((row.get('safety') and row['safety'].get('state') != 'pass_rules')
            or 'upstream_safety_not_passed' in row.get('text_normalization', {}).get('holds', [])):
        holds.append('upstream_safety_not_passed')
    tools = runtime_schemas() if tool_schemas is None else tool_schemas
    try:
        validate_messages(row['messages'], tools)
        protocol_ok = True
    except ValueError:
        protocol_ok = False
    kept, messages = [], []
    for old_index, message in enumerate(out['messages']):
        if (message['role'] == 'user' and not is_observation(message) and messages
                and messages[-1]['role'] == 'user' and not is_observation(messages[-1])
                and message['content'].startswith('Solve the following task:')
                and message['content'][len('Solve the following task:'):].strip()
                    == messages[-1]['content'].strip() and messages[-1]['content'].strip()):
            events.append({'rule': 'duplicate_adjacent_question', 'source_message': old_index,
                           'retained_source_message': kept[-1]})
            continue
        kept.append(old_index)
        messages.append(message)
    out['messages'] = messages
    unsafe_messages = set()
    instructions = [m['content'] for m in messages if m['role'] in ('system', 'developer', 'user') and not is_observation(m)]
    for mi, message in enumerate(messages):
        text = message['content']
        if message['role'] in ('system', 'developer'):
            continue
        if message['role'] == 'assistant':
            for call in message.get('call_links', []):
                calls[call['call_id']] = call['name']
            text, repair = _repair_thought(text, message.get('call_links'), protocol_ok)
            if repair:
                events.append({'rule': repair, 'message': mi})
            if not _thought_well_formed(text):
                holds.append('unbalanced_think_boundary')
                unsafe_messages.add(mi)
                continue  # Preserve the entire ambiguous message for source-level repair.
            renamed = _rename_thought_aliases(text)
            if renamed != text:
                events.append({'rule': 'assistant_tool_reference', 'message': mi})
            text = renamed
            if re.search(r'\bfunctions\.(?:search|visit|visit_urls)\b', text):
                holds.append('legacy_tool_reference_ambiguous')
        if is_observation(message):
            match = re.match(r'(\s*<tool_response>\s*Tool:\s*)(search|visit_urls)(\s*\n)', text)
            if match:
                name = calls.get(message.get('response_to'))
                if name != RETURNS[match[2]] or message.get('tool_call_id') != message.get('response_to'):
                    holds.append('response_label_association_mismatch')
                elif any(k in message for k in ('batch_response_parts', 'merged_response_parts')):
                    holds.append('response_label_requires_offset_rebuild')
                else:
                    audit = message.get('search_return_audit')
                    if audit and audit.get('rendered_response_sha256') != hashlib.sha256(text.encode()).hexdigest():
                        holds.append('stale_search_return_audit')
                    else:
                        text = text[:match.start(2)] + name + text[match.end(2):]
                        if audit:
                            audit['rendered_response_sha256'] = hashlib.sha256(text.encode()).hexdigest()
                        events.append({'rule': 'response_tool_label', 'message': mi})
        elif message['role'] == 'user':
            if 'RESEARCH STATE SUMMARY' in text or re.search(r'\bprev_state\b', text):
                out['task_context_kind'] = 'research_continuation'
            if text.startswith('You are running one browse-comparison evaluation task.'):
                marker = re.search(r'\nTask type: [^\n]+\nQuestion:\n', text)
                if marker and text.count('\nQuestion:\n') == 1:
                    text = text[marker.end():]
                    events.append({'rule': 'benchmark_user_wrapper', 'message': mi})
                else:
                    holds.append('benchmark_wrapper_boundary_ambiguous')
        message['content'] = text
    _normalize_final(out, instructions, unsafe_messages, events, holds)
    _rebind_supervision(row, out, kept, events, holds)
    try:
        validate_messages(messages, tools)
    except ValueError as exc:
        holds.append('protocol_after_normalization:' + str(exc))
    before_hash, after_hash = digest(payload(row)), digest(payload(out))
    changed = before_hash != after_hash
    flags = set(out.get('review_flags', []))
    # Recompute our own previous findings only; upstream safety flags are retained.
    flags.difference_update(OBSOLETE_FLAGS)
    flags.difference_update(row.get('text_normalization', {}).get('holds', []))
    flags.update(holds)
    if changed:
        flags.add('normalized_content_requires_revalidation')
        for key in ('acceptance', 'content_sha256', 'payload_sha256', 'safety'):
            out.pop(key, None)
    out['training_ready'] = False
    out['harness_replayed'] = False
    out.pop('acceptance_status', None)
    out['review_flags'] = sorted(flags)
    out['text_normalization'] = {
        'version': VERSION, 'source_content_sha256': before_hash, 'content_sha256': after_hash,
        'message_index_map': kept, 'events': events, 'holds': sorted(set(holds)),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    counts, rules, holds = collections.Counter(), collections.Counter(), collections.Counter()
    tools = runtime_schemas()
    with (args.output/'clean.candidates.jsonl').open('w') as clean, (args.output/'held.candidates.jsonl').open('w') as held, (args.output/'audit.jsonl').open('w') as audit:
        for line in args.input.open():
            row = normalize(json.loads(line), tool_schemas=tools)
            info = row['text_normalization']
            status = 'held' if info['holds'] else 'clean'
            counts[status] += 1
            (held if info['holds'] else clean).write(json.dumps(row, ensure_ascii=False)+'\n')
            rules.update({e['rule'] for e in info['events']})
            holds.update(info['holds'])
            audit.write(json.dumps({'id': row['id'], 'source': row['source'], **info}, ensure_ascii=False)+'\n')
    report = {'rows': dict(counts), 'rules_rows': dict(rules), 'hold_reasons': dict(holds), 'training_ready': False}
    (args.output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
