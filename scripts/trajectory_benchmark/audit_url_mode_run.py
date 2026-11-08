"""Audit completed native trajectories without issuing model or web requests."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = re.compile(r'(?:U[1-9]\d*|D[1-9]\d*@v[1-9]\d*(?:#E[1-9]\d*)?)')
XML = re.compile(r'<url>\s*(' + REFERENCE.pattern + r')\s*</url>')


def arguments(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return json.loads(value.replace('\\"', '"'))
    return value


def audit(mode, case, messages, sessions, registries):
    counts = Counter()
    calls = []
    seen = set()
    visible = {}
    visible_text = {}
    results = {}
    for msg in messages:
        if msg['role'] == 'tool' and msg.get('tool_call_id'):
            results[msg['tool_call_id']] = msg.get('content') or ''
    for msg in messages:
        sid = msg['session_id']
        known = visible.setdefault(sid, set())
        texts = visible_text.setdefault(sid, [])
        if msg['role'] == 'user':
            texts.append(msg.get('content') or '')
        if msg['role'] == 'tool':
            known.update(REFERENCE.findall(msg.get('content') or ''))
            texts.append(msg.get('content') or '')
            continue
        if msg['role'] != 'assistant' or msg.get('inherited_from_previous_session'):
            continue
        for call in arguments(msg.get('tool_calls') or '[]'):
            cid = call.get('id') or call.get('call_id')
            if cid in seen:
                continue
            seen.add(cid)
            fn = call.get('function', call)
            name = fn['name']
            counts['tool_calls'] += 1
            counts[name + '_calls'] += 1
            if name != 'web_extract':
                continue
            item = {'message_id': msg['id'], 'session_id': sid, 'call_id': cid,
                    'raw_arguments': fn.get('arguments'), 'targets': []}
            try:
                args = arguments(fn.get('arguments') or '{}')
                values = args.get('urls', args.get('url', []))
                if isinstance(values, str):
                    values = arguments(values) if values.lstrip().startswith('[') else [values]
                for value in values:
                    counts['extract_targets'] += 1
                    target = {'value': value}
                    wrapped = XML.fullmatch(value) if isinstance(value, str) else None
                    if isinstance(value, str) and (REFERENCE.fullmatch(value) or wrapped):
                        ref = wrapped[1] if wrapped else value
                        base = ref.split('#E')[0]
                        kind = 'xml_argument' if wrapped else 'bare_id_argument'
                        counts[kind] += 1
                        target.update(kind=kind, previously_observed=base in known)
                        if base not in known:
                            counts['id_not_in_previous_observations'] += 1
                        reg = registries.get(sid, {})
                        if base.startswith('D'):
                            counts['snapshot_reads'] += 1
                            if '#E' in ref:
                                counts['span_reads'] += 1
                            base = reg.get('documents', {}).get(base, {}).get('url_ref', base)
                        target['resolved_url'] = reg.get('entries', {}).get(base, {}).get('url')
                        if any(mark in (target['resolved_url'] or '') for mark in ('](http', ')[', ')![', '<url>')):
                            counts['suspicious_registered_url'] += 1
                            target['mapping_warning'] = 'Possible Markdown fragments inside registered address; inspect source observation'
                    elif isinstance(value, str) and value.startswith(('http://', 'https://')):
                        counts['literal_url_argument'] += 1
                        target['kind'] = 'literal_url'
                        reg = registries.get(sid, {})
                        addresses = {value['url'] for ref, value in reg.get('entries', {}).items() if ref in known}
                        appeared = value in addresses or any(value in text for text in texts)
                        target['previously_observed_literal'] = appeared
                        if not appeared:
                            counts['literal_not_previously_observed'] += 1
                        embedded = re.findall(r'\bU[1-9]\d*\b', value)
                        if embedded:
                            counts['id_embedded_in_url'] += 1
                            target['embedded_ids'] = embedded
                    else:
                        counts['other_argument'] += 1
                        target['kind'] = 'other'
                    item['targets'].append(target)
            except (ValueError, TypeError, AttributeError) as exc:
                counts['argument_parse_error'] += 1
                item['parse_error'] = str(exc)
            response = results.get(cid, '')
            item['response_excerpt'] = response[:600]
            if not response:
                counts['missing_extract_response'] += 1
            if 'url_reference_error' in response:
                counts['url_reference_errors'] += 1
            try:
                parsed = json.loads(response[response.find('{'):response.rfind('}') + 1])
                item['page_results'] = []
                for page in parsed.get('results', []):
                    counts['extract_page_results'] += 1
                    if page.get('error'):
                        counts['extract_page_errors'] += 1
                    item['page_results'].append({
                        'url': page.get('url'), 'error': page.get('error'),
                        'content_chars': len(page.get('content') or ''),
                        'content_excerpt': (page.get('content') or '')[:240],
                        'cached': page.get('cached', False), 'truncated': page.get('truncated', False)})
            except (ValueError, TypeError):
                pass
            calls.append(item)
    return {'mode': mode, 'question_id': case['question_id'], 'question': case['question'],
            'reference_answer': case['answer'], 'final_answer': case.get('model_response', case.get('final')),
            'status': case.get('status'), 'trace_complete': case.get('trace_complete', True),
            'duration_seconds': case['duration_seconds'],
            'uncached_input_tokens': sum(s.get('input_tokens', 0) or 0 for s in sessions),
            'input_tokens_including_cache': sum((s.get('input_tokens', 0) or 0) + (s.get('cache_read_tokens', 0) or 0) + (s.get('cache_write_tokens', 0) or 0) for s in sessions),
            'output_tokens': sum(s.get('output_tokens', 0) or 0 for s in sessions),
            'cache_read_tokens': sum(s.get('cache_read_tokens', 0) or 0 for s in sessions),
            'counts': dict(counts), 'extract_calls': calls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    state = json.loads((args.run_dir / 'status.json').read_text())
    old, new = [], []
    ids = state['question_ids']
    for mode, run in state['runs'].items():
        for qid in ids:
            record = json.loads((args.baseline / (mode + '-' + qid + '.json')).read_text())
            regs = dict(zip([s['id'] for s in record['sessions']], record.get('registries', [])))
            old.append(audit(mode, record, record['messages'], record['sessions'], regs))
        folder = ROOT / 'Evaluation/output/preds' / run['run'] / state['dataset']
        for path in sorted(folder.glob('conv/**/*.json')):
            case = json.loads(path.read_text())
            dbpath = ROOT / 'Evaluation/data/runtime/hermes/profiles' / case['profile'] / 'state.db'
            with sqlite3.connect('file:' + str(dbpath) + '?mode=ro', uri=True) as db:
                db.row_factory = sqlite3.Row
                sessions = [dict(r) for r in db.execute('SELECT * FROM sessions')]
                messages = [dict(r) for r in db.execute('SELECT * FROM messages ORDER BY timestamp,id')]
            registries = {}
            for session in sessions:
                snapshot = Path(run['url_state_dir']) / (hashlib.sha256(session['id'].encode()).hexdigest() + '.json')
                if snapshot.exists():
                    registries[session['id']] = json.loads(snapshot.read_text())
            row = audit(mode, case, messages, sessions, registries)
            row['conversation_path'] = str(path)
            row['db_path'] = str(dbpath)
            new.append(row)
    report = {'scope': 'Two paired questions, four modes, one fresh trajectory per condition',
              'limitations': ['Dynamic web results and sampling differ; this is not a controlled causal estimate.',
                              'Input tokens include repeated history and cached input, not billed uncached input.',
                              'Correction to earlier audit: Hermes sessions.input_tokens excludes cache reads/writes; total input is their sum.',
                              'Successful completion is not answer correctness. Review final answers and supporting observations.',
                              'vLLM limit is 16; only eight top-level trajectories exist in this subset.'],
              'baseline': old, 'new': new, 'completed_new': len(new), 'expected_new': len(ids) * len(state['runs'])}
    target = args.run_dir / 'comparison.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2));temporary.replace(target)
    print(json.dumps({'completed': len(new), 'expected': report['expected_new'],
                      'new': [{k: r[k] for k in ('mode', 'question_id', 'input_tokens_including_cache', 'output_tokens', 'counts')} for r in new]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
