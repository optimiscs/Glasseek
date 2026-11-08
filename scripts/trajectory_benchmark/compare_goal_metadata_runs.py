"""Compare completed offline releases; audit goal/telemetry without model calls."""
import argparse
import collections
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trajectory_preprocess'))
from runtime_metadata import clean_response, TOOLS, VERSION
from migration_render import target_messages


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csv_write(path, records):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def held_index(root):
    reasons = collections.defaultdict(collections.Counter)
    categories = collections.defaultdict(collections.Counter)
    index = {}
    for r in rows(root / 'held.jsonl'):
        key = r.get('source', r.get('agent_role'))
        reasons[key].update(set(r['reasons']))
        categories[key].update(set(r.get('categories', [])))
        index[r['id']] = (key, r['reasons'])
    return reasons, categories, index


def transitions(before, after):
    br, bc, bi = held_index(before)
    ar, ac, ai = held_index(after)
    bd = {r['id'] for r in rows(before / 'duplicates.jsonl')}
    ad = {r['id'] for r in rows(after / 'duplicates.jsonl')}
    gained = collections.defaultdict(collections.Counter)
    lost = collections.defaultdict(collections.Counter)
    examples = {'held_to_candidate': [], 'candidate_to_held': []}
    for ident in bi.keys() - ai.keys() - ad:
        source, reasons = bi[ident]
        gained[source]['records'] += 1
        gained[source].update(reasons)
        if len(examples['held_to_candidate']) < 30:
            examples['held_to_candidate'].append({'id': ident, 'source': source, 'old_reasons': reasons})
    for ident in ai.keys() - bi.keys() - bd:
        source, reasons = ai[ident]
        lost[source]['records'] += 1
        lost[source].update(reasons)
        if len(examples['candidate_to_held']) < 30:
            examples['candidate_to_held'].append({'id': ident, 'source': source, 'new_reasons': reasons})
    return {'before_reasons_overlapping': br, 'after_reasons_overlapping': ar,
            'before_categories_overlapping': bc, 'after_categories_overlapping': ac,
            'held_to_candidate': gained, 'candidate_to_held': lost,
            'new_exact_duplicate_ids': len(ad - bd), 'no_longer_exact_duplicate_ids': len(bd - ad),
            'examples': examples}


def candidate_audit(paths):
    by_source = collections.defaultdict(collections.Counter)
    fields = collections.Counter()
    skipped = []
    for path in paths:
        for i, item in enumerate(rows(path), 1):
            source = item['source'] + (':' + item['agent_role'] if 'agent_role' in item else '')
            count = by_source[source]
            count['candidates'] += 1
            assert item['training_ready'] is False
            has_goal = False
            for step in item['steps']:
                calls = step.get('calls', [])
                for call in calls:
                    if call['name'] == 'web_extract':
                        count['extract_calls'] += 1
                        if 'goal' in call['arguments']:
                            assert isinstance(call['arguments']['goal'], str)
                            count['extract_calls_with_goal'] += 1
                            has_goal = True
                if len(calls) == 1 and calls[0]['name'].casefold() in TOOLS:
                    cleaned, event = clean_response(step['observation'])
                    assert cleaned == step['observation'], (item['id'], 'uncleaned runtime metadata')
                    if event and event['status'] == 'skipped':
                        count['unparseable_delegate_responses'] += 1
                        skipped.append({'id': item['id'], 'source': source, 'step_id': step['step_id'], **event})
            count['trajectories_with_goal'] += int(has_goal)
            events = item.get('runtime_metadata_cleaning', {}).get('events', [])
            cleaned_events = [e for e in events if e['status'] == 'cleaned']
            if cleaned_events:
                count['metadata_cleaned_trajectories'] += 1
                count['metadata_cleaned_responses'] += len(cleaned_events)
                for event in cleaned_events:
                    count['metadata_chars_removed'] += event['chars_removed']
                    fields.update(p.rsplit('.', 1)[-1] for p in event['removed_paths'])
            if i % 20000 == 0:
                print('CANDIDATES_CHECKED', path.name, i, flush=True)
    return {'status': 'passed', 'policy': VERSION, 'by_source': by_source,
            'removed_fields_in_candidates': fields, 'unparseable_delegate_responses': skipped,
            'scope': 'Every candidate: goal type and idempotent delegate telemetry cleanup. Unparseable responses remain explicitly reviewable; original source archives retained.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', type=Path, required=True)
    ap.add_argument('--after', type=Path, required=True)
    ap.add_argument('--migration-before', type=Path, required=True)
    ap.add_argument('--migration-after', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    old, new = [read(p / 'cleaning_summary.json') for p in (args.before, args.after)]
    bm, am = [read(p / 'manifest.json') for p in (args.before, args.after)]
    assert bm['inputs'] == am['inputs'], 'Input inventory changed; counts alone are not comparable'
    assert bm['config']['max_sample_chars'] == am['config']['max_sample_chars']
    mo, mn = [read(p / 'summary.json') for p in (args.migration_before, args.migration_after)]
    for s in (old, new, mo, mn):
        assert s['status'] == 'completed'
    assert read(args.after / 'audit/tool_contract_checks.json')['status'] == 'passed'
    assert read(args.migration_after / 'audit.json')['status'] == 'passed'
    assert read(args.migration_after / 'comparison/mask_checks.json')['status'] == 'passed'
    comparison = []
    for source, n in new['sources'].items():
        o = old['sources'][source]
        assert n['scanned'] == o['scanned']
        assert n['scanned'] == sum(n.get(k, 0) for k in ('eligible', 'held', 'exact_duplicates'))
        r = {'source': source, 'read': n['scanned']}
        for label, s in (('before', o), ('after', n)):
            r.update({label + '_' + k: s.get(k, 0) for k in ('eligible', 'held', 'exact_duplicates', 'near_duplicate_flagged')})
            r[label + '_pass_rate_percent'] = round(100 * s.get('eligible', 0) / s['scanned'], 4)
        r['candidate_delta'] = n.get('eligible', 0) - o.get('eligible', 0)
        comparison.append(r)
    migrations = []
    for role, denom in (('parent', 'bundles'), ('child', 'children_read')):
        assert mo['counts'][denom] == mn['counts'][denom]
        r = {'role': role, 'read': mn['counts'][denom]}
        for label, s in (('before', mo), ('after', mn)):
            c = s['counts']
            assert c[denom] == sum(c.get(role + '_' + k, 0) for k in ('candidates', 'held', 'exact_duplicates'))
            r.update({label + '_' + k: c.get(role + '_' + k, 0) for k in ('candidates', 'held', 'exact_duplicates')})
            r[label + '_pass_rate_percent'] = round(100 * c[role + '_candidates'] / c[denom], 4)
        migrations.append(r)
    csv_write(out / 'dataset_comparison.csv', comparison)
    csv_write(out / 'searchswarm_comparison.csv', migrations)
    write(out / 'held_reasons_and_transitions.json', transitions(args.before, args.after))
    write(out / 'searchswarm_reasons_and_transitions.json', transitions(args.migration_before, args.migration_after))
    audit = candidate_audit([args.after / 'clean.jsonl', args.migration_after / 'parent_candidates.jsonl', args.migration_after / 'child_candidates.jsonl'])
    write(out / 'goal_metadata_checks.json', audit)
    # Every exported Parquet row must match its canonical candidate rendering.
    import pyarrow.parquet as pq
    candidates = itertools.chain(rows(args.migration_after / 'parent_candidates.jsonl'), rows(args.migration_after / 'child_candidates.jsonl'))
    n = 0
    for batch in pq.ParquetFile(args.migration_after / 'review_sft.parquet').iter_batches(batch_size=32, columns=['id', 'messages', 'training_ready', 'tools']):
        for row in batch.to_pylist():
            item = next(candidates)
            assert row['id'] == item['id'] and row['training_ready'] is False
            assert json.loads(row['messages']) == target_messages(item)[0]
            extract = next(t['function'] for t in json.loads(row['tools']) if t['function']['name'] == 'web_extract')
            assert extract['parameters']['properties']['goal']['type'] == 'string'
            n += 1
        if n % 8000 == 0:
            print('PARQUET_CHECKED', n, flush=True)
    assert next(candidates, None) is None
    assert n == read(args.migration_after / 'export_summary.json')['total']
    write(out / 'comparison.json', {'status': 'passed', 'datasets': comparison, 'searchswarm': migrations,
        'before_counts': old['counts'], 'after_counts': new['counts'], 'parquet_rows_verified': n,
        'input_inventory_unchanged': True, 'annotation_char_budget': am['config']['max_sample_chars'],
        'processing_seconds': {'full_before': old['elapsed_seconds'], 'full_after': new['elapsed_seconds'],
                               'migration_before': mo['elapsed_seconds'], 'migration_after': mn['elapsed_seconds']},
        'paths': {k: str(v.resolve()) for k, v in vars(args).items()},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limits': ['Static compatibility candidates, not factual/LLM approval; training_ready=false.',
                   'Full intake counts SearchSwarm bundles as adapter holds; expanded migration is separate.',
                   'Near duplicates are flags, not removals. Migration has exact dedup only; no union-wide dedup.',
                   'Hold reasons overlap. Full intake applies the existing 250k-character annotation budget; migration only records it diagnostically.',
                   'No live replay, no model API calls, no full-corpus tokenizer-length screening.']})
    print('COMPARISON_COMPLETE', out, flush=True)
    from unify_trajectory_candidates import build_release
    build_release(args.after, args.migration_after, args.after.parent / (args.after.name + '-unified'))


if __name__ == '__main__':
    main()
