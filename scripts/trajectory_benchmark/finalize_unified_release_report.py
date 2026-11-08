"""Collect completed release counts and provenance; never scans or changes content."""
import csv
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = ROOT / 'data/processed/trajectory-glasseek-v7-unified'
    report = ROOT / 'reports/trajectory-glasseek-v7-comparison'
    unified = json.loads((out / 'summary.json').read_text())
    compared = json.loads((report / 'comparison.json').read_text())
    assert unified['status'] == 'completed' and compared['status'] == 'passed'
    full = compared['datasets']; migration = compared['searchswarm']
    table = []
    for row in full:
        name = row['source']; merged = unified['by_source'][name]
        if name == 'SearchSwarm':
            read = sum(r['read'] for r in migration)
            before = sum(r['before_candidates'] for r in migration)
            held = sum(r['after_held'] for r in migration)
            prior_duplicates = sum(r['after_exact_duplicates'] for r in migration)
            near = None
        else:
            read, before, held = row['read'], row['before_eligible'], row['after_held']
            prior_duplicates = row['after_exact_duplicates']; near = row['after_near_duplicate_flagged']
        final = merged['candidates']; extra = merged.get('exact_duplicates_removed_at_union', 0)
        assert read == final + held + prior_duplicates + extra
        table.append({'dataset': name, 'read': read, 'previous_candidates': before,
            'previous_pass_rate_percent': round(before * 100 / read, 4),
            'unified_candidates': final, 'unified_pass_rate_percent': round(final * 100 / read, 4),
            'held': held, 'prior_exact_duplicates': prior_duplicates,
            'additional_union_exact_duplicates': extra, 'near_duplicate_flags': near})
    with (out / 'dataset_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    sources = ['compare_goal_metadata_runs.py', 'unify_trajectory_candidates.py', 'finalize_unified_release_report.py',
               'audit_glasseek_contract.py', 'audit_searchswarm_migration.py', 'compare_searchswarm_glasseek.py',
               'export_migrated_review_sft.py', 'verify_migration_masks.py']
    frozen = out / 'frozen_postprocess'; frozen.mkdir(exist_ok=True)
    for name in sources:
        shutil.copyfile(Path(__file__).with_name(name), frozen / name)
    for name in ('migration_render.py', 'runtime_metadata.py'):
        shutil.copyfile(ROOT / 'scripts/trajectory_preprocess' / name, frozen / name)
    for name in ('comparison.json', 'held_reasons_and_transitions.json', 'searchswarm_reasons_and_transitions.json', 'goal_metadata_checks.json'):
        shutil.copyfile(report / name, out / name)
    release = {'status': 'completed', 'canonical_output': str(out), 'datasets': table,
        'totals': {key: sum(r[key] for r in table) for key in
                   ('read', 'previous_candidates', 'unified_candidates', 'held', 'prior_exact_duplicates', 'additional_union_exact_duplicates')},
        'goaled_extract_calls': sum(r.get('extract_calls_with_goal', 0) for r in unified['by_source'].values()),
        'safety_screened': False, 'training_ready': False,
        'notes': ['SearchSwarm read count includes 6732 parent bundles plus 41300 child trajectories; bundles are not counted twice.',
                  'No LLM annotation, factual approval, live replay or newly proposed safety scan was run.',
                  'Near duplicates are flags across normalized intake records, including held records. SearchSwarm has no near-duplicate scan.',
                  'Safety pipeline changes requested later are a separate stage; this release is its candidate input.']}
    assert release['totals']['unified_candidates'] == unified['candidates']
    (out / 'release.json').write_text(json.dumps(release, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(release, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
