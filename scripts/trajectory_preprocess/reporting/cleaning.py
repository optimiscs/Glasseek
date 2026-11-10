"""Account for every raw trajectory after strict offline cleaning.

Rule memberships overlap. A rule's migrated count includes only final exported
trajectories; held includes review and quarantine, and duplicates are separate.
"""
import collections
import csv
import json
from pathlib import Path
import sqlite3

from client import digest, save


def rows(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def csvout(path, values):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)


def report(run):
    intake = json.loads((run/'intake/cleaning_summary.json').read_text())
    swarm = json.loads((run/'searchswarm/summary.json').read_text())
    export = json.loads((run/'safety/export/summary.json').read_text())
    assert intake['status'] == swarm['status'] == export['status'] == 'completed'
    assert export['strict_protocol_verified']
    decisions = {}

    def add(row, state, source=None, reasons=None):
        ident = row.get('id', row.get('source_id'))
        assert ident not in decisions, ident
        decisions[ident] = {'id': ident, 'source': source or row['source'], 'state': state,
                            'agent_role': row.get('agent_role', 'parent'),
                            'migration_rules': row.get('migration_rules') or ['unparsed'],
                            'reasons': sorted(set(reasons or row.get('reasons', [])))}

    for directory, default_source in [('intake', None), ('searchswarm', 'SearchSwarm')]:
        for filename, state in [('held', 'held'), ('duplicates', 'duplicate')]:
            for row in rows(run/directory/(filename+'.jsonl')):
                add(row, state, default_source, ['exact_duplicate'] if state == 'duplicate' else None)
    for row in rows(run/'unified/exact_duplicates.jsonl'):
        add(row, 'duplicate', reasons=['exact_duplicate'])

    withheld = {row['source_row']: row for row in rows(run/'safety/export/withheld.jsonl')}
    db = sqlite3.connect('file:'+str((run/'safety/state.sqlite').resolve())+'?mode=ro', uri=True)
    scanner_rows = iter(db.execute('SELECT result FROM decisions ORDER BY row_index'))
    count = 0
    for index, item in enumerate(rows(run/'unified/candidates.jsonl')):
        scanned = json.loads(next(scanner_rows)[0])
        assert scanned['source_row'] == index and scanned['content_hash'] == digest(item)
        held = withheld.get(index)
        state = ('quarantine' if held['state'] == 'quarantine' else 'held') if held else 'migrated'
        reasons = [finding['rule_id'] for finding in scanned['findings']]
        if held: reasons.extend(held.get('reasons', []))
        add(item, state, reasons=reasons)
        count += 1
    assert next(scanner_rows, None) is None
    db.close()
    assert count == export['input_records']

    sources = collections.defaultdict(collections.Counter)
    rules = collections.defaultdict(collections.Counter)
    reason_counts = collections.Counter()
    for row in decisions.values():
        source, state = row['source'], row['state']
        sources[source]['read'] += 1; sources[source][state] += 1
        for rule in set(row['migration_rules']):
            rules[source, rule][state] += 1
        for reason in row['reasons']:
            reason_counts[source, reason] += 1
    table = []
    for source, counts in sorted(sources.items()):
        expected = (swarm['counts']['bundles']+swarm['counts']['children_read']
                    if source == 'SearchSwarm' else intake['sources'][source]['scanned'])
        assert counts['read'] == expected, (source, counts['read'], expected)
        table.append({'dataset': source, **{k: counts[k] for k in
                     ('read', 'migrated', 'held', 'quarantine', 'duplicate')}})
    assert sum(row['migrated'] for row in table) == export['exported']
    rule_names = sorted({rule for _, rule in rules} | {
        'search', 'extract', 'browser.open', 'code', 'subagent',
        'argument.source', 'argument.parse_type', 'argument.other', 'unsupported_tool', 'unparsed'})
    cells = [{'dataset': source, 'rule': rule, 'migrated': rules[source, rule]['migrated'],
              'held': rules[source, rule]['held']+rules[source, rule]['quarantine'],
              'quarantine': rules[source, rule]['quarantine'],
              'duplicate': rules[source, rule]['duplicate']}
             for source in sorted(sources) for rule in rule_names]
    matrix = [{'dataset': source, **{rule: str(rules[source, rule]['migrated'])+'/'+
               str(rules[source, rule]['held']+rules[source, rule]['quarantine'])
               for rule in rule_names}} for source in sorted(sources)]
    destination = run/'report'; destination.mkdir(exist_ok=True)
    final = run/'final'; final.mkdir(exist_ok=True)
    with (final/'decisions.jsonl').open('w') as output:
        for row in decisions.values(): output.write(json.dumps(row, ensure_ascii=False)+'\n')
    candidate_file = export.get('candidate_file', 'candidates.jsonl')
    passed = final/candidate_file.replace('candidates', 'offline_passed', 1)
    if not passed.exists(): passed.symlink_to('../safety/export/'+candidate_file)
    csvout(destination/'by_dataset.csv', table)
    csvout(destination/'rule_counts.csv', cells)
    csvout(destination/'rule_matrix.csv', matrix)
    reason_table = [{'dataset': source, 'reason': reason, 'trajectories': n}
                    for (source, reason), n in sorted(reason_counts.items())]
    if reason_table: csvout(destination/'reasons.csv', reason_table)
    summary = {'status': 'completed', 'by_dataset': table, 'rule_counts': cells,
               'totals': {k: sum(row[k] for row in table) for k in
                          ('read', 'migrated', 'held', 'quarantine', 'duplicate')},
               'count_unit': 'trajectory; rule columns overlap; migrated means final export, held includes quarantine; duplicates separate',
               'unparsed': 'Normalization failures without recoverable tool membership; not attributed to individual tool rules.',
               'raw_searchswarm_bundles': swarm['counts']['bundles'],
               'expanded_searchswarm_children': swarm['counts']['children_read'],
               'training_ready': False, 'runtime_replayed': False, 'llm_calls': 0,
               'excluded_sources': ['OpenResearcher'], 'protocol_metrics': export['protocol_metrics']}
    save(destination/'summary.json', summary)
    manifest = json.loads((run/'manifest.json').read_text())
    manifest.update(status='completed', totals=summary['totals'])
    save(run/'manifest.json', manifest)
    print(json.dumps({'totals': summary['totals'], 'by_dataset': table}, ensure_ascii=False))
    return summary
