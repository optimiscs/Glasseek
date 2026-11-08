"""One review-only release for full intake and expanded SearchSwarm candidates.

Deduplicate exact rendered messages + tool schema, retain origin aliases, and
union split groups so reusing a child never separates its parent bundles.
"""
import collections
import hashlib
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trajectory_preprocess'))
from client import digest
from migrate_searchswarm import review_messages
from migration_render import target_messages
from call_protocol import semantic_messages, VERSION
from glasseek_shared.dataset_policy import excluded_source


def records(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def dump(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False) + '\n')


def build_release(full, migration, out):
    if out.exists():
        raise RuntimeError('Unified output already exists: ' + str(out))
    out.mkdir(parents=True)
    full_manifest = json.loads((full / 'manifest.json').read_text())
    contract = full_manifest['config']['harness']['contract']
    assert contract['sha256'] == json.loads((migration / 'manifest.json').read_text())['contract']['sha256']
    groups, origins, seen, stats = {}, {}, {}, collections.defaultdict(collections.Counter)
    canonical_flags = collections.defaultdict(set)
    def find(group):
        groups.setdefault(group, group)
        if groups[group] != group:
            groups[group] = find(groups[group])
        return groups[group]
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            groups[max(a, b)] = min(a, b)
    mismatched = set()
    for edge in records(migration / 'links.jsonl'):
        if edge['child_id'] and not edge['report_matches_child_final']:
            mismatched.update((edge['parent_id'], edge['child_id']))
    paths = [full / 'clean.jsonl', migration / 'parent_candidates.jsonl', migration / 'child_candidates.jsonl']
    stage = out / 'candidates.staging.jsonl'
    with stage.open('w') as output, (out / 'exact_duplicates.jsonl').open('w') as duplicates:
        for path in paths:
            for index, item in enumerate(records(path)):
                if excluded_source(item.get('source')):
                    continue
                assert not item['training_ready'] and not item['harness']['errors']
                source = item['source']; role = item.get('agent_role', 'parent')
                stats[source]['input_candidates'] += 1
                group = item.get('split_group_id') or item['question_hash']
                find(group)
                if path == full / 'clean.jsonl':
                    assert source != 'SearchSwarm'
                    # The comparison stage has already checked every canonical
                    # observation for idempotent cleanup; render it directly.
                    messages = review_messages(item)
                    transforms = []
                else:
                    messages, transforms = target_messages(item)
                tools = [{'type': 'function', 'function': schema} for name, schema in contract['schemas'].items()
                         if role != 'child' or name not in contract['delegation']['blocked_child_tools']]
                # IDs, usage/cost metadata and parent IDs do not enter the training-payload hash.
                fingerprint = digest({'messages': semantic_messages(messages), 'tools': tools})
                flags = set(item.get('review_flags', [])) | set(item['harness']['flags'])
                flags.update(['factual_review_pending', 'runtime_replay_pending'])
                if item.get('instruction_messages') or item.get('source_instruction_messages'):
                    flags.add('source_system_prompt_reconciliation_required')
                if item['id'] in mismatched:
                    flags.add('linked_report_final_mismatch')
                alias = {'source_id': item['id'], 'source': source, 'agent_role': role,
                         'source_artifact': str(path.resolve()), 'source_line_zero_based': index,
                         'provenance': item['provenance'], 'original_split_group_id': group,
                         'bundle_id': item.get('bundle_id'), 'parent_id': item.get('parent_id'),
                         'reference_answer': item.get('reference_answer', '')}
                if fingerprint in seen:
                    canonical, prior_group = seen[fingerprint]
                    union(group, prior_group)
                    stats[source]['exact_duplicates_removed_at_union'] += 1
                    alias['canonical_id'] = canonical
                    dump(duplicates, alias)
                else:
                    canonical = item['id']; seen[fingerprint] = (canonical, group)
                    alias['canonical_id'] = canonical
                    row = {'id': canonical, 'source': source, 'agent_role': role, 'messages': messages, 'tools': tools,
                           'enable_thinking': True, 'training_ready': False, 'harness_replayed': False,
                           'protocol_version': VERSION,
                           'bundle_id': item.get('bundle_id'), 'parent_id': item.get('parent_id'),
                           'split_group_id': group, 'provenance': item['provenance'],
                           'source_artifact': str(path.resolve()), 'source_line_zero_based': index,
                           'reference_answer': item.get('reference_answer', ''),
                           'review_flags': sorted(flags), 'render_transforms': transforms,
                           'message_loss_mask': [int(m['role'] == 'assistant') for m in messages],
                           'payload_sha256': fingerprint}
                    dump(output, row)
                    stats[source]['candidates'] += 1
                    stats[source][role + '_candidates'] += 1
                    stats[source]['extract_calls_with_goal'] += sum(
                        c['name'] == 'web_extract' and 'goal' in c['arguments']
                        for step in item['steps'] for c in step.get('calls', []))
                assert item['id'] not in origins
                origins[item['id']] = alias
                canonical_flags[canonical].update(flags)
                if (index + 1) % 20000 == 0:
                    print('UNION_INPUT', path.name, index + 1, flush=True)
    schema = pa.schema([(k, pa.string()) for k in ('id', 'source', 'agent_role', 'messages', 'tools', 'bundle_id',
        'parent_id', 'canonical_parent_id', 'split_group_id', 'provenance', 'source_artifact', 'reference_answer_json',
        'review_flags', 'render_transforms', 'message_loss_mask', 'payload_sha256', 'protocol_version')] +
        [('source_line_zero_based', pa.int64()), ('enable_thinking', pa.bool_()), ('training_ready', pa.bool_()), ('harness_replayed', pa.bool_())])
    count = 0; batch = []
    with (out / 'candidates.jsonl').open('w') as output, pq.ParquetWriter(out / 'review_sft.parquet.tmp', schema, compression='zstd') as writer:
        for row in records(stage):
            row['split_group_id'] = find(row['split_group_id'])
            row['review_flags'] = sorted(canonical_flags[row['id']])
            row['canonical_parent_id'] = origins.get(row['parent_id'], {}).get('canonical_id')
            dump(output, row)
            # Source labels can be strings, lists or objects; keep their JSON type.
            row['reference_answer_json'] = json.dumps(row.pop('reference_answer'), ensure_ascii=False)
            for key in ('messages', 'tools', 'provenance', 'review_flags', 'render_transforms', 'message_loss_mask'):
                row[key] = json.dumps(row[key], ensure_ascii=False)
            batch.append(row); count += 1
            if len(batch) == 32:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema)); batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    assert pq.ParquetFile(out / 'review_sft.parquet.tmp').metadata.num_rows == count == len(seen)
    (out / 'review_sft.parquet.tmp').replace(out / 'review_sft.parquet')
    stage.unlink()
    with (out / 'origins.jsonl').open('w') as output:
        for origin in origins.values():
            origin['split_group_id'] = find(origin['original_split_group_id'])
            dump(output, origin)
    edge_counts = collections.Counter()
    with (out / 'delegation_links.jsonl').open('w') as output:
        for edge in records(migration / 'links.jsonl'):
            parent, child = origins.get(edge['parent_id']), origins.get(edge['child_id'])
            edge['canonical_parent_id'] = parent['canonical_id'] if parent else None
            edge['canonical_child_id'] = child['canonical_id'] if child else None
            edge['both_endpoints_are_candidates'] = bool(parent and child)
            if parent and child:
                assert find(parent['original_split_group_id']) == find(child['original_split_group_id'])
                edge_counts['both_candidate_endpoints'] += 1
            if parent or child:
                edge['split_group_id'] = find((parent or child)['original_split_group_id'])
            dump(output, edge); edge_counts['links'] += 1
    for counts in stats.values():
        assert counts['input_candidates'] == counts['candidates'] + counts['exact_duplicates_removed_at_union']
    result = {'status': 'completed', 'candidates': count, 'input_candidates': len(origins),
        'union_exact_duplicates_removed': len(origins) - count, 'by_source': stats, 'links': edge_counts,
        'split_groups': len({find(g) for g in groups}), 'training_ready': False, 'llm_calls': 0,
        'contract_sha256': contract['sha256'], 'sources': [str(p.resolve()) for p in paths],
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dedup_policy': 'Exact rendered messages + role-appropriate tool schemas; source aliases and bundle links retained. Duplicate-linked split groups unioned. No near-duplicate removal.',
        'limitations': ['Review-only unified schema, no factual approval or live replay.',
                       'Input length gates retain source-pipeline settings; apply uniform token limits before training.',
                       'Near-duplicate flags remain in the full-intake audit; no union-wide near-duplicate scan.',
                       'No train/validation split yet. Split by the resolved split_group_id.',
                       'Role-level masks retained; actual tokenizer masks checked only on the migration comparison examples.']}
    (out / 'tools.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2) + '\n')
    (out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('UNIFIED_COMPLETE', out, count, flush=True)
    return result
