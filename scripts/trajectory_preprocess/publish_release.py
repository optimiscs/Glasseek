"""Publish only hash-bound accepted trajectories; never auto-approve candidates."""
import argparse
import collections
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Shared'))
from glasseek_shared.trajectory_release import (
    PROTOCOL, SCHEMA_VERSION, RELEASE_VERSION, RESPONSE_FORMAT, SELECTION_VERSION, REQUIRED_CHECKS, digest, payload,
    runtime_schemas, file_hash, validate_row, verify_release_file)
from glasseek_shared.dataset_policy import excluded_source


def publish(source, acceptance_path, output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    accepted = {}
    with Path(acceptance_path).open() as stream:
        for line in stream:
            entry = json.loads(line)
            if entry['id'] in accepted:
                raise ValueError('duplicate_acceptance_id')
            accepted[entry['id']] = entry
    tools = runtime_schemas()
    counts, reasons = collections.Counter(), collections.Counter()
    writer = None
    buffer = []
    def flush():
        nonlocal writer
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer)
        if writer is None:
            writer = pq.ParquetWriter(output / 'train.parquet', table.schema)
        writer.write_table(table)
        buffer.clear()
    seen = set()
    try:
        with Path(source).open() as stream, (output / 'held.jsonl').open('w') as held, (output / 'content_hashes.jsonl').open('w') as inventory:
            for line in stream:
                candidate = json.loads(line)
                counts['scanned'] += 1
                ident = candidate['id']
                review = accepted.get(ident, {})
                try:
                    if excluded_source(candidate.get('source')):
                        raise ValueError('excluded_dataset_source')
                    if any(review.get(k) != 'passed' for k in REQUIRED_CHECKS):
                        raise ValueError('acceptance_missing_or_not_passed')
                    if set(candidate.get('review_flags', [])) - set(review.get('resolved_review_flags', [])):
                        raise ValueError('unresolved_review_flags')
                    from normalize_training_text import normalize
                    normalized=normalize(candidate, tool_schemas=tools)
                    if normalized['text_normalization']['holds'] or digest(payload(normalized))!=digest(payload(candidate)):
                        raise ValueError('training_text_normalization_required')
                    value = payload(candidate)
                    if review.get('content_sha256') != digest(value):
                        raise ValueError('stale_acceptance_content_hash')
                    row = {**value, 'source': candidate.get('source', ''),
                           'protocol_version': PROTOCOL, 'schema_version': SCHEMA_VERSION,
                           'training_ready': True, 'harness_replayed': True, 'acceptance_status': 'passed',
                           'acceptance': json.dumps(review, ensure_ascii=False), 'content_sha256': digest(value)}
                    validate_row(row, tools)
                    if ident in seen:
                        raise ValueError('duplicate_release_id')
                except (ValueError, KeyError, TypeError) as exc:
                    reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                    # Reasons are fixed codes; no evidence, URL parameters or credentials.
                    reasons[reason] += 1
                    held.write(json.dumps({'id': ident, 'reason': reason}, ensure_ascii=False) + '\n')
                    continue
                seen.add(ident)
                row['messages'] = json.dumps(row['messages'], ensure_ascii=False)
                row['tools'] = json.dumps(row['tools'], ensure_ascii=False)
                row['provenance'] = json.dumps(row['provenance'], ensure_ascii=False)
                # Keep a stable Arrow column even if legacy rows have no plan.
                row['supervision'] = json.dumps(row['supervision'],ensure_ascii=False) if 'supervision' in row else None
                buffer.append(row)
                if len(buffer) >= 64:
                    flush()
                inventory.write(json.dumps({'id': ident, 'content_sha256': row['content_sha256']}) + '\n')
                counts['passed'] += 1
            flush()
    finally:
        if writer is not None:
            writer.close()
    files = [{'path': 'train.parquet', 'rows': counts['passed'], 'sha256': file_hash(output / 'train.parquet')}] if counts['passed'] else []
    manifest = {'version': RELEASE_VERSION, 'protocol_version': PROTOCOL, 'schema_version': SCHEMA_VERSION,
                'response_format': RESPONSE_FORMAT, 'dataset_selection_version': SELECTION_VERSION,
                'acceptance_status': 'passed' if counts['passed'] else 'empty_no_accepted_rows',
                'rows': counts['passed'], 'counts': dict(counts), 'held_reasons': dict(reasons), 'files': files,
                'tool_schema_sha256': digest(tools),
                'content_inventory': {'path': 'content_hashes.jsonl', 'sha256': file_hash(output / 'content_hashes.jsonl')},
                'source': {'path': str(Path(source).resolve()), 'sha256': file_hash(source)},
                'acceptance_source': {'path': str(Path(acceptance_path).resolve()), 'sha256': file_hash(acceptance_path)},
                'publisher_sha256': file_hash(__file__),
                'validator_sha256': file_hash(Path(__file__).resolve().parents[2] / 'Shared/glasseek_shared/trajectory_release.py')}
    (output / 'release.manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    if counts['passed']:
        verify_release_file(output / 'train.parquet', tools)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--acceptance', type=Path, required=True,
                        help='JSONL: id, content_sha256, safety/quality/replay/review=passed, resolved_review_flags')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.input, args.acceptance, args.output), ensure_ascii=False))
