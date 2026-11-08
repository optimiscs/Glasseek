"""Resumable, offline full-corpus intake; never calls a model or live tools.

SQLite checkpoints commit dedup state and fsynced JSONL byte offsets together.
On restart uncommitted output tails are truncated before input is reread.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import fcntl
import glob
import hashlib
import json
import os
import multiprocessing
import pickle
from pathlib import Path
import time

from client import digest, save
from ingest import Dedup, harness_plan, normalize, records, simhash, semantic_hash
from dispositions import categorize
from glasseek_shared.dataset_policy import excluded_source

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ('clean', 'held', 'duplicates', 'near_duplicates', 'file_errors',
           'unsupported_tools', 'adapter_holds', 'quality_issues', 'format_review', 'incomplete_episodes', 'length_budget')


def inventory(specs):
    files = []
    seen = set()
    for spec in specs:
        if excluded_source(spec.get('source')):
            continue
        paths = sorted(glob.glob(str(ROOT / spec['path']), recursive=True))
        if not paths:
            raise ValueError('No local input files: ' + spec['path'])
        for name in paths:
            path = Path(name).resolve()
            if str(path) in seen:
                raise ValueError('Overlapping input patterns: ' + str(path))
            seen.add(str(path))
            stat = path.stat()
            files.append({**spec, 'path': str(path), 'bytes': stat.st_size,
                          'mtime_ns': stat.st_mtime_ns})
    return files


def input_rows(spec):
    if spec['source'] == 'SearchSwarm':
        # This unsupported bundle's nested text column has a multi-GB row group.
        # Project only scalar metadata; retain source row coordinates for repair.
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(spec['path']).iter_batches(
                batch_size=64, columns=['question', 'num_subagents'], use_threads=False):
            yield from batch.to_pylist()
    else:
        yield from records(spec['path'])


def preflight(files):
    """Fail before processing any data if a required reader is unavailable.

    Read Parquet footers up front, both to verify accessibility and to record
    expected row counts for end-of-run completeness checks.
    """
    parquet = [spec for spec in files if Path(spec['path']).suffix == '.parquet']
    if not parquet:
        return {}
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError('Parquet inputs require pyarrow in the selected Python environment; '
                           'no input has been processed') from exc
    return {spec['path']: pq.ParquetFile(spec['path']).metadata.num_rows for spec in parquet}


def prepare_record(row, spec, index, config):
    """Pure per-record work; quality rules are identical in serial and parallel modes."""
    source = spec['source']
    ident = digest([source, spec['path'], index])[:24]
    location = {'id': ident, 'source': source, 'path': spec['path'], 'row_index': index}
    item, fingerprint, reasons = None, None, []
    if source == 'SearchSwarm':
        reasons = ['hierarchical_bundle_adapter_required']
        location['projection'] = 'scalar_metadata_only'
    else:
        try:
            item = normalize(row, source, spec['path'], index, spec.get('adapter', 'auto'))
            item['adapter_id'] = item['id']
            item['id'] = ident
            item['harness'] = harness_plan(item, config['harness'])
            item['program_errors'] = sorted(set(item['program_errors'] + item['harness']['errors']))
            reasons.extend(item['program_errors'])
            if config['harness'].get('contract') and not reasons:
                from glasseek_contract import materialize
                item['source_content_hash']=item['content_hash']
                materialize(item)
                item['content_hash']=semantic_hash(item)
            if len(json.dumps(item, ensure_ascii=False)) > config['max_sample_chars']:
                reasons.append('annotation_size_budget_exceeded')
            fingerprint = simhash(item['question'])
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            reasons = ['normalization_error']
            location['error'] = type(exc).__name__ + ': ' + str(exc)
            item = None
    return index, location, item, reasons, fingerprint


def prepare_batch(batch, spec, config):
    # Bytes originate only from this process's local readers, never remote pickle files.
    return [prepare_record(pickle.loads(raw), spec, index, config) for index, raw in batch]


def prepared_rows(spec, config, skip, pool, workers):
    reader = enumerate(input_rows(spec))
    if pool is None:
        for index, row in reader:
            if index >= skip:
                yield prepare_record(row, spec, index, config)
        return
    pending = deque()
    pending_bytes = 0
    finished, read_error = False, None
    try:
        while pending or not finished:
            # Bound both task count and serialized input bytes. One oversized row
            # may exceed the byte target; no unbounded executor.map prefetch.
            while not finished and len(pending) < workers * 2 and pending_bytes < 128 * 1024**2:
                batch, size = [], 0
                while len(batch) < 8 and size < 8 * 1024**2:
                    try:
                        index, row = next(reader)
                    except StopIteration:
                        finished = True
                        break
                    except Exception as exc:
                        read_error, finished = exc, True
                        break
                    if index < skip:
                        continue
                    raw = pickle.dumps(row, protocol=5)
                    size += len(raw)
                    batch.append((index, raw))
                if batch:
                    pending.append((pool.submit(prepare_batch, batch, spec, config), size))
                    pending_bytes += size
            if pending:
                future, size = pending.popleft()
                # Resolve strictly in source order: duplicate representatives and
                # output order must not depend on which worker finishes first.
                result = future.result()
                pending_bytes -= size
                yield from result
        if read_error is not None:
            raise read_error
    finally:
        for future, _ in pending:
            future.cancel()


def run(specs, config, out, checkpoint_every=100, stop_after=None, workers=1):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another process owns this output directory') from None
        if workers < 1:
            raise ValueError('workers must be positive')
        if workers == 1:
            return run_locked(specs, config, out, checkpoint_every, stop_after, None, workers)
        # spawn avoids inheriting SQLite connections, file locks and Arrow threads.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            return run_locked(specs, config, out, checkpoint_every, stop_after, pool, workers)


def run_locked(specs, config, out, checkpoint_every, stop_after, pool, workers):
    files = inventory(specs)
    parquet_rows = preflight(files)
    source_files = [Path(__file__), *[Path(__file__).with_name(n) for n in
                                     ('ingest.py', 'tool_xml.py', 'client.py', 'message_adapter.py', 'dispositions.py', 'glasseek_contract.py', 'runtime_metadata.py', 'call_protocol.py')],
                    Path(__file__).resolve().parents[2] / 'Shared/glasseek_shared/extract_protocol.py',
                    Path(__file__).resolve().parents[2] / 'Shared/glasseek_shared/dataset_policy.py']
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    manifest = {'version': 'offline-full-prepare-8-call-response', 'inputs': files, 'config': config,
                'parquet_expected_rows': parquet_rows,
                'code_sha256': source_hashes,
                'input_identity': 'absolute path + bytes + mtime_ns; not a download completeness certificate',
                'scope': 'All configured local SFT files; no QA/DPO or gated MiroVerse',
                'searchswarm': 'metadata-only quarantine; bundles NOT expanded',
                'near_dedup': 'question SimHash; bounded candidates; review only',
                'llm_calls': 0, 'runtime_replay': False, 'training_ready': False}
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Inputs, code or rules changed; use a new output directory')
    else:
        save(manifest_path, manifest)
        frozen = out / 'frozen_source'
        frozen.mkdir(exist_ok=True)
        for path in source_files:
            (frozen / path.name).write_bytes(path.read_bytes())
    save(out / 'runtime.json', {'workers': workers, 'checkpoint_every': checkpoint_every,
                              'max_pending_batches': workers * 2, 'batch_rows': 8,
                              'pending_input_target_mib': 128, 'batch_input_target_mib': 8,
                              'ordering': 'source order; single SQLite writer'})

    dedup = Dedup(out / 'dedup.sqlite', config['dedup'])
    db = dedup.db
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS checkpoint (id INTEGER PRIMARY KEY, state TEXT)')
    db.commit()
    prior = db.execute('SELECT state FROM checkpoint WHERE id=1').fetchone()
    state = json.loads(prior[0]) if prior else {
        'file_index': 0, 'next_row': 0, 'counts': {}, 'sources': {}, 'reasons': {},
        'offsets': {name: 0 for name in OUTPUTS}, 'elapsed_seconds': 0,
        'status': 'running', 'completed_files': 0}
    handles = {}
    for name in OUTPUTS:
        path = out / (name + '.jsonl')
        handle = path.open('r+b' if path.exists() else 'w+b')
        if handle.seek(0, 2) < state['offsets'][name]:
            raise RuntimeError('Checkpointed output was truncated: ' + str(path))
        handle.truncate(state['offsets'][name])
        handle.seek(0, 2)
        handles[name] = handle
    start = time.monotonic()
    previous_elapsed = state['elapsed_seconds']
    processed_this_run = 0

    def emit(name, value):
        if name == 'held':
            value['categories'] = categorize(value['reasons'],value.get('error',''))
            for category in value['categories']:
                handles[category].write((json.dumps(value,ensure_ascii=False)+'\n').encode('utf-8'))
        handles[name].write((json.dumps(value, ensure_ascii=False) + '\n').encode('utf-8'))

    def count(source, key):
        state['counts'][key] = state['counts'].get(key, 0) + 1
        totals = state['sources'].setdefault(source, {})
        totals[key] = totals.get(key, 0) + 1

    def checkpoint(status='running'):
        for name, handle in handles.items():
            handle.flush()
            os.fsync(handle.fileno())
            state['offsets'][name] = handle.tell()
        state['elapsed_seconds'] = previous_elapsed + time.monotonic() - start
        state['status'] = status
        state['updated_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        db.execute('INSERT OR REPLACE INTO checkpoint VALUES (1,?)', (json.dumps(state),))
        db.commit()
        save(out / 'progress.json', state)
        print(json.dumps({k: state[k] for k in ('status', 'file_index', 'next_row',
                                               'counts', 'elapsed_seconds')}, ensure_ascii=False), flush=True)

    try:
        checkpoint()
        for file_index in range(state['file_index'], len(files)):
            spec = files[file_index]
            source = spec['source']
            skip = state['next_row'] if file_index == state['file_index'] else 0
            state['file_index'] = file_index
            print('FILE ' + spec['path'] + ' resume_row=' + str(skip), flush=True)
            iterator = iter(prepared_rows(spec, config, skip, pool, workers))
            index = skip - 1
            while True:
                try:
                    index, location, item, reasons, fingerprint = next(iterator)
                except StopIteration:
                    expected = parquet_rows.get(spec['path'])
                    if expected is not None and index + 1 != expected:
                        count(source, 'file_errors')
                        emit('file_errors', {'source': source, 'path': spec['path'],
                                            'error': 'Parquet row count mismatch',
                                            'expected_rows': expected, 'read_rows': index + 1})
                    else:
                        state['completed_files'] += 1
                    break
                except Exception as exc:
                    # Incomplete file is explicit and never reported as a complete scan.
                    count(source, 'file_errors')
                    emit('file_errors', {'source': source, 'path': spec['path'],
                                        'next_row': index + 1, 'error': type(exc).__name__ + ': ' + str(exc)})
                    break
                count(source, 'scanned')
                if item is not None:
                    count(source, 'normalized')
                    item['dedup'] = dedup.add(item, question_simhash=fingerprint)
                    if item['dedup']['exact_duplicate_of']:
                        count(source, 'exact_duplicates')
                        emit('duplicates', {**location, **item['dedup'], 'content_hash': item['content_hash']})
                    else:
                        if item['dedup']['near_duplicates']:
                            count(source, 'near_duplicate_flagged')
                            emit('near_duplicates', {**location, **item['dedup']})
                        if reasons:
                            count(source, 'held')
                            # Keep source pointer + hashes, not another copy of long text.
                            emit('held', {**location, 'reasons': reasons, 'content_hash': item['content_hash'],
                                          'raw_sha256': item['provenance']['raw_sha256'],
                                          'adapter_flags': item['harness']['flags']})
                        else:
                            count(source, 'eligible')
                            item.update(training_ready=False, processing_status='eligible_for_annotation')
                            emit('clean', item)
                else:
                    count(source, 'held')
                    emit('held', {**location, 'reasons': reasons})
                for reason in set(reasons):
                    state['reasons'][reason] = state['reasons'].get(reason, 0) + 1
                state['next_row'] = index + 1
                processed_this_run += 1
                if processed_this_run % checkpoint_every == 0:
                    checkpoint()
                if stop_after is not None and processed_this_run >= stop_after:
                    checkpoint('paused')
                    return state
            state['file_index'] = file_index + 1
            state['next_row'] = 0
            checkpoint()
        status = 'completed_with_file_errors' if state['counts'].get('file_errors') else 'completed'
        checkpoint(status)
        save(out / 'cleaning_summary.json', state)
        return state
    finally:
        # Never commit partially written rows on an unexpected failure.
        db.rollback()
        db.close()
        for handle in handles.values():
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=Path(__file__).with_name('inputs.datasets.json'))
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--harness-profile', choices=('web-only','glasseek-eval'), default='glasseek-eval')
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error('--checkpoint-every must be positive')
    config = {'max_sample_chars': 250000,
              'dedup': {'near_question_hamming_distance': 3, 'max_bucket_candidates': 200},
              'harness': {'search_name': 'web_search', 'extract_name': 'web_extract', 'extract_max_urls': 5}}
    if args.harness_profile=='glasseek-eval':
        from glasseek_contract import build_contract
        config['harness']['contract']=build_contract()
    result = run(json.loads(args.inputs.read_text()), config, args.output, args.checkpoint_every, workers=args.workers)
    if result['counts'].get('file_errors'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
