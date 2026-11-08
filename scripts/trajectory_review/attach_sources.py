"""Attach verified, redacted source records to an existing review batch."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'scripts/trajectory_preprocess'), str(ROOT/'Shared')]
from trajectory_review.core import connect
from client import digest
from ingest import records
from safety_rules import Scanner


def attach(batch, scan, key_file):
    db = connect(batch/'reviews.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS case_files (case_id TEXT, kind TEXT, metadata TEXT, content TEXT, PRIMARY KEY(case_id,kind))')
    targets = {r['content_hash']: r['id'] for r in db.execute('SELECT id,content_hash FROM cases')}
    manifest = json.loads((scan/'manifest.json').read_text())
    source = Path(manifest['input']['path'])
    scanner = Scanner(json.loads((batch/'policy.json').read_text()), json.loads((scan/'contract.json').read_text()), key_file.read_bytes().strip())
    old = sqlite3.connect('file:'+str(scan/'state.sqlite')+'?mode=ro', uri=True)
    pending = collections.defaultdict(list)

    def store(ident, kind, item, path, index, expected=None):
        encoded = json.dumps(item, ensure_ascii=False, indent=2)
        safe = json.dumps(json.loads(scanner.scrub(encoded)), ensure_ascii=False, indent=2)
        metadata = dict(path=str(path), row_index=index, line_number=index+1,
                        original_sha256=digest(item), redacted=safe!=encoded,
                        displayed_sha256=hashlib.sha256(safe.encode()).hexdigest(), characters=len(safe),
                        filename=f'{ident}-{kind}.json', provenance_sha256=expected,
                        provenance_hash_matches=digest(item)==expected if expected else None)
        db.execute('INSERT OR REPLACE INTO case_files VALUES(?,?,?,?)', (ident, kind, json.dumps(metadata,ensure_ascii=False), safe))

    with source.open('rb') as stream:
        for (encoded,) in old.execute('SELECT result FROM decisions'):
            receipt = json.loads(encoded)
            ident = targets.get(receipt['content_hash'])
            if not ident: continue
            stream.seek(receipt['offset_start'])
            item = json.loads(stream.read(receipt['offset_end']-receipt['offset_start']))
            if digest(item)!=receipt['content_hash']: raise ValueError('Candidate hash mismatch')
            store(ident, 'candidate', item, source, receipt['source_row'])
            provenance = item.get('provenance', {})
            if provenance.get('path') and type(provenance.get('row_index')) is int:
                pending[provenance['path']].append((provenance['row_index'], ident, provenance.get('raw_sha256')))
    attached = collections.Counter()
    for path, wanted in pending.items():
        path = Path(path).resolve()
        if not path.is_relative_to((ROOT/'data').resolve()): raise ValueError('Source outside dataset directory')
        by_index = collections.defaultdict(list)
        for index, ident, expected in wanted: by_index[index].append((ident, expected))
        for index, raw in enumerate(records(path, max_record_chars=128000000)):
            if index in by_index:
                for ident, expected in by_index.pop(index):
                    # Some adapters hash an intermediate projection. Preserve the
                    # coordinate-based source record, but never claim hash equality.
                    store(ident, 'dataset', raw, path, index, expected)
                    attached['dataset']+=1
            if not by_index: break
        if by_index: raise ValueError('Source rows not found')
    db.commit()
    attached['candidate']=db.execute("SELECT COUNT(*) FROM case_files WHERE kind='candidate'").fetchone()[0]
    db.close(); old.close()
    print(json.dumps(dict(attached)))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--scan', type=Path, required=True)
    p.add_argument('--key-file', type=Path, required=True)
    args=p.parse_args()
    attach(args.batch, args.scan, args.key_file)
