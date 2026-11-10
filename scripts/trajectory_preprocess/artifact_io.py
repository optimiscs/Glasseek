"""Read plain or losslessly compressed JSONL intermediates."""
import gzip
from pathlib import Path


def resolve_jsonl(path):
    path = Path(path)
    return path if path.exists() else path.with_suffix(path.suffix + '.gz')


def open_jsonl(path, mode='rt'):
    path = resolve_jsonl(path)
    options = {} if 'b' in mode else {'encoding': 'utf-8'}
    return gzip.open(path, mode, **options) if path.suffix == '.gz' else path.open(mode, **options)
