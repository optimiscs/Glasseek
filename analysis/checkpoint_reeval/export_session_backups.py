"""Stream consistent SQLite backups for sessions listed in a live summary."""
import json
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path

summary = Path(sys.argv[1])
profiles = Path(sys.argv[2])
with tempfile.TemporaryDirectory(prefix='glasseek-session-backup-') as tmp:
    tmp = Path(tmp)
    seen = set()
    for line in (summary.read_text().splitlines() if summary.exists() else []):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        profile = row.get('profile')
        if not profile or Path(profile).name != profile or profile in seen:
            continue
        source = profiles / profile / 'state.db'
        if not source.is_file():
            continue
        seen.add(profile)
        out = tmp / (profile + '.sqlite')
        with sqlite3.connect(f'{source.as_uri()}?mode=ro', uri=True, timeout=20) as src:
            with sqlite3.connect(out) as dst:
                src.backup(dst)
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
        for path in sorted(tmp.iterdir()):
            archive.add(path, arcname=path.name)
