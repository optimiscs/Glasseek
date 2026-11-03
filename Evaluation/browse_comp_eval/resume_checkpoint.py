"""Small, opt-in checkpoints for resuming interrupted Hermes attempts."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path

FILENAME = '.glasseek_resume.json'


def checkpoint_path(workspace: Path) -> Path:
    return workspace / FILENAME


def write_checkpoint(workspace: Path, *, run_id: str, profile: str, started_at: float | None = None) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    value = {'version': 1, 'run_id': run_id, 'profile': profile,
             'started_at': time.time() if started_at is None else started_at}
    fd, name = tempfile.mkstemp(prefix='.resume-', dir=workspace)
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, checkpoint_path(workspace))
    finally:
        if os.path.exists(name):
            os.unlink(name)


def existing_session(workspace: Path, *, run_id: str, profile: str, hermes_home: Path) -> str | None:
    path = checkpoint_path(workspace)
    if not path.exists():
        return None
    checkpoint = json.loads(path.read_text(encoding='utf8'))
    if checkpoint.get('version') != 1 or checkpoint.get('run_id') != run_id or checkpoint.get('profile') != profile:
        raise ValueError('resume checkpoint does not match the current run and profile')
    db = hermes_home / 'profiles' / profile / 'state.db'
    if not db.exists():
        return None
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True, timeout=5) as connection:
        pinned = checkpoint.get('session_id')
        if pinned:
            row = connection.execute('SELECT 1 FROM messages WHERE session_id=? LIMIT 1', (pinned,)).fetchone()
            if not row:
                raise ValueError('pinned resume session has no persisted messages')
            return str(pinned)
        row = connection.execute(
            "SELECT s.id FROM sessions s WHERE s.started_at >= ? "
            "AND EXISTS (SELECT 1 FROM messages m WHERE m.session_id=s.id) "
            "ORDER BY s.started_at DESC LIMIT 1", (float(checkpoint['started_at']) - 5,)).fetchone()
    return str(row[0]) if row else None
