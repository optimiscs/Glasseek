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
        # Main sessions can have CLI descendants after context compression.
        # Never resume a delegate (or a CLI descendant of a delegate) as the
        # main task merely because it was created more recently.
        main_sessions = connection.execute(
            "WITH RECURSIVE main(id, started_at) AS ("
            "SELECT id, started_at FROM sessions "
            "WHERE source='cli' AND parent_session_id IS NULL AND started_at >= ? "
            "UNION SELECT s.id, s.started_at FROM sessions s JOIN main p "
            "ON s.parent_session_id=p.id WHERE s.source='cli') "
            "SELECT id FROM main WHERE EXISTS "
            "(SELECT 1 FROM messages m WHERE m.session_id=main.id) "
            "ORDER BY started_at DESC, id DESC",
            (float(checkpoint['started_at']) - 5,),
        ).fetchall()
        main_ids = [str(row[0]) for row in main_sessions]
        pinned = checkpoint.get('session_id')
        if pinned:
            if str(pinned) not in main_ids:
                raise ValueError('pinned resume session is not a persisted main CLI session')
            return str(pinned)
        if main_ids:
            return main_ids[0]
        if connection.execute('SELECT 1 FROM messages LIMIT 1').fetchone():
            raise ValueError('persisted messages exist but no eligible main CLI session was found')
    return None
