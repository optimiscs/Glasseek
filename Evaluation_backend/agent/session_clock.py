"""Preserve the original clock when reopening a persisted session."""
import json
from datetime import datetime


def restored_session_start(session_id, default, snapshot_path, session_db=None):
    candidates = [default.timestamp()]
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
        if snapshot.get('session_id') == session_id:
            candidates.append(datetime.fromisoformat(snapshot['session_start']).timestamp())
    except (OSError, ValueError, TypeError, KeyError):
        pass
    if session_db is not None:
        try:
            row = session_db.get_session(session_id)
            if row and row.get('started_at') is not None:
                candidates.append(float(row['started_at']))
        except Exception:
            # Optional session store must not prevent starting the agent.
            pass
    return datetime.fromtimestamp(min(candidates))
