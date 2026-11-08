import importlib.util
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('session_clock', Path(__file__).resolve().parents[1] / 'Evaluation_backend/agent/session_clock.py')
clock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clock)


class SessionClockTests(unittest.TestCase):
    def test_resume_and_second_resume_keep_original_snapshot_start(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session.json'
            original = datetime(2026, 1, 1, 12)
            path.write_text(json.dumps({'session_id': 's', 'session_start': original.isoformat()}))
            db = SimpleNamespace(get_session=lambda sid: {'started_at': original.timestamp() + 5})
            for hour in (13, 14):
                restored = clock.restored_session_start('s', datetime(2026, 1, 1, hour), path, db)
                self.assertEqual(restored, original)
                path.write_text(json.dumps({'session_id': 's', 'session_start': restored.isoformat()}))

    def test_db_repairs_legacy_reset_snapshot_and_ignores_different_session(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session.json'
            original = datetime(2026, 1, 1, 12)
            now = datetime(2026, 1, 1, 14)
            path.write_text(json.dumps({'session_id': 's', 'session_start': now.isoformat()}))
            db = SimpleNamespace(get_session=lambda sid: {'started_at': original.timestamp()} if sid == 's' else None)
            self.assertEqual(clock.restored_session_start('s', now, path, db), original)
            self.assertEqual(clock.restored_session_start('new', now, path, db), now)
            path.write_text('broken')
            self.assertEqual(clock.restored_session_start('new', now, path, db), now)
