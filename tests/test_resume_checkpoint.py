import json
import sqlite3
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from browse_comp_eval.resume_checkpoint import existing_session, write_checkpoint


class ResumeCheckpointTest(unittest.TestCase):
    def test_infrastructure_abort_is_eligible_without_rerunning_model_failures(self):
        from browse_comp_eval.runner import BrowseCompRunner
        runner = BrowseCompRunner(None, SimpleNamespace(force=False, rerun_failed=False, max_retries=1, skip_statuses={'success'}))
        spec = SimpleNamespace()
        with patch.dict('os.environ', {'GLASSEEK_RESUME_INTERRUPTED': '1'}):
            with patch.object(runner, '_load_existing_result', return_value={'status': 'failed', 'attempt': 1, 'attempt_consumed': False}):
                self.assertTrue(runner._should_run(spec))
            with patch.object(runner, '_load_existing_result', return_value={'status': 'failed', 'attempt': 1, 'attempt_consumed': True}):
                self.assertFalse(runner._should_run(spec))

    def test_resume_uses_latest_persisted_session_and_checks_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / 'workspace'
            db = root / 'profiles' / 'profile' / 'state.db'
            db.parent.mkdir(parents=True)
            with sqlite3.connect(db) as connection:
                connection.executescript('CREATE TABLE sessions (id TEXT, started_at REAL, source TEXT, parent_session_id TEXT); CREATE TABLE messages (session_id TEXT);')
                connection.executemany('INSERT INTO sessions VALUES (?,?,?,?)', [
                    ('old', 10, 'cli', None), ('new', 20, 'cli', 'old'),
                    ('empty', 30, 'cli', 'new'), ('delegate', 40, 'subagent', 'new'),
                    ('delegate_continuation', 50, 'cli', 'delegate')])
                connection.executemany('INSERT INTO messages VALUES (?)', [(s,) for s in ['old', 'new', 'delegate', 'delegate_continuation']])
            write_checkpoint(workspace, run_id='question__r1', profile='profile', started_at=1)
            self.assertEqual(existing_session(workspace, run_id='question__r1', profile='profile', hermes_home=root), 'new')
            marker = workspace / '.glasseek_resume.json'
            checkpoint = json.loads(marker.read_text())
            checkpoint['session_id'] = 'old'
            marker.write_text(json.dumps(checkpoint))
            self.assertEqual(existing_session(workspace, run_id='question__r1', profile='profile', hermes_home=root), 'old')
            checkpoint['session_id'] = 'delegate'
            marker.write_text(json.dumps(checkpoint))
            with self.assertRaisesRegex(ValueError, 'main CLI'):
                existing_session(workspace, run_id='question__r1', profile='profile', hermes_home=root)
            with self.assertRaises(ValueError):
                existing_session(workspace, run_id='different', profile='profile', hermes_home=root)

    def test_checkpoint_without_session_starts_normally(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / 'workspace'
            write_checkpoint(workspace, run_id='q__r1', profile='profile', started_at=time.time())
            self.assertIsNone(existing_session(workspace, run_id='q__r1', profile='profile', hermes_home=root))


if __name__ == '__main__':
    unittest.main()
