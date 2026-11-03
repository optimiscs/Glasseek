import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'Evaluation/scripts/fetch_watchdog.py'
spec = importlib.util.spec_from_file_location('fetch_watchdog', path)
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


class WatchdogResumeTests(unittest.TestCase):
    def test_fetch_abort_does_not_consume_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = root / 'runs.txt'
            ids.write_text('row0013__r1\n')
            args = argparse.Namespace(run_ids_file=str(ids), data_path=str(root/'dataset.jsonl'),
                project_root=str(root), save_name='run', output_root=str(root/'output'),
                dataset='dataset', model='model', numbered_output=False,
                fetch_server_url='http://127.0.0.1:18081', reason='probe failed')
            meta = {'row0013': {'source_line': 1, 'question': 'q', 'answer': 'a', 'type': 'Other'}}
            with patch.object(watchdog, '_dataset_metadata', return_value=meta):
                self.assertEqual(watchdog.mark_failed(args), 0)
            base = root/'output/run/dataset'
            result = json.loads((base/'conv/row0013/row0013__r1.json').read_text())
            summary = json.loads((base/'summary.jsonl').read_text())
            self.assertIs(result['attempt_consumed'], False)
            self.assertIs(summary['attempt_consumed'], False)
            self.assertEqual(result['failure_type'], 'infrastructure')


if __name__ == '__main__':
    unittest.main()
