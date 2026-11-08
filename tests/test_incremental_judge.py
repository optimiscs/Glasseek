import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


SERVICES = Path(__file__).resolve().parents[1] / 'Evaluation/services/remote44'
sys.path.insert(0, str(SERVICES.parents[1]))
sys.path.insert(0, str(SERVICES))
spec = importlib.util.spec_from_file_location('incremental_judge', SERVICES / 'run_sft9b_16_and_judge.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class IncrementalJudgeTest(unittest.TestCase):
    def test_terminal_only_excludes_inflight_checkpoints(self):
        from browse_comp_eval.judge import load_prediction_records
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'one.json').write_text(json.dumps({'run_id': 'one', 'status': 'interrupted'}))
            (root / 'two.json').write_text(json.dumps({'run_id': 'two', 'status': 'success'}))
            with patch.dict(os.environ, {'GLASSEEK_JUDGE_TERMINAL_ONLY': '1'}):
                self.assertEqual([x.run_id for x in load_prediction_records(root)], ['two'])

    def test_each_summary_update_is_processed_without_concurrent_judges(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / 'summary.jsonl'
            calls = []
            stop = threading.Event()

            def run(*args, **kwargs):
                calls.append(summary.read_text())
                return type('Result', (), {'returncode': 0})()

            with patch.object(module, 'SUMMARY', summary), patch.object(module.subprocess, 'run', side_effect=run):
                thread = threading.Thread(target=module.judge_loop, args=(stop,), kwargs={'poll_seconds': .01, 'retry_seconds': 60})
                thread.start()
                try:
                    summary.write_text('first\n')
                    self.wait_for(lambda: len(calls) == 1)
                    with summary.open('a') as stream:
                        stream.write('second\n')
                    self.wait_for(lambda: len(calls) == 2)
                    time.sleep(.05)
                    self.assertEqual(calls, ['first\n', 'first\nsecond\n'])
                finally:
                    stop.set()
                    thread.join(2)
                self.assertFalse(thread.is_alive())

    def wait_for(self, condition):
        deadline = time.monotonic() + 2
        while not condition() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(condition())


if __name__ == '__main__':
    unittest.main()
