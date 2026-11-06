import json
import os
import subprocess
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Evaluation/services/remote44'))
import supervise_evaluation
from judge_after_existing_run import complete


class LauncherTests(unittest.TestCase):
    def test_interrupted_attempt_does_not_count_as_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); dataset = root/'data.jsonl'; summary = root/'summary.jsonl'
            dataset.write_text('{"id":"row1"}\n')
            summary.write_text('{"question_id":"row1","status":"failed","attempt_consumed":false}\n')
            self.assertEqual(complete(dataset, summary), (False, 1, 0))
            with summary.open('a') as stream:
                stream.write('{"question_id":"row1","status":"success"}\n')
            self.assertEqual(complete(dataset, summary), (True, 1, 1))

    def test_supervisor_publishes_and_runs_judge_after_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); site = root/'site'; (site/'datasets').mkdir(parents=True)
            (site/'datasets/index.json').write_text('{"datasets": []}')
            runtime = root/'runtime'; runtime.mkdir()
            child = Mock(pid=123)
            completed = Mock(returncode=0)
            thread = Mock()
            with patch.dict(os.environ, {'QUEST_VIEWER_SITE': str(site)}), \
                 patch.object(supervise_evaluation, 'RUNTIME', runtime), \
                 patch.object(supervise_evaluation, 'ROOT', root), \
                 patch.object(supervise_evaluation.subprocess, 'Popen', return_value=child), \
                 patch.object(supervise_evaluation.subprocess, 'run', return_value=completed) as run, \
                 patch.object(supervise_evaluation.threading, 'Thread', return_value=thread), \
                 patch.object(supervise_evaluation.incremental, 'SCRIPT'), \
                 patch.object(supervise_evaluation.incremental, 'SUMMARY'), \
                 patch.object(supervise_evaluation, 'complete', return_value=(True, 128, 128)):
                code = supervise_evaluation.supervise(run='my-run', workspace_id='my-workspace',
                    model='model', label='label', description='description', script=root/'launch.sh', log_name='viewer.log')
            self.assertEqual(code, 0)
            self.assertEqual([call.args[0][-1] for call in run.call_args_list], ['full', 'judge'])
            self.assertEqual(json.loads((site/'datasets/index.json').read_text())['datasets'][0]['id'], 'my-workspace')
            thread.start.assert_called_once()
            thread.join.assert_called_once()

    def test_full_launch_preserves_parameters_with_custom_paths(self):
        for launcher, model, workers, limit in (
            ('Evaluation/services/run_rl9b_eval.sh', 'glasseek-rl-9b', '8', '32'),
            ('Evaluation/services/remote44/run_rl4_c16.sh', 'glasseek-rl-4b', '4', '4'),
        ):
            with self.subTest(launcher=launcher), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp); evaluation = base / 'evaluation'; runtime = base / 'runtime'
                (evaluation / 'scripts').mkdir(parents=True)
                (evaluation / 'scripts/run_eval.sh').write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
                (evaluation / 'scripts/run_eval.sh').chmod(0o755)
                env = dict(os.environ, GLASSEEK_EVAL_ROOT=str(evaluation), GLASSEEK_RUNTIME_DIR=str(runtime),
                           QUEST_ROOT=str(base/'quest'), HERMES_REMOTE_PURE_COMPUTE='sandbox-host')
                result = subprocess.run(['bash', str(ROOT/launcher), 'full'], env=env, capture_output=True, text=True, check=True)
                args = result.stdout.splitlines()
                self.assertEqual(args[args.index('--model')+1], model)
                self.assertEqual(args[args.index('--workers-per-endpoint')+1], workers)
                self.assertEqual(args[args.index('--ramp-max')+1], limit)
                self.assertEqual(args[args.index('--timeout-seconds')+1], '3600')
                self.assertEqual(args[args.index('--max-retries')+1], '1')
                self.assertTrue(args[args.index('--data-path')+1].startswith(str(runtime)))


if __name__ == '__main__':
    unittest.main()
