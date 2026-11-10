"""Verify the shell-free CLI, worker budget and watchdog without inference."""
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run
from suite import ROOT
sys.path.insert(0, str(ROOT / 'Evaluation/services'))


class PythonCLITests(unittest.TestCase):
    def test_native_command_preserves_protocol_and_viewer(self):
        args = run.build_parser().parse_args([
            '--phase', 'all', '--per-benchmark', '2', '--workers', '5',
            '--context-manager', 'clm', '--url-mode', 'raw'])
        cmd, env = run.infer_command(args, 'gaia', 2)
        self.assertTrue(cmd[1].endswith('/native_eval.py'))
        self.assertNotIn('bash', cmd)
        self.assertFalse(any(x.endswith('.sh') for x in cmd))
        for flag, value in {'--num-workers': '5', '--workers-per-endpoint': '5',
                            '--expected-questions': '2', '--context-manager': 'clm',
                            '--url-mode': 'raw', '--context-length': '131072',
                            '--max-tokens': '16384'}.items():
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertIn('--viewer', cmd)
        self.assertIn('--endpoint-watchdog', cmd)
        self.assertIn('--ignore-rules', cmd)
        self.assertNotIn('--timeout-seconds', cmd)
        self.assertEqual(env['GLASSEEK_MAX_TOOL_CALLS'], '120')
        self.assertNotIn('HTTPS_PROXY', env)
        self.assertEqual(env['FETCH_SERVER_URL'], env['GLASSEEK_FETCH_URL'])
        args.viewer = False
        cmd, _ = run.infer_command(args, 'gaia', 2)
        self.assertIn('--no-viewer', cmd)

    def test_native_prepare_parses_all_generated_options_without_requests(self):
        import native_eval
        args = run.build_parser().parse_args(['--context-manager', 'clm', '--url-mode', 'raw'])
        cmd, env = run.infer_command(args, 'gaia', 2)
        with patch.dict(native_eval.os.environ, env), \
             patch('browse_comp_eval.dataset.load_dataset', return_value=[1, 2]), \
             patch.object(native_eval.os, 'access', return_value=True), \
             patch.object(native_eval, 'supervise') as supervise:
            self.assertEqual(native_eval.main(cmd[2:] + ['--prepare-only']), 0)
            self.assertEqual(native_eval.os.environ['GLASSEEK_URL_MODE'], 'raw')
            supervise.assert_not_called()

    def test_legacy_launcher_is_unchanged(self):
        from eval_run import build
        cmd, _ = build('base9b', 'full')
        self.assertEqual(cmd[:2], ['bash', str(ROOT / 'Evaluation/scripts/run_eval.sh')])

    def test_watchdog_preflight_failure_never_starts_inference(self):
        import native_eval
        args = SimpleNamespace(fetch_server_url='http://local', fetch_watchdog_target='https://example.com')
        with patch.object(native_eval, 'probe', side_effect=RuntimeError('unhealthy')), \
             patch.object(native_eval.subprocess, 'Popen') as popen:
            with self.assertRaises(RuntimeError):native_eval.supervise(['fake'], args, None)
            popen.assert_not_called()

    def test_watchdog_stops_group_and_marks_active_cases(self):
        import native_eval
        from itertools import count
        args = SimpleNamespace(fetch_server_url='http://local', fetch_watchdog_target='')
        runner = SimpleNamespace(data_path='input.jsonl', output_root='out', save_name='r',
                                 model='m', dataset='d', numbered_output=True)
        process = SimpleNamespace(pid=123, poll=lambda: None)
        ticks = count(0, 11)
        def health(*unused):
            if health.calls:
                raise RuntimeError('unhealthy')
            health.calls += 1
        health.calls = 0
        with patch.object(native_eval, 'probe', side_effect=health), \
             patch.object(native_eval.subprocess, 'Popen', return_value=process), \
             patch.object(native_eval.time, 'monotonic', side_effect=lambda: next(ticks)), \
             patch.object(native_eval.time, 'sleep'), patch.object(native_eval.os, 'killpg') as killpg, \
             patch.object(native_eval, 'terminate') as stop, \
             patch.object(native_eval, 'active_run_ids', return_value=['case__r1']), \
             patch.object(native_eval, 'mark_failed') as mark:
            self.assertEqual(native_eval.supervise(['fake'], args, runner), 75)
            stop.assert_called_with(process)
            mark.assert_called_once()
            self.assertEqual(mark.call_args.args[0].reason, 'health: unhealthy')
            self.assertEqual(killpg.call_args.args[0], 123)


if __name__ == '__main__':unittest.main()


def test_system_override_preserves_hle_and_user_protocol(tmp_path):
    prompt=tmp_path/'compact.txt';prompt.write_text('Be accurate.')
    args=run.build_parser().parse_args(['--system-prompt-file',str(prompt)])
    for bench in ['browsecomp','gaia','drbench','widesearch','liveresearchbench']:
        cmd,env=run.infer_command(args,bench,2)
        assert cmd[cmd.index('--custom-system-prompt-file')+1]==str(prompt)
        assert env['GLASSEEK_ANSWER_PROMPT_REGISTRY'].endswith('/prompts/registry.json')
    cmd,_=run.infer_command(args,'hle',2)
    assert cmd[cmd.index('--custom-system-prompt-file')+1].endswith('/prompts/hle-system.txt')


def test_arex_switches_prompt_pair_and_leaves_graders_separate(tmp_path, monkeypatch):
    from browse_comp_eval.hermes_client import build_query
    args=run.build_parser().parse_args(['--research-prompt','arex-native','--url-mode','raw','--output-root',str(tmp_path)])
    cmd,env=run.infer_command(args,'hle',1)
    assert 'arex-native' == env['GLASSEEK_RESEARCH_PROTOCOL']
    assert cmd[cmd.index('--tool-whitelist')+1]=='web,code_execution,vision'
    prompt=__import__('pathlib').Path(cmd[cmd.index('--custom-system-prompt-file')+1]).read_text()
    assert 'finish ALONE' in prompt and 'Your response should be in the following format' not in prompt
    monkeypatch.setenv('GLASSEEK_RESEARCH_PROTOCOL','arex-native')
    question=build_query(SimpleNamespace(item=SimpleNamespace(question='unmodified problem')), 'suite_hle')
    assert question.count('unmodified problem')==1 and 'finish' in question
    args.url_mode='ref'
    import pytest
    with pytest.raises(ValueError,match='raw URLs'):run.infer_command(args,'hle',1)
