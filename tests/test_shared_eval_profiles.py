import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Evaluation/services'))
import eval_run


def test_tool_config_and_overrides_reach_clients(tmp_path, monkeypatch):
    config = json.loads((ROOT / 'Evaluation/config/tool_services.json').read_text())
    config.update(fetch_port=29081, bridge_port=29217)
    path = tmp_path / 'tools.json'
    path.write_text(json.dumps(config))
    monkeypatch.setenv('GLASSEEK_TOOL_SERVICES_CONFIG', str(path))
    monkeypatch.delenv('GLASSEEK_FETCH_URL', raising=False)
    monkeypatch.delenv('GLASSEEK_QUEST_RETRIEVAL_URL', raising=False)
    argv, env = eval_run.build('sft9b', 'full')
    assert argv[argv.index('--fetch-server-url') + 1] == 'http://127.0.0.1:29081'
    assert env['GLASSEEK_QUEST_RETRIEVAL_URL'] == 'http://127.0.0.1:29217'
    monkeypatch.setenv('GLASSEEK_FETCH_URL', 'http://127.0.0.1:30081')
    assert eval_run.build('rl4', 'smoke')[0][-2:] == ['--limit', '1']
    argv, _ = eval_run.build('rl4', 'full')
    assert argv[argv.index('--fetch-server-url') + 1].endswith(':30081')


def test_judge_plan_needs_no_credentials(monkeypatch):
    monkeypatch.setenv('GLASSEEK_API_CONFIG', '/missing/private.json')
    monkeypatch.setenv('JUDGE_MODEL', 'fixture-model')
    argv, env = eval_run.build('rl9b', 'judge')
    assert argv[argv.index('--judge-model') + 1] == 'fixture-model'
    assert env['GLASSEEK_JUDGE_TERMINAL_ONLY'] == '1'


def test_runtime_contains_no_service_implementations():
    folder = ROOT / 'Evaluation/data/runtime/services'
    assert not [p for p in folder.glob('*') if p.is_file() and not p.is_symlink() and p.suffix in ('.py', '.sh')]
    assert not [p for p in folder.glob('*') if p.is_symlink() and p.suffix in ('.py', '.sh')]


def test_base9b_matches_step39_protocol_and_has_fixed_eight_workers(monkeypatch):
    monkeypatch.setenv('GLASSEEK_MAX_TOOL_CALLS', '999')
    argv, env = eval_run.build('base9b', 'full', {'run': 'fixture_base'})
    expected = {'--workers-per-endpoint': '8', '--expected-questions': '128', '--max-tokens': '16384',
                '--max-rounds': '200', '--max-retries': '1', '--context-length': '131072',
                '--context-compression': 'true', '--compression-threshold': '0.5', '--repeats': '1',
                '--save-name': 'fixture_base'}
    for flag, value in expected.items():
        assert argv[argv.index(flag) + 1] == value
    assert '--ramp-registry' not in argv and '--timeout-seconds' not in argv
    assert env['NO_TOTAL_TIMEOUT'] == '1' and env['GLASSEEK_MAX_TOOL_CALLS'] == '120'
    assert Path(argv[argv.index('--custom-system-prompt-file') + 1]).is_file()
    for mode in ('publish', 'watch-judge'):
        cmd, _ = eval_run.build('base9b', mode, {'run': 'fixture_base'})
        assert cmd[cmd.index('--run') + 1] == 'fixture_base'


def test_base_model_uses_same_parsers_with_eight_sequences():
    import model_server
    argv, _ = model_server.build('base9b', 0, 19109)
    for flag, value in {'--max-num-seqs': '8', '--max-model-len': '131072', '--kv-cache-dtype': 'fp8',
                        '--gpu-memory-utilization': '0.92', '--tool-call-parser': 'qwen3_coder'}.items():
        assert argv[argv.index(flag) + 1] == value
    assert json.loads(argv[argv.index('--default-chat-template-kwargs') + 1])['enable_thinking'] is True


def test_publisher_waits_for_first_snapshot_without_registering_missing_data(tmp_path, monkeypatch):
    import viewer_publish
    quest = tmp_path / 'Quest'
    registry = quest / 'web/trace-viewer/datasets/index.json'
    registry.parent.mkdir(parents=True)
    registry.write_text('{"datasets": []}')
    monkeypatch.setenv('QUEST_ROOT', str(quest))
    monkeypatch.setenv('GLASSEEK_RUNTIME_DIR', str(tmp_path / 'runtime'))
    monkeypatch.setattr(sys, 'argv', ['viewer_publish', '--run', 'fixture', '--dataset', 'quest_qwen27b_128',
                                     '--model', 'base', '--label', 'base', '--once'])
    calls = []
    monkeypatch.setattr(viewer_publish.subprocess, 'run', lambda argv, **kwargs: calls.append(argv))
    viewer_publish.main()
    assert len(calls) == 1
    assert json.loads(registry.read_text())['datasets'] == []


def test_publisher_registers_only_after_export_and_imports_existing_snapshot(tmp_path, monkeypatch):
    import viewer_publish
    quest = tmp_path / 'Quest'
    registry = quest / 'web/trace-viewer/datasets/index.json'
    registry.parent.mkdir(parents=True)
    registry.write_text('{"datasets": [{"id": "existing"}]}')
    monkeypatch.setenv('QUEST_ROOT', str(quest))
    monkeypatch.setenv('GLASSEEK_RUNTIME_DIR', str(tmp_path / 'runtime'))
    monkeypatch.setattr(sys, 'argv', ['viewer_publish', '--run', 'fixture', '--dataset', 'quest_qwen27b_128',
                                     '--model', 'base', '--label', 'base', '--once'])
    calls = []
    def invoke(argv, **kwargs):
        calls.append(argv)
        if '--output' in argv:
            assert kwargs['env']['GLASSEEK_VIEWER_MODEL'] == 'base'
            target = Path(argv[argv.index('--output') + 1])
            target.parent.mkdir(parents=True)
            target.write_text('{"cases": []}')
    monkeypatch.setattr(viewer_publish.subprocess, 'run', invoke)
    viewer_publish.main()
    assert len(calls) == 2
    assert [r['id'] for r in json.loads(registry.read_text())['datasets']] == ['fixture', 'existing']
