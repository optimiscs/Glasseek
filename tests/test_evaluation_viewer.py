"""Evaluator-owned publishing: no model calls or production services required."""
import json
import signal
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Evaluation'))
from browse_comp_eval import viewer


def make_viewer(tmp_path, **kwargs):
    return viewer.EvaluationViewer(run='run.1', dataset='sample', model='test',
        output=tmp_path / 'results', data_path=tmp_path / 'questions.jsonl',
        count=2, workers=2, **kwargs)


def test_auto_start_preserves_eval_inputs_and_stops_with_final_refresh(tmp_path, monkeypatch):
    process = Mock(pid=12345)
    process.poll.return_value = None
    popen = Mock(return_value=process)
    monkeypatch.setattr(viewer.subprocess, 'Popen', popen)
    monkeypatch.setenv('GLASSEEK_URL_MODE', 'ref')
    service = make_viewer(tmp_path, parent_pid=999, interval=7).start()
    argv = popen.call_args.args[0]
    env = popen.call_args.kwargs['env']
    assert argv[argv.index('--parent-pid') + 1] == '999'
    assert env['GLASSEEK_VIEWER_OUTPUT_DIR'] == str(tmp_path / 'results')
    assert env['GLASSEEK_VIEWER_DATA_PATH'] == str(tmp_path / 'questions.jsonl')
    assert env['GLASSEEK_VIEWER_LIMIT'] == '2'
    assert env['GLASSEEK_URL_MODE'] == 'ref'
    metadata = json.loads((tmp_path / 'results/viewer.json').read_text())
    assert metadata['refresh_seconds'] == 7 and metadata['automatic']
    service.close()
    process.send_signal.assert_called_once_with(signal.SIGTERM)
    process.wait.assert_called_once_with(timeout=60)


def test_opt_out_does_not_create_files_or_process(tmp_path, monkeypatch):
    popen = Mock()
    monkeypatch.setattr(viewer.subprocess, 'Popen', popen)
    service = make_viewer(tmp_path, enabled=False).start()
    service.close()
    popen.assert_not_called()
    assert not (tmp_path / 'results').exists()


def test_publish_failure_does_not_abort_evaluation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(viewer.subprocess, 'Popen', Mock(side_effect=OSError('unavailable')))
    service = make_viewer(tmp_path).start()
    service.close()
    assert 'could not start' in capsys.readouterr().err


def test_web_service_reuses_existing_listener(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Evaluation/services'))
    import viewer_publish
    quest = tmp_path / 'Quest'
    (quest / 'web').mkdir(parents=True)
    (quest / 'web/serve.py').touch()
    lock_dir = tmp_path / 'locks'
    lock_dir.mkdir()
    from unittest.mock import MagicMock
    monkeypatch.setattr(viewer_publish.socket, 'create_connection', MagicMock())
    popen = Mock()
    monkeypatch.setattr(viewer_publish.subprocess, 'Popen', popen)
    viewer_publish.ensure_web_service(quest, lock_dir)
    popen.assert_not_called()


def test_parent_exit_performs_one_final_export_then_stops(tmp_path, monkeypatch):
    import viewer_publish
    quest = tmp_path / 'Quest'
    registry = quest / 'web/trace-viewer/datasets/index.json'
    registry.parent.mkdir(parents=True)
    registry.write_text('{"datasets": []}')
    monkeypatch.setenv('QUEST_ROOT', str(quest))
    monkeypatch.setenv('GLASSEEK_RUNTIME_DIR', str(tmp_path / 'runtime'))
    monkeypatch.setattr(sys, 'argv', ['viewer_publish', '--run', 'fixture', '--dataset', 'dataset',
        '--model', 'model', '--label', 'label', '--parent-pid', '99999999'])
    monkeypatch.setattr(viewer_publish.signal, 'signal', Mock())
    monkeypatch.setattr(viewer_publish.os, 'kill', Mock(side_effect=ProcessLookupError()))
    calls = []
    def invoke(argv, **kwargs):
        calls.append(argv)
        if '--output' in argv:
            path = Path(argv[argv.index('--output') + 1])
            path.parent.mkdir(parents=True)
            path.write_text('{"run": {"requestedCases": 34}, "cases": []}')
    monkeypatch.setattr(viewer_publish.subprocess, 'run', invoke)
    viewer_publish.main()
    assert len(calls) == 2
    assert calls[1][-1] == 'fixture' and '-c' in calls[1]
    assert json.loads(registry.read_text())['datasets'][0]['benchmark'] == 'dataset · 34 题'
