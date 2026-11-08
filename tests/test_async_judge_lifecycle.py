import json
from pathlib import Path
import sys
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Evaluation/services'))
import async_judge


def test_credentials_fall_back_to_existing_shared_config(tmp_path, monkeypatch):
    for key in ('JUDGE_API_KEY', 'REEVAL_CREDENTIALS', 'GLASSEEK_API_CONFIG'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(async_judge, 'E', tmp_path / 'Evaluation')
    monkeypatch.setenv('QUEST_ROOT', str(tmp_path / 'Quest'))
    target = tmp_path / 'Quest/workspace/strict-reproduction/.secrets/api_config.json'
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({'common': {'JUDGE_OPENAI_API_KEY': 'test-only-key'}}))
    assert async_judge.load_api_key() == 'test-only-key'
    monkeypatch.setenv('JUDGE_API_KEY', 'environment-test-key')
    assert async_judge.load_api_key() == 'environment-test-key'


def test_explicit_credential_path_is_not_silently_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv('JUDGE_API_KEY', raising=False)
    monkeypatch.setenv('REEVAL_CREDENTIALS', str(tmp_path / 'missing.json'))
    import pytest
    with pytest.raises(RuntimeError, match='credentials unavailable'):
        async_judge.load_api_key()


def test_exited_owner_drains_empty_run_without_model_calls(tmp_path, monkeypatch):
    config = Path(__file__).resolve().parents[1] / 'Evaluation/config/async_judge.json'
    monkeypatch.setattr(async_judge, 'E', tmp_path)
    monkeypatch.setattr(async_judge, 'load_api_key', lambda: 'test-only-key')
    monkeypatch.setattr(async_judge, 'owner_alive', lambda pid: False)
    client = Mock()
    monkeypatch.setattr(async_judge, 'JudgeClient', Mock(return_value=client))
    monkeypatch.setattr(sys, 'argv', ['async_judge', '--run', 'fixture', '--dataset', 'fixture',
        '--parent-pid', '99999999', '--config', str(config)])
    async_judge.main()
    client.judge.assert_not_called()
