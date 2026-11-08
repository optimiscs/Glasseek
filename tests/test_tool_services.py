"""Verify shared routing and actually boot the relocated fetch application."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'Evaluation/services/tool_services.py'
spec = importlib.util.spec_from_file_location('tool_services', LAUNCHER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_search_delegates_and_preserves_local_proxy_bypass(monkeypatch):
    monkeypatch.setenv('NO_PROXY', 'custom.internal')
    config = module.load_config()
    env = module.environment(config)
    assert 'custom.internal' in env['NO_PROXY']
    assert '127.0.0.1' in env['NO_PROXY']
    assert module.command('serper', config, env)[1:] == [str(ROOT / 'Evaluation/services/provider_services.py'), 'serve', '--services', 'serper']
    assert module.command('search', config, env) == module.command('serper', config, env)
    assert not any('KEY' in key or 'CREDENTIAL' in key for key in env)


def test_bind_and_client_urls_share_override(monkeypatch):
    monkeypatch.setenv('GLASSEEK_QUEST_RETRIEVAL_URL', 'http://127.0.0.1:29017')
    monkeypatch.setenv('GLASSEEK_FETCH_URL', 'http://127.0.0.1:29081')
    monkeypatch.setenv('VISIT_BM25_POOL_URL', 'http://remote.example:18215')
    config = module.load_config()
    env = module.environment(config)
    assert env['GLASSEEK_FETCH_SAFETY_URL'] == 'http://127.0.0.1:29081/validate'
    assert module.command('bridge', config, env)[-1] == '29017'
    with pytest.raises(ValueError, match='binds only'):
        module.command('bm25', config, env)


def test_fetch_boots_from_unrelated_cwd_without_legacy_runtime_source(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts/validation'))
    from fetch_pool_load import FetchSandbox
    # Real Redis and mock Reader; never attach a test key to production state.
    old = os.getcwd()
    try:
        os.chdir(tmp_path)
        with FetchSandbox(tmp_path, replicas=1) as sandbox:
            health = sandbox.health()
            assert health['status'] == 'ok'
            assert health['service'] == 'glasseek-fetch-pool'
            response = sandbox.http.post(sandbox.urls[0]+'/fetch', json={'url':'file:///invalid'})
            assert response.status_code == 400 and response.json()['success'] is False
    finally:
        os.chdir(old)
