"""Exercise real HTTP routing for all three consumers without paid upstream calls."""
import concurrent.futures
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVICES = ROOT / 'Evaluation/services'
sys.path.insert(0, str(SERVICES))
import search_pool


def test_config_is_cwd_independent_and_client_env_contains_no_upstream_secrets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = search_pool.load_config({})
    assert cfg['SERPER_KEY_POOL_FILE'] == str(ROOT / 'Evaluation/data/runtime/search_pool/keys.json')
    assert cfg['SERPER_KEY_CONCURRENCY'] == '2'
    cfg = search_pool.load_config({'SEARCH_SERVER_ENDPOINT': 'http://pool.internal:18083/search',
                                   'SERPER_KEY_CONCURRENCY': '3'})
    env = search_pool.client_env(cfg)
    assert env['GLASSEEK_SEARCH_POOL_ENDPOINT'] == 'http://pool.internal:18083/search'
    assert not any(k.startswith('SERPER_') for k in env)
    assert env['SEARCH_SERVER_MAX_ATTEMPTS'] == '1'
    with pytest.raises(ValueError):
        search_pool.load_config({'SEARCH_SERVER_ENDPOINT': 'https://google.serper.dev/search'})
    with pytest.raises(ValueError):
        search_pool.load_config({'SERPER_KEY_RPS': 'nan'})


def test_migration_removes_duplicate_and_preserves_existing_state(tmp_path, monkeypatch):
    monkeypatch.setattr(search_pool, 'ROOT', tmp_path)
    canonical = tmp_path / 'Evaluation/services/serper_proxy.py'
    canonical.parent.mkdir(parents=True)
    canonical.write_text('# canonical implementation\n')
    legacy = tmp_path / 'Evaluation/data/runtime/services/serper_proxy.py'
    legacy.parent.mkdir(parents=True)
    legacy.write_text("# google.serper.dev\ndef upstream_search(): pass\n")
    keys = tmp_path / 'source.json'
    keys.write_text(json.dumps({'serper': ['test-key'], 'unrelated': ['do-not-copy']}))
    state = tmp_path / 'source-state.json'
    state.write_text('{"fingerprint":{"reason":"quota_exhausted"}}')
    cfg = {'SERPER_KEY_POOL_FILE': str(tmp_path / 'pool/keys.json'),
           'SERPER_KEY_STATE_FILE': str(tmp_path / 'pool/state.json')}
    search_pool.migrate_runtime(cfg, keys, state)
    assert not legacy.exists() and not legacy.is_symlink()
    assert json.loads(Path(cfg['SERPER_KEY_POOL_FILE']).read_text()) == {'serper': ['test-key']}
    assert Path(cfg['SERPER_KEY_POOL_FILE']).stat().st_mode & 0o777 == 0o600
    Path(cfg['SERPER_KEY_STATE_FILE']).write_text('{"new-state":true}')
    search_pool.migrate_runtime(cfg, keys, state)
    assert json.loads(Path(cfg['SERPER_KEY_STATE_FILE']).read_text()) == {'new-state': True}


@pytest.fixture
def running_pool(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts/validation'))
    from search_pool_load import Sandbox
    with Sandbox(tmp_path, replicas=2, overrides={'SERPER_PROXY_CONCURRENCY':4}) as sandbox:
        class Pool:
            env=sandbox.env
            health=sandbox.health()
            endpoint=sandbox.env['SEARCH_SERVER_ENDPOINT']
            tmp=tmp_path
            @property
            def calls(self):
                return [{'q':row['q']} for row in sandbox.stats()['rows']]
            def peak(self):
                return sandbox.stats()['peak'].get('total',0)
        yield Pool()


def run_consumer(kind, query, env):
    if kind == 'training':
        code = f'''
import sys,json
sys.path.insert(0,{str(ROOT / 'RL_training')!r})
from glasseek_training.web_tools import WebTools,search_pool_worker_env
assert not any(k.startswith('SERPER_') for k in search_pool_worker_env())
print(WebTools()._execute({{'name':'web_search','arguments':{{'query':{query!r}}}}}))
'''
    else:
        backend = ROOT / ('Evaluation_backend' if kind == 'evaluation' else 'Data_curation/multi_agent_mpcs_v5_publish/hermes')
        setup = ''
        if kind == 'curation':
            workflow_env = Path(env['SERPER_KEY_STATE_FILE']).parent / 'stale-workflow.env'
            workflow_env.write_text('SEARCH_SERVER_ENDPOINT=http://127.0.0.1:1/search\n')
            setup = f'''
sys.path.insert(0,{str(backend.parent)!r})
from browsecomp_v2.config import load_config
load_config({str(workflow_env)!r})
'''
        code = f'''
import sys,os,json
{setup}
sys.path.insert(0,{str(backend)!r})
# Simulate a stale profile dotenv overwriting the generic endpoint.
os.environ['SEARCH_SERVER_ENDPOINT']='http://127.0.0.1:1/search'
os.environ['SEARCH_SERVER_API_KEY']='old-profile-value'
from tools.web_tools import _get_search_backend
from plugins.web.search_server.provider import _search_server_response
assert _get_search_backend()=='search_server'
print(json.dumps(_search_server_response({query!r})))
'''
    client_env = {k: v for k, v in env.items() if not k.startswith('SERPER_')}
    client_env['HERMES_HOME'] = str(Path(env['SERPER_KEY_STATE_FILE']).parent / ('hermes-' + kind))
    result = subprocess.run([sys.executable, '-c', code], env=client_env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_three_consumers_share_one_pool_and_bounded_upstream(running_pool):
    pool = running_pool
    kinds = ['training', 'evaluation', 'curation']
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda kind: run_consumer(kind, kind, pool.env), kinds))
    assert results[0]['organic'][0]['snippet'] == 'training'
    assert results[1]['data']['web'][0]['description'] == 'evaluation'
    assert results[2]['data']['web'][0]['description'] == 'curation'
    assert sorted(c['q'] for c in pool.calls) == sorted(kinds)
    assert 1 <= pool.peak() <= 2
    assert pool.health['service'] == 'glasseek-search-pool'
    checked = subprocess.run([sys.executable, str(SERVICES / 'search_pool.py'), 'check'],
                             env=pool.env, capture_output=True, text=True, timeout=10)
    assert checked.returncode == 0, checked.stderr


def test_all_clients_preserve_errors_without_replay_and_keep_zero_results(running_pool):
    for kind in ('training', 'evaluation', 'curation'):
        error = run_consumer(kind, 'invalid', running_pool.env)
        assert error['success'] is False
        assert error['error_code'] == 'search_invalid_request'
        assert error['retryable'] is False
        empty = run_consumer(kind, 'empty', running_pool.env)
        assert empty.get('organic', empty.get('data', {}).get('web')) == []
    assert [c['q'] for c in running_pool.calls].count('invalid') == 3
    assert len(running_pool.calls) == 6



def test_eval_mock_and_disabled_do_not_leak_real_pool(monkeypatch):
    sys.path.insert(0, str(ROOT / 'Evaluation'))
    from browse_comp_eval.hermes_client import HermesClient, HermesConfig
    from browse_comp_eval.dataset import DatasetItem, RunSpec
    monkeypatch.setenv('GLASSEEK_SEARCH_POOL_ENDPOINT', 'http://real-pool:18083/search')
    spec = RunSpec('q1', 'run', 1, DatasetItem('q1', 'question', 'answer', 'test', 1))
    for mode in ('disabled', 'mock', 'external'):
        cfg = HermesConfig(hermes_bin='hermes', api_key='test', base_url='', model='test',
            provider='custom', save_name='test', dataset='test', max_rounds=1,
            timeout_seconds=10, toolsets=[], skills=[], search_mode=mode)
        client = HermesClient(cfg)
        mock = SimpleNamespace(endpoint='http://mock:19000/search', api_key='mock-key') if mode == 'mock' else None
        env = client._build_env(spec, mock_search_server=mock)
        if mode != 'external':
            assert 'GLASSEEK_SEARCH_POOL_ENDPOINT' not in env
        if mode == 'disabled':
            assert 'SEARCH_SERVER_ENDPOINT' not in env
        elif mode == 'mock':
            assert env['SEARCH_SERVER_ENDPOINT'] == mock.endpoint
        else:
            assert env['GLASSEEK_SEARCH_POOL_ENDPOINT'] == search_pool.load_config()['SEARCH_SERVER_ENDPOINT']


def test_training_requires_pool_not_upstream_key_and_forwards_routing(monkeypatch):
    sys.path.insert(0, str(ROOT / 'RL_training'))
    from glasseek_training.web_tools import WebTools, search_pool_worker_env
    from glasseek_training.validation import validate_environment
    monkeypatch.delenv('GLASSEEK_SEARCH_POOL_ENDPOINT', raising=False)
    monkeypatch.delenv('SEARCH_SERVER_ENDPOINT', raising=False)
    monkeypatch.setenv('SERPER_API_KEY', 'unused-old-key')
    with pytest.raises(ValueError, match='search_pool_env'):
        WebTools()
    monkeypatch.setenv('GLASSEEK_SEARCH_POOL_ENDPOINT', 'http://shared-pool:18083/search')
    monkeypatch.delenv('SERPER_API_KEY')
    validate_environment(SimpleNamespace(cm_gspo=SimpleNamespace(reward_mode='exact_match')))
    assert WebTools().search_endpoint == 'http://shared-pool:18083/search'
    assert search_pool_worker_env()['SEARCH_SERVER_ENDPOINT'] == 'http://shared-pool:18083/search'
    assert 'SERPER_API_KEY' not in search_pool_worker_env()
