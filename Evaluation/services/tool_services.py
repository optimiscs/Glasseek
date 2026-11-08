"""Shared tool service launcher. Search always delegates to search_pool.py.

Configuration contains no credentials. `command` and `env` never read secrets.
Serve runs in the foreground so existing supervisors retain process ownership.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
from urllib.parse import urlsplit

EVALUATION = Path(__file__).resolve().parents[1]
SERVICES = ('search', 'serper', 'fetch', 'bm25', 'embedding', 'bridge')


def load_config():
    path = Path(os.environ.get('GLASSEEK_TOOL_SERVICES_CONFIG', EVALUATION / 'config/tool_services.json'))
    return json.loads(path.read_text())


def environment(config):
    host = config['host']
    defaults = {
        'VISIT_BM25_POOL_URL': f"http://{host}:{config['bm25_port']}",
        'VISIT_EMBEDDING_URL': f"http://{host}:{config['embedding_port']}",
        'GLASSEEK_EMBEDDING_UPSTREAM': f"http://{host}:{config['embedding_upstream_port']}",
        'GLASSEEK_FETCH_URL': f"http://{host}:{config['fetch_port']}",
        'GLASSEEK_FETCH_SAFETY_URL': f"http://{host}:{config['fetch_port']}/validate",
        'GLASSEEK_FETCH_POOL_CLIENT': '1',
        'GLASSEEK_QUEST_PAGE_RETRIEVAL': '1',
        'GLASSEEK_QUEST_RETRIEVAL_URL': f"http://{host}:{config['bridge_port']}",
        'GLASSEEK_QUEST_RETRIEVAL_TIMEOUT': str(config['retrieval_timeout']),
        'VISIT_RETRIEVAL_HTTP_TIMEOUT': str(max(1, config['retrieval_timeout'] - 15)),
        'VISIT_RETRIEVAL_CANDIDATES': str(config['retrieval_candidates']),
        'VISIT_RETRIEVAL_TOP_K': str(config['retrieval_top_k']),
        'GLASSEEK_RETRIEVAL_CONCURRENCY': str(config['retrieval_concurrency']),
        'HTTPS_PROXY': config['proxy_url'], 'HTTP_PROXY': config['proxy_url'],
        'GLASSEEK_DOH_PROXY': config['proxy_url'],
        'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false',
    }
    result = {key: os.environ.get(key, value) for key, value in defaults.items()}
    if not os.environ.get('GLASSEEK_FETCH_SAFETY_URL'):
        result['GLASSEEK_FETCH_SAFETY_URL'] = result['GLASSEEK_FETCH_URL'].rstrip('/') + '/validate'
    bypass = os.environ.get('NO_PROXY', os.environ.get('no_proxy', '')).split(',')
    result['NO_PROXY'] = result['no_proxy'] = ','.join(dict.fromkeys(
        entry for entry in [*bypass, 'localhost', '127.0.0.1'] if entry))
    return result


def command(name, config, env):
    quest = Path(os.environ.get('QUEST_ROOT', EVALUATION.parent.parent / 'Quest'))
    python = os.environ.get('TOOL_SERVICES_PYTHON', os.environ.get('QUEST_PYTHON', str(quest / 'workspace/env/bin/python')))
    def address(key):
        url = urlsplit(env[key])
        if url.scheme != 'http' or not url.hostname or not url.port or url.path not in ('', '/'):
            raise ValueError(f'{key} must be an HTTP origin with an explicit port')
        return url.hostname, str(url.port)
    if name in ('search', 'serper', 'fetch'):
        return [python, str(EVALUATION / 'services/provider_services.py'), 'serve', '--services',
                'jina' if name == 'fetch' else 'serper']
    if name in ('fetch', 'bridge'):
        host, port = address('GLASSEEK_FETCH_URL' if name == 'fetch' else 'GLASSEEK_QUEST_RETRIEVAL_URL')
        module = 'jina_fetch:app' if name == 'fetch' else 'quest_retrieval_bridge:app'
        return [python, '-m', 'uvicorn', module, '--app-dir', str(EVALUATION / 'services'), '--host', host, '--port', port]
    if name == 'bm25':
        host, port = address('VISIT_BM25_POOL_URL')
        if host != '127.0.0.1':
            raise ValueError('Quest BM25 launcher binds only to 127.0.0.1')
        return [python, str(quest / 'inference/bm25_pool_service.py'),
                '--workers', str(config['bm25_workers']), '--queue-limit', str(config['bm25_queue_limit']), '--port', port]
    host, port = address('VISIT_EMBEDDING_URL')
    if host != '127.0.0.1':
        raise ValueError('Quest embedding adapter binds only to 127.0.0.1')
    return [python, str(quest / 'inference/retrieval_vllm_adapter.py'), '--role', 'embedding',
            '--model', config['embedding_model'], '--upstream', env['GLASSEEK_EMBEDDING_UPSTREAM'], '--port', port]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('serve', 'command', 'env'))
    parser.add_argument('service', nargs='?', choices=SERVICES)
    args = parser.parse_args()
    config = load_config()
    env = environment(config)
    if args.action == 'env':
        for key, value in env.items():
            print(f'export {key}={shlex.quote(value)}')
        return
    if not args.service:
        parser.error('serve/command requires a service')
    argv = command(args.service, config, env)
    if args.action == 'command':
        print(json.dumps(argv))
        return
    os.execvpe(argv[0], argv, dict(os.environ, **env))


if __name__ == '__main__':
    main()
