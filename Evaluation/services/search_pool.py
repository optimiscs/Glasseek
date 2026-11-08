"""Configuration and launcher for the shared asynchronous search service.

Only this service owns upstream credentials. Clients consume the env subcommand.
Relative paths in configuration are resolved against the Glasseek repository.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'Evaluation/config/search_pool.json'
PATH_KEYS = ('SERPER_KEY_POOL_FILE', 'SERPER_KEY_STATE_FILE')


def load_config(environ=None):
    env = os.environ if environ is None else environ
    path = Path(env.get('GLASSEEK_SEARCH_POOL_CONFIG', DEFAULT_CONFIG)).expanduser().resolve()
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError('Search pool configuration must be an object')
    config = {key: str(env.get(key, value)) for key, value in config.items()}
    return validate_config(config, env, path)


def validate_config(config, env, path):
    required = set(json.loads(DEFAULT_CONFIG.read_text()))
    if required - config.keys():
        raise ValueError('Missing search pool settings: ' + ', '.join(sorted(required - config.keys())))
    for key in PATH_KEYS:
        value = Path(config[key]).expanduser()
        config[key] = str(value if value.is_absolute() else ROOT / value)
    for key in ('SEARCH_POOL_PORT', 'SERPER_PROXY_CONCURRENCY', 'SERPER_KEY_CONCURRENCY', 'SEARCH_SERVER_MAX_ATTEMPTS'):
        if int(config[key]) < 1:
            raise ValueError(f'{key} must be a positive integer')
    if int(config['SEARCH_POOL_PORT']) > 65535:
        raise ValueError('Invalid search pool port')
    if int(config['SEARCH_SERVER_MAX_ATTEMPTS']) != 1:
        raise ValueError('Client retries must be 1; the shared pool owns upstream retries')
    if int(config['SERPER_PROXY_QUEUE_LIMIT']) < 0 or float(config['SERPER_PROXY_RPS']) < 0:
        raise ValueError('Queue limit and global RPS must be nonnegative')
    if not math.isfinite(float(config['SERPER_PROXY_RPS'])):
        raise ValueError('Global RPS must be finite')
    for key in ('SERPER_KEY_RPS', 'SERPER_PROXY_QUEUE_TIMEOUT', 'SERPER_UPSTREAM_BUDGET',
                'SERPER_UPSTREAM_TIMEOUT', 'SEARCH_SERVER_TIMEOUT', 'SEARCH_POOL_LEASE_SECONDS'):
        value = float(config[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('SEARCH_POOL_WORKERS', 'SEARCH_POOL_RESULT_TTL'):
        if int(config[key]) < 1:
            raise ValueError(f'{key} must be positive')
    if int(config['SEARCH_POOL_MAX_UPSTREAM_ATTEMPTS']) < 0:
        raise ValueError('Attempt budget must be nonnegative')
    if float(config['SEARCH_POOL_LEASE_SECONDS']) <= float(config['SERPER_UPSTREAM_TIMEOUT']) + .1:
        raise ValueError('Lease must exceed the total HTTP timeout plus recovery margin')
    if config['SEARCH_POOL_REQUIRE_DURABILITY'] not in ('0', '1'):
        raise ValueError('SEARCH_POOL_REQUIRE_DURABILITY must be 0 or 1')
    if float(config['SEARCH_SERVER_TIMEOUT']) <= 2:
        raise ValueError('Client timeout must leave a two-second response margin')
    if float(config['SERPER_PROXY_QUEUE_TIMEOUT']) >= float(config['SEARCH_SERVER_TIMEOUT'])-2:
        raise ValueError('Queue timeout must leave execution and response time')
    if not config['SEARCH_POOL_NAMESPACE'] or any(c in config['SEARCH_POOL_NAMESPACE'] for c in '{}'):
        raise ValueError('Invalid Redis namespace')
    upstream = urlsplit(config['SERPER_UPSTREAM_URL'])
    if (upstream.scheme not in ('https', 'http') or not upstream.hostname or upstream.username
            or upstream.password or upstream.fragment):
        raise ValueError('Invalid upstream URL')
    if upstream.scheme == 'http' and upstream.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Plain HTTP upstream is only allowed for local mock tests')
    endpoint = env.get('SEARCH_SERVER_ENDPOINT') or config.get('SEARCH_SERVER_ENDPOINT')
    if not endpoint:
        host = config['SEARCH_POOL_HOST']
        host = '127.0.0.1' if host in ('0.0.0.0', '::') else host
        host = f'[{host}]' if ':' in host else host
        endpoint = f'http://{host}:{config["SEARCH_POOL_PORT"]}/search'
    parsed = urlsplit(endpoint)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('SEARCH_SERVER_ENDPOINT must be an HTTP(S) service URL without credentials')
    if parsed.hostname == 'google.serper.dev':
        raise ValueError('Clients must use the shared search pool, not Serper directly')
    config['SEARCH_SERVER_ENDPOINT'] = endpoint
    config['GLASSEEK_SEARCH_POOL_CONFIG'] = str(path)
    return config


def client_env(config):
    # Pin the endpoint separately: Hermes reloads profile dotenv files lazily.
    return {
        'GLASSEEK_SEARCH_POOL_CONFIG': config['GLASSEEK_SEARCH_POOL_CONFIG'],
        'GLASSEEK_SEARCH_POOL_ENDPOINT': config['SEARCH_SERVER_ENDPOINT'],
        'GLASSEEK_SEARCH_POOL_TIMEOUT': config['SEARCH_SERVER_TIMEOUT'],
        'SEARCH_SERVER_ENDPOINT': config['SEARCH_SERVER_ENDPOINT'],
        'SEARCH_SERVER_TIMEOUT': config['SEARCH_SERVER_TIMEOUT'],
        'SEARCH_SERVER_MAX_ATTEMPTS': config['SEARCH_SERVER_MAX_ATTEMPTS'],
        'SEARCH_SERVER_API_KEY': 'shared-pool',
    }


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def implementation_revision():
    directory = Path(__file__).resolve().parent
    files = ('serper_proxy.py', 'provider_pool.py', 'provider_services.py', 'search_pool_state.py', 'search_pool_scripts/state.lua', 'search_pool.py')
    return hashlib.sha256(b''.join((directory / name).read_bytes() for name in files)).hexdigest()[:16]


def migrate_runtime(config, key_source=None, state_source=None):
    """Explicit, idempotent migration; never overwrite existing credential/state files."""
    for source, name in ((key_source, 'SERPER_KEY_POOL_FILE'), (state_source, 'SERPER_KEY_STATE_FILE')):
        target = Path(config[name])
        if source and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            # Exclusive creation prevents a rerun from overwriting live state.
            with Path(source).open('rb') as src:
                payload = src.read()
                if name == 'SERPER_KEY_POOL_FILE':
                    keys = json.loads(payload).get('serper')
                    if not isinstance(keys, list) or not keys or any(not isinstance(k, str) or not k.strip() for k in keys):
                        raise ValueError('Source must contain a nonempty serper key list')
                    payload = json.dumps({'serper': keys}).encode()
                fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, 'wb') as dst:
                    dst.write(payload)
    if state_source:
        source_usage = Path(state_source).with_name('search-key-usage.jsonl')
        target_usage = Path(config['SERPER_KEY_STATE_FILE']).with_name('search-key-usage.jsonl')
        if source_usage.is_file() and not target_usage.exists():
            target_usage.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(target_usage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as dst, source_usage.open('rb') as src:
                shutil.copyfileobj(src, dst)
    legacy = ROOT / 'Evaluation/data/runtime/services/serper_proxy.py'
    canonical = ROOT / 'Evaluation/services/serper_proxy.py'
    if legacy.is_symlink() and legacy.resolve() == canonical:
        legacy.unlink()
        return
    if legacy.is_symlink():
        raise ValueError('Unrecognized runtime link; refusing to remove it')
    if legacy.exists():
        text = legacy.read_text()
        if "google.serper.dev" not in text or 'def upstream_search' not in text:
            raise ValueError('Unrecognized runtime implementation; refusing to replace it')
        legacy.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('serve', 'env', 'check', 'migrate-runtime', 'sync-keys', 'restore-key', 'configure'))
    parser.add_argument('--json', action='store_true', help='emit client environment as JSON')
    parser.add_argument('--key-pool-source', type=Path)
    parser.add_argument('--state-source', type=Path)
    parser.add_argument('--key-id', help='Full SHA256 credential fingerprint; never the credential itself')
    args = parser.parse_args()
    config = load_config()
    if args.command == 'env':
        values = client_env(config)
        if args.json:
            print(json.dumps(values))
        else:
            for key, value in values.items():
                print(f'export {key}={shlex.quote(value)}')
    elif args.command == 'migrate-runtime':
        migrate_runtime(config, args.key_pool_source, args.state_source)
        print('Search pool runtime migration complete; existing target files preserved.')
    elif args.command == 'check':
        import urllib.request
        endpoint = config['SEARCH_SERVER_ENDPOINT'].rsplit('/', 1)[0] + '/health'
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(endpoint, timeout=5) as response:
            health = json.load(response)
        revision = implementation_revision()
        if (health.get('service') != 'glasseek-search-pool'
                or health.get('config_digest') != config_digest(config)
                or health.get('revision') != revision):
            raise SystemExit('Search pool implementation/configuration differs from the selected configuration')
        print(json.dumps({k: health[k] for k in ('service', 'revision', 'config_digest', 'eligible_keys', 'pending')}))
    elif args.command in ('sync-keys', 'restore-key', 'configure'):
        import asyncio
        from search_pool_state import PoolState, read_keys, scheduler_config
        async def administer():
            state = PoolState(config)
            try:
                if args.command == 'configure':
                    cfg = scheduler_config(config)
                    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
                    if not await state.redis.exists(state.prefix + 'config'):
                        raise ValueError('Initialize the search service before reconfiguring it')
                    await state.call('configure', config=cfg, digest=digest)
                    print('Shared configuration updated; restart all replicas with this configuration')
                    return
                await state.initialize()
                if args.command == 'sync-keys':
                    keys = read_keys(config['SERPER_KEY_POOL_FILE'])
                    await state.call('sync_keys', keys=keys)
                    print(json.dumps({'synced_keys': len(keys)}))
                else:
                    if not args.key_id or len(args.key_id) != 64 or any(c not in '0123456789abcdef' for c in args.key_id):
                        raise ValueError('restore-key requires a full lowercase SHA256 --key-id')
                    restored = await state.call('restore_key', key=args.key_id)
                    if not restored:
                        raise ValueError('Credential fingerprint is not in this pool')
                    print('Credential restored in shared state')
            finally:
                await state.close()
        asyncio.run(administer())
    else:
        os.environ.update(config)
        # Preserve the public CLI while using the common multi-provider launcher.
        import asyncio
        from provider_services import serve
        asyncio.run(serve(['serper']))


if __name__ == '__main__':
    main()
