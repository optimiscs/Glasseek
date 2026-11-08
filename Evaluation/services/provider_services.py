"""One configurable launcher and administration surface for all provider pools."""
import argparse
import asyncio
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
from urllib.parse import urlsplit
from search_pool import ROOT, DEFAULT_CONFIG, validate_config

REGISTRY = ROOT / 'Evaluation/config/provider_services.json'
FIELD_MAP = {
    'concurrency':'SERPER_PROXY_CONCURRENCY', 'queue_limit':'SERPER_PROXY_QUEUE_LIMIT',
    'workers':'SEARCH_POOL_WORKERS', 'per_key_concurrency':'SERPER_KEY_CONCURRENCY',
    'per_key_rps':'SERPER_KEY_RPS', 'global_rps':'SERPER_PROXY_RPS',
    'queue_timeout':'SERPER_PROXY_QUEUE_TIMEOUT', 'upstream_timeout':'SERPER_UPSTREAM_TIMEOUT',
    'upstream_budget':'SERPER_UPSTREAM_BUDGET', 'client_timeout':'SEARCH_SERVER_TIMEOUT',
    'lease_seconds':'SEARCH_POOL_LEASE_SECONDS', 'result_ttl':'SEARCH_POOL_RESULT_TTL',
    'max_upstream_attempts':'SEARCH_POOL_MAX_UPSTREAM_ATTEMPTS',
    'redis_url':'SEARCH_POOL_REDIS_URL', 'namespace':'SEARCH_POOL_NAMESPACE',
    'keys_file':'SERPER_KEY_POOL_FILE', 'upstream_url':'SERPER_UPSTREAM_URL',
}


def registry():
    return json.loads(REGISTRY.read_text())


def load_service(name):
    spec = registry()[name]
    if spec['format'] == 'legacy-search':
        from search_pool import load_config
        return load_config()
    path = Path(os.environ.get(spec['config_env'], REGISTRY.parent / spec['config']))
    values = json.loads(path.read_text())
    config = json.loads(DEFAULT_CONFIG.read_text())
    config.update({target:str(values[source]) for source,target in FIELD_MAP.items()})
    key_path = Path(values['keys_file'])
    config.update(POOL_PROVIDER=spec['name'], POOL_KEY_FIELD=spec['key_field'],
                  POOL_PROXY=values.get('proxy_url',''),
                  POOL_DNS_SAFETY='1' if values.get('dns_safety') else '0',
                  SERPER_KEY_STATE_FILE=str(key_path.with_name('legacy-state.json')))
    if spec.get('requires_dns_safety') and not values.get('dns_safety'):
        if not values['upstream_url'].startswith('http://127.0.0.1:'):
            raise ValueError('DNS safety can only be disabled for a loopback mock upstream')
    from tool_services import environment, load_config as tool_config
    origin = os.environ.get(spec['routing_env']) or environment(tool_config())[spec['routing_env']]
    address = urlsplit(origin)
    if address.scheme != 'http' or not address.hostname or not address.port or address.username:
        raise ValueError('Provider bind URL must be an HTTP origin with explicit port')
    config.update(SEARCH_POOL_HOST=address.hostname,SEARCH_POOL_PORT=str(address.port))
    return validate_config(config, {'SEARCH_SERVER_ENDPOINT':origin.rstrip('/')+'/'+spec['name']}, path)


def provision_keys(config):
    """First migration only; existing central key state is never reset on restart."""
    if config.get('POOL_KEY_FIELD') != 'jina':
        return
    target=Path(config['SERPER_KEY_POOL_FILE'])
    if target.exists():return
    keys=[k.strip() for k in os.environ.get('JINA_API_KEYS','').split(',') if k.strip()]
    if not keys:
        quest=Path(os.environ.get('QUEST_ROOT', ROOT.parent/'Quest'))
        source=Path(os.environ.get('TOOL_SERVICES_CREDENTIALS',os.environ.get('REEVAL_CREDENTIALS',
            str(quest/'workspace/strict-reproduction/.secrets/api_key_pool.json'))))
        keys=json.loads(source.read_text())['jina']
    target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as output:json.dump({'jina':keys},output)


async def serve(names):
    import uvicorn
    servers=[];configs=[load_service(name) for name in names]
    if len({c['SEARCH_POOL_NAMESPACE'] for c in configs})!=len(configs):
        raise ValueError('Services must have distinct Redis namespaces')
    if len({(c['SEARCH_POOL_HOST'],c['SEARCH_POOL_PORT']) for c in configs})!=len(configs):
        raise ValueError('Services must have distinct listener addresses')
    for name,config in zip(names,configs):
        provision_keys(config)
        module=importlib.import_module(registry()[name]['module'])
        app=module.create_app(config)
        server=uvicorn.Server(uvicorn.Config(app,host=config['SEARCH_POOL_HOST'],port=int(config['SEARCH_POOL_PORT']),log_level='warning'))
        # One process-level signal handler must stop all listeners together.
        from contextlib import nullcontext
        server.capture_signals=lambda:nullcontext()
        servers.append(server)
    loop=asyncio.get_running_loop()
    def stop():
        for server in servers:server.should_exit=True
    for sig in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(sig,stop)
    tasks=[asyncio.create_task(server.serve()) for server in servers]
    try:
        await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop()
        await asyncio.gather(*tasks)


async def administer(action,name,key=None):
    from search_pool_state import PoolState,read_keys,scheduler_config
    config=load_service(name);state=PoolState(config)
    try:
        if action=='configure':
            cfg=scheduler_config(config)
            if not await state.redis.exists(state.prefix+'config'):raise ValueError('Initialize service before configuring')
            await state.call('configure',config=cfg,digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest())
            return {'configured':name,'restart_required':True}
        await state.initialize()
        if action=='sync-keys':
            keys=read_keys(config['SERPER_KEY_POOL_FILE'],config.get('POOL_KEY_FIELD','serper'))
            await state.call('sync_keys',keys=keys);return {'synced_keys':len(keys)}
        if action=='restore-key':
            if not key or len(key)!=64 or any(c not in '0123456789abcdef' for c in key):raise ValueError('Use full SHA256 key fingerprint')
            if not await state.call('restore_key',key=key):raise ValueError('Unknown key fingerprint')
            return {'restored':key[:12]}
        return await state.health()
    finally:await state.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['serve','status','sync-keys','restore-key','configure'])
    parser.add_argument('--services',default='serper,jina',help='Comma-separated registry service names')
    parser.add_argument('--key-id')
    args=parser.parse_args();names=args.services.split(',')
    if any(name not in registry() for name in names):parser.error('Unknown service')
    if args.action=='serve':asyncio.run(serve(names))
    else:
        for name in names:print(json.dumps({'service':name,**asyncio.run(administer(args.action,name,args.key_id))}))


if __name__=='__main__':main()
