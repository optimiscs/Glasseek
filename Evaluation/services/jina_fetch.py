"""Jina Reader adapter using the same durable provider pool as /search."""
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastapi import Request
from fastapi.responses import JSONResponse
from provider_pool import create_app as create_provider_app
from provider_services import load_service
from fetch_dns import FetchDNS


class FetchProvider:
    name = 'fetch'
    path = '/fetch'

    def __init__(self, config):
        self.config = config
        self.dns = None
        os.environ['GLASSEEK_DOH_PROXY'] = config['POOL_PROXY']

    def validate(self, payload):
        if not isinstance(payload, dict) or not isinstance(payload.get('url'), str):
            return 'A URL is required'
        try:
            parsed = urlsplit(payload['url'])
            if (parsed.scheme not in ('http', 'https') or not parsed.hostname or
                    parsed.username or parsed.password or len(payload['url']) > 8192):
                return 'Invalid HTTP/HTTPS URL'
            parsed.port
        except ValueError:
            return 'Invalid HTTP/HTTPS URL'
        if payload.get('extractMode', 'text') not in ('text', 'markdown'):
            return 'extractMode must be text or markdown'

    async def check(self, payload, pool):
        if self.config['POOL_DNS_SAFETY'] != '1':
            return {'safe':True, 'error_code':'', 'error':''}
        if self.dns is None:
            self.dns = FetchDNS(pool.state)
        return await self.dns.check(payload['url'])

    def normalize(self, payload):
        # The former request model ignored unrelated fields, including client
        # credentials. Keep those out of durable jobs; this pool owns its keys.
        return {key:payload[key] for key in ('url', 'extractMode') if key in payload}

    async def prepare(self, payload, pool):
        result = await self.check(payload, pool)
        if not result['safe']:
            return {'error_code':result['error_code'], 'error':result['error']}

    async def request(self, client, payload, secret):
        async with client.get(self.config['SERPER_UPSTREAM_URL'] + payload['url'],
                headers={'Authorization':'Bearer '+secret}, proxy=self.config['POOL_PROXY'] or None,
                allow_redirects=False) as response:
            content = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                content.extend(chunk)
                if len(content) > 8 * 1024 * 1024:
                    raise ValueError('Reader response exceeds 8 MiB')
            return response.status, bytes(content), response.headers.get('Retry-After', '')

    def decode(self, content):
        return ({'text':True} if content.strip() else None), None  # Reader does not report credit usage here.

    def quota_error(self, status, content):
        return False

    def success_body(self, content):
        return json.dumps({'success':True, 'data':content.decode('utf-8', errors='replace')})


def create_app(config=None):
    config = config or load_service('jina')
    provider = FetchProvider(config)
    directory = Path(__file__).resolve().parent
    revision = hashlib.sha256(b''.join((directory / name).read_bytes() for name in
        ('jina_fetch.py','provider_pool.py','search_pool_state.py','fetch_dns.py','provider_services.py',
         'search_pool_scripts/state.lua'))).hexdigest()[:16]
    app = create_provider_app(config, provider, revision)

    @app.post('/validate')
    async def validate(request: Request):
        try:
            payload = await request.json()
            error = provider.validate(payload)
            if error:
                return JSONResponse({'safe':False,'error_code':'invalid_url','error':error},status_code=400)
            return await provider.check(payload, app.state.pool)
        except Exception as exc:
            return JSONResponse({'safe':False,'error_code':'dns_verification_failed',
                                 'error':'Central verification unavailable: '+type(exc).__name__},status_code=503)

    return app


app = create_app()
