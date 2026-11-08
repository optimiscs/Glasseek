"""Serper protocol adapter for the shared durable provider pool."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from provider_pool import ProviderPool, create_app as create_provider_app
from search_pool import load_config, implementation_revision


class SearchProvider:
    name = 'search'
    path = '/search'

    def validate(self, payload):
        if not isinstance(payload, dict) or not isinstance(payload.get('q'), str) or not payload['q'].strip():
            return 'A nonempty q is required.'

    async def prepare(self, payload, pool):
        return None

    async def request(self, client, payload, secret):
        async with client.post(self.config['SERPER_UPSTREAM_URL'], json=payload,
                               headers={'X-API-KEY':secret}, allow_redirects=False) as response:
            return response.status, await response.read(), response.headers.get('Retry-After', '')

    def decode(self, content):
        try:
            data = json.loads(content)
            credits = data.get('credits') if isinstance(data, dict) else None
            return data, None if isinstance(credits, bool) else credits
        except ValueError:
            return None, None

    def quota_error(self, status, content):
        return status == 400 and b'not enough credits' in content.lower()

    def success_body(self, content):
        return content.decode('utf-8', errors='replace')

    def __init__(self, config):
        self.config = config


class SearchPool(ProviderPool):
    def __init__(self, config):
        super().__init__(config, SearchProvider(config))


def create_app(config=None):
    config = config or load_config()
    return create_provider_app(config, SearchProvider(config), implementation_revision())


app = create_app()
