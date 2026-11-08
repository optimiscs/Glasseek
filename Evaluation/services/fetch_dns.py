"""Shared DNS validation cache; failed checks never become permission to fetch."""
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import uuid
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Evaluation_backend'))
from tools.url_safety import check_external_fetch_url


class FetchDNS:
    def __init__(self, state):
        self.state = state
        self.slots = asyncio.Semaphore(8)
        self.pending = 0

    async def check(self, url):
        host = urlsplit(url).hostname or ''
        cache = self.state.prefix + 'dns:' + hashlib.sha256(host.lower().encode()).hexdigest()
        redis = self.state.redis
        owner = uuid.uuid4().hex
        for _ in range(550):
            result = await redis.get(cache)
            if result:
                await redis.hincrby(self.state.prefix+'metrics', 'dns_cache_hits', 1)
                return json.loads(result)
            if await redis.set(cache+':lock', owner, nx=True, ex=50):
                try:
                    if self.pending >= 128:
                        return {'safe':False,'error_code':'dns_queue_full','error':'DNS verification capacity exhausted'}
                    self.pending += 1
                    try:
                        await asyncio.wait_for(self.slots.acquire(), 10)
                    except TimeoutError:
                        return {'safe':False,'error_code':'dns_queue_timeout','error':'DNS verification queue timed out'}
                    finally:
                        self.pending -= 1
                    task = asyncio.create_task(asyncio.to_thread(check_external_fetch_url, url))
                    task.add_done_callback(lambda _: self.slots.release())
                    try:
                        safe, code, error = await asyncio.wait_for(asyncio.shield(task), 35)
                    except TimeoutError:
                        safe, code, error = False, 'dns_verification_failed', 'DNS verification timed out'
                    result = dict(safe=safe, error_code=code, error=error)
                    await redis.set(cache, json.dumps(result), ex=5 if safe else 30 if code == 'dns_not_found' else 3)
                    await redis.hincrby(self.state.prefix+'metrics', 'dns_checks', 1)
                    return result
                finally:
                    await redis.eval("if redis.call('GET',KEYS[1])==ARGV[1] then return redis.call('DEL',KEYS[1]) end return 0", 1, cache+':lock', owner)
            await asyncio.sleep(.1)
        return {'safe':False,'error_code':'dns_verification_failed','error':'DNS verification queue timeout'}
