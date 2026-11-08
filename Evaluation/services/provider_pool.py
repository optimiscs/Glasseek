"""Shared durable provider queue: admission, leases, HTTP lifecycle and idempotency."""
import asyncio
from contextlib import asynccontextmanager, suppress
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
import time
import uuid

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from redis.exceptions import RedisError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_pool import config_digest
from search_pool_state import PoolState, key_id

logger = logging.getLogger('search_pool')
MAX_KEY_ATTEMPTS = 3


def failure(code, message, retryable=False, **extra):
    return json.dumps(dict(success=False, error_code=code, error=message, retryable=retryable, **extra))


class ProviderPool:
    def __init__(self, config, provider):
        self.provider = provider
        self.config = config
        self.state = PoolState(config)
        self.instance = uuid.uuid4().hex
        self.lease = float(config['SEARCH_POOL_LEASE_SECONDS'])
        self.workers = int(config['SEARCH_POOL_WORKERS'])
        self.tasks = set()
        self.client = None
        self.dispatcher = None

    def failure(self, code, message, retryable=False, **extra):
        return failure(code.replace('search_', self.provider.name + '_'),
                       message.replace('Search', self.provider.name.title()).replace('search', self.provider.name),
                       retryable, **extra)

    async def start(self):
        await self.state.initialize()
        self.client = aiohttp.ClientSession(trust_env=False,
            timeout=aiohttp.ClientTimeout(total=float(self.config['SERPER_UPSTREAM_TIMEOUT'])),
            connector=aiohttp.TCPConnector(limit=self.workers,keepalive_timeout=20),
            cookie_jar=aiohttp.DummyCookieJar())
        self.dispatcher = asyncio.create_task(self.dispatch())

    async def close(self):
        if self.dispatcher:
            self.dispatcher.cancel()
            with suppress(asyncio.CancelledError): await self.dispatcher
        # Cancel HTTP I/O; durable leases recover unsent work and mark sent work uncertain.
        for task in list(self.tasks): task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.client: await self.client.close()
        await self.state.close()

    async def dispatch(self):
        last_recovery = 0
        while True:
            try:
                if time.monotonic()-last_recovery > min(1, self.lease/3):
                    await self.state.call('recover')
                    last_recovery = time.monotonic()
                if len(self.tasks) < self.workers:
                    owner = self.instance + ':' + uuid.uuid4().hex
                    result = await self.state.call('claim', owner=owner, lease=self.lease)
                    if result:
                        task = asyncio.create_task(self.execute(result[0], owner, json.loads(result[1]),float(result[2])))
                        self.tasks.add(task)
                        task.add_done_callback(self.tasks.discard)
                        continue
            except RedisError:
                # Fail closed: never issue upstream calls without central admission.
                logger.warning('Search state unavailable; dispatch paused')
                await asyncio.sleep(.5)
            await asyncio.sleep(.01)

    async def heartbeat(self, ident, owner, execution):
        try:
            while True:
                await asyncio.sleep(self.lease/3)
                if not await self.state.call('renew', id=ident, owner=owner, lease=self.lease):
                    execution.cancel(); return
        except RedisError:
            execution.cancel()

    async def execute(self, ident, owner, payload, queue_age=0):
        pulse = asyncio.create_task(self.heartbeat(ident, owner, asyncio.current_task()))
        sent = False
        try:
            # Queueing and execution share the original caller's deadline. A
            # recovered job does not receive a fresh full timeout budget.
            budget=min(float(self.config['SERPER_UPSTREAM_BUDGET']),
                       float(self.config['SEARCH_SERVER_TIMEOUT'])-2-queue_age)
            deadline = time.monotonic()+budget
            async with asyncio.timeout(max(.001, deadline-time.monotonic())):
                error = await self.provider.prepare(payload, self)
            if error:
                await self.state.call('finish', id=ident, owner=owner, status=400, unknown=False,
                                      body=json.dumps(dict(success=False, retryable=False, **error)))
                return
            excluded = {}
            for attempt in range(MAX_KEY_ATTEMPTS):
                while True:
                    if time.monotonic() >= deadline:
                        await self.state.call('finish', id=ident, owner=owner, status=503, unknown=False,
                            body=self.failure('search_rate_wait_timeout','Timed out waiting for search admission.',True))
                        return
                    chosen = await self.state.call('admit', id=ident, owner=owner, lease=self.lease,
                                                   excluded=excluded, attempt=attempt+1)
                    if chosen[0] != 'wait': break
                    await asyncio.sleep(random.uniform(.04,.08))
                if chosen[0] == 'lost': return
                if chosen[0] != 'ok':
                    code = 'search_budget_exhausted' if chosen[0]=='budget' else 'search_credentials_unavailable'
                    await self.state.call('finish', id=ident, owner=owner, status=503, unknown=False,
                        body=self.failure(code,'No available search credential or upstream attempt budget.',chosen[0]!='budget'))
                    return
                _, key, secret = chosen
                excluded[key] = True
                sent = True
                # asyncio timeout bounds the entire request, not just each socket operation.
                remaining = min(float(self.config['SERPER_UPSTREAM_TIMEOUT']), deadline-time.monotonic())
                try:
                    async with asyncio.timeout(max(.001, remaining)):
                        status, content, retry = await self.provider.request(self.client, payload, secret)
                except (aiohttp.ClientError, TimeoutError) as exc:
                    logger.warning('Upstream transport outcome unknown: %s',type(exc).__name__)
                    await self.state.call('finish', id=ident, owner=owner, status=503, unknown=True,
                        body=self.failure('search_outcome_unknown','Search transport failed after admission; billing may have occurred. Not replayed.',
                                     transport_error=type(exc).__name__))
                    return
                data, credits = self.provider.decode(content)
                reason, cooldown, rotate = '', 0, False
                result_status, body = status, self.provider.success_body(content)
                if status in (401,402) or self.provider.quota_error(status, content):
                    reason = 'invalid_key' if status==401 else 'quota_exhausted'
                    rotate = attempt+1 < MAX_KEY_ATTEMPTS
                    result_status, body = 503, self.failure('search_credential_attempt_limit',
                        f'Upstream HTTP {status}: {reason}; keys quarantined; bounded attempt limit reached.',
                        True, upstream_status=status, credential_reason=reason, attempts=attempt+1)
                elif status==429:
                    cooldown = max(60,min(3600,float(retry))) if retry.isdigit() else 60
                    reason='rate_limited'
                    result_status, body = 503, self.failure('search_rate_limited','Search credential cooling down.',True,retry_after=cooldown)
                elif status>=500:
                    result_status, body = 503, self.failure('search_upstream_error',f'Search provider HTTP {status}.',True)
                elif status>=400:
                    result_status, body = 400, self.failure('search_invalid_request',f'Search provider rejected request (HTTP {status}).', upstream_status=status)
                elif status!=200 or not isinstance(data,dict):
                    result_status, body = 502, self.failure('search_invalid_response','Search provider returned an invalid response.')
                await self.state.call('response', id=ident, owner=owner, status=status, credits=credits,
                    reason=reason,cooldown=cooldown,rotate=rotate,result_status=result_status,body=body)
                sent=False
                if not rotate: return
        except RedisError:
            # Recovery must decide whether dispatch happened; do not blindly retry billing operations.
            logger.warning('Search state lost during execution; awaiting lease recovery')
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception('Search worker failed')
            with suppress(RedisError):
                await self.state.call('finish', id=ident,owner=owner,status=503,unknown=sent,
                    body=self.failure('search_outcome_unknown' if sent else 'search_internal_error','Search worker failed.'))
        finally:
            pulse.cancel()
            with suppress(asyncio.CancelledError): await pulse


def create_app(config, provider, revision):
    pool = ProviderPool(config, provider)
    failure = pool.failure

    @asynccontextmanager
    async def lifespan(app):
        try:
            await pool.start()
            yield
        finally:
            await pool.close()

    app = FastAPI(lifespan=lifespan)
    app.state.pool = pool

    @app.get('/health')
    async def health():
        try:
            values = await pool.state.health()
            return dict(status='ok',service='glasseek-'+provider.name+'-pool',revision=revision,
                        instance_id=pool.instance,config_digest=config_digest(config),
                        max_concurrency=int(config['SERPER_PROXY_CONCURRENCY']),
                        queue_limit=int(config['SERPER_PROXY_QUEUE_LIMIT']),
                        per_key_concurrency=int(config['SERPER_KEY_CONCURRENCY']),
                        per_key_rps=float(config['SERPER_KEY_RPS']),
                        requests_per_second=float(config['SERPER_PROXY_RPS']),**values)
        except RedisError:
            return JSONResponse({'status':'unavailable','error_code':provider.name+'_state_unavailable'},status_code=503)

    @app.post(provider.path)
    async def search(request: Request):
        try:
            payload = await request.json()
        except (ValueError, UnicodeError):
            return JSONResponse({'success':False,'error':'Invalid JSON','retryable':False},status_code=400)
        error = provider.validate(payload)
        if error:
            return JSONResponse({'success':False,'error':error,'retryable':False},status_code=400)
        if hasattr(provider, 'normalize'):
            payload = provider.normalize(payload)
        encoded = json.dumps(payload,sort_keys=True,separators=(',',':'))
        if len(encoded.encode())>65536:
            return JSONResponse({'success':False,'error':'Search payload exceeds 64 KiB','retryable':False},status_code=413)
        token = request.headers.get('Idempotency-Key')
        if token and len(token)>256:
            return JSONResponse({'success':False,'error':'Idempotency-Key too long'},status_code=400)
        ident = hashlib.sha256(token.encode()).hexdigest() if token else uuid.uuid4().hex
        headers={'X-'+provider.name.title()+'-Request-ID':ident}
        try:
            admitted = await pool.state.call('enqueue',id=ident,payload=encoded,
                                             digest=hashlib.sha256(encoded.encode()).hexdigest())
            if admitted=='conflict':
                return Response(failure('search_idempotency_conflict','Idempotency key reused with different payload.'),status_code=409,media_type='application/json')
            if admitted=='full':
                return Response(failure('search_queue_full','Search queue is full.',True,retry_after=2),status_code=503,
                                media_type='application/json',headers={'Retry-After':'2'})
            deadline=time.monotonic()+float(config['SEARCH_SERVER_TIMEOUT'])-1
            while time.monotonic()<deadline:
                result=await pool.state.job(ident)
                if result.get('phase')=='done':
                    headers['X-'+provider.name.title()+'-Queue-Ms']=result.get('queue_ms','0')
                    return Response(result['body'],status_code=int(result['status']),media_type='application/json',headers=headers)
                if await request.is_disconnected(): return Response(status_code=499,headers=headers)
                await asyncio.sleep(.1)
            return Response(failure('search_result_pending','Request still tracked; reuse the same Idempotency-Key to retrieve its result.'),
                            status_code=503,media_type='application/json',headers=headers)
        except RedisError:
            return Response(failure('search_state_unavailable','Central search state unavailable; no uncontrolled fallback.'),
                            status_code=503,media_type='application/json',headers=headers)

    return app
