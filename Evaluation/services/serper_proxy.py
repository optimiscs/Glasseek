"""Bounded search proxy with persistent key quarantine and explicit failures."""
import asyncio
import collections
import hashlib
import http.client
import json
import logging
import os
from pathlib import Path
import threading
import time
import ssl
import uuid
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

MAX_CONCURRENCY = max(1, int(os.environ.get('SERPER_PROXY_CONCURRENCY', '64')))
MAX_QUEUE = max(0, int(os.environ.get('SERPER_PROXY_QUEUE_LIMIT', '128')))
RPS = max(0, float(os.environ.get('SERPER_PROXY_RPS', '0')))
KEY_CONCURRENCY = max(1, int(os.environ.get('SERPER_KEY_CONCURRENCY', '1')))
KEY_RPS = max(.01, float(os.environ.get('SERPER_KEY_RPS', '.5')))
QUEUE_TIMEOUT = float(os.environ.get('SERPER_PROXY_QUEUE_TIMEOUT', '5'))
UPSTREAM_BUDGET = float(os.environ.get('SERPER_UPSTREAM_BUDGET', '45'))
UPSTREAM_TIMEOUT = float(os.environ.get('SERPER_UPSTREAM_TIMEOUT', '20'))
MAX_KEY_ATTEMPTS = 3
STATE_PATH = Path(os.environ.get('SERPER_KEY_STATE_FILE', '/tmp/glasseek-serper-key-state.json'))
USAGE_PATH = STATE_PATH.with_name('search-key-usage.jsonl')
usage_lock = threading.Lock()
lifetime_metrics = collections.defaultdict(collections.Counter)


def count_usage(row):
    stats = lifetime_metrics[row['key_id']]
    if row['event'] == 'attempt':
        stats['attempts'] += 1
    elif row['event'] == 'response':
        stats['responses'] += 1
        stats[f'http_{row["status"]}'] += 1
        credit = row.get('credits')
        if isinstance(credit, (int, float)) and not isinstance(credit, bool):
            stats['confirmed_credits'] += credit
        else:
            stats['responses_without_credit_field'] += 1
    elif row['event'] == 'transport_error':
        stats['transport_errors_unknown_billing'] += 1


if USAGE_PATH.exists():
    for line in USAGE_PATH.read_text().splitlines():
        count_usage(json.loads(line))


def record_usage(ident, event, request_id, **fields):
    row = dict(at=time.time(), key_id=ident, event=event, request_id=request_id, **fields)
    with usage_lock:
        USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(USAGE_PATH, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            payload = (json.dumps(row) + '\n').encode()
            while payload:
                payload = payload[os.write(fd, payload):]
            os.fsync(fd)
        finally:
            os.close(fd)
        count_usage(row)


def unavailable(ident, now):
    state = key_state.get(ident, {})
    return state.get('reason') in ('quota_exhausted', 'invalid_key', 'retired') or state.get('until', 0) > now
logger = logging.getLogger('serper_proxy')
app = FastAPI()
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY, thread_name_prefix='serper')
slots = asyncio.Semaphore(MAX_CONCURRENCY)
lock = threading.Lock()
rate_lock = threading.Lock()
next_request = 0.0
inflight_keys = collections.Counter()
key_next_request = {}
key_metrics = collections.defaultdict(collections.Counter)
tls_context = ssl.create_default_context()
idle_connections = collections.deque()


def borrow_connection(timeout):
    with lock:
        while idle_connections:
            conn, returned_at = idle_connections.pop()
            if time.monotonic() - returned_at < 20:
                conn.timeout = timeout
                if conn.sock:
                    conn.sock.settimeout(timeout)
                metrics['connection_reused'] += 1
                return conn
            conn.close()
        metrics['connection_created'] += 1
    return http.client.HTTPSConnection('google.serper.dev', timeout=timeout, context=tls_context)
cursor = 0
pending = 0
metrics = collections.Counter()
try:
    key_state = json.loads(STATE_PATH.read_text())
except FileNotFoundError:
    key_state = {}


def search_keys():
    path = os.environ.get('SERPER_KEY_POOL_FILE')
    if path:
        keys = json.loads(Path(path).read_text()).get('serper', [])
        return list(dict.fromkeys(k for k in keys if isinstance(k, str) and k))
    return [os.environ['SERPER_KEY_ID']] if os.environ.get('SERPER_KEY_ID') else []


def key_id(key):
    return hashlib.sha256(key.encode()).hexdigest()


def save_state():
    # Caller holds lock. Persist fingerprints only, never credentials.
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix('.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(key_state, f)
    os.replace(tmp, STATE_PATH)


def acquire_key(keys, excluded, deadline=None):
    global cursor
    while True:
        with lock:
            now = time.time()
            eligible = False
            for offset in range(len(keys)):
                i = (cursor + offset) % len(keys)
                ident = key_id(keys[i])
                if ident in excluded or unavailable(ident, now):
                    continue
                eligible = True
                if inflight_keys[ident] >= KEY_CONCURRENCY or key_next_request.get(ident, 0) > time.monotonic():
                    continue
                inflight_keys[ident] += 1
                key_next_request[ident] = time.monotonic() + 1 / KEY_RPS
                km = key_metrics[ident]
                km['requests'] += 1
                km['peak_inflight'] = max(km['peak_inflight'], inflight_keys[ident])
                cursor = (i + 1) % len(keys)
                return keys[i], ident
        if not eligible or deadline is None or time.monotonic() >= deadline:
            return None
        time.sleep(min(.05, max(0, deadline-time.monotonic())))


def quarantine(ident, reason, seconds):
    with lock:
        key_state[ident] = {'reason': reason, 'until': time.time() + seconds,
                            'permanent': reason in ('quota_exhausted', 'invalid_key', 'retired')}
        metrics[reason] += 1
        save_state()
    logger.warning('search credential quarantined reason=%s cooldown_s=%s', reason, seconds)


def failure(code, message, retryable=False, retry_after=0):
    body = {'success': False, 'error_code': code, 'error': message,
            'retryable': retryable, 'retry_after': retry_after}
    return 503, json.dumps(body).encode()


def upstream_search(payload):
    global next_request
    keys = search_keys()
    deadline = time.monotonic() + UPSTREAM_BUDGET
    excluded = set()
    for attempt in range(MAX_KEY_ATTEMPTS):
        selected = acquire_key(keys, excluded, deadline)
        if not selected:
            return failure('search_credentials_unavailable',
                'No eligible search credential: keys are exhausted, invalid, cooling down or busy.', True, 5)
        key, ident = selected
        request_id = uuid.uuid4().hex
        excluded.add(ident)
        try:
            with rate_lock:
                delay = max(0, next_request - time.monotonic())
                if delay + 1 >= deadline - time.monotonic():
                    return failure('search_rate_wait_timeout', 'Search rate limiter reached request deadline.', True, 2)
                time.sleep(delay)
                next_request = time.monotonic() + (1 / RPS if RPS else 0)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return failure('search_upstream_timeout', 'Search upstream deadline exceeded.', True, 2)
            conn = borrow_connection(min(UPSTREAM_TIMEOUT, remaining))
            reusable = False
            try:
                record_usage(ident, 'attempt', request_id, requested_num=payload.get('num', 10))
                conn.request('POST', '/search', body=json.dumps(payload), headers={
                    'Content-Type': 'application/json', 'X-API-KEY': key})
                response = conn.getresponse()
                status, data = response.status, response.read()
                retry_header = response.getheader('Retry-After')
                reusable = (response.getheader('Connection') or '').lower() != 'close' and bool(conn.sock)
                try:
                    credits = json.loads(data).get('credits')
                except (ValueError, AttributeError):
                    credits = None
                record_usage(ident, 'response', request_id, status=status, credits=credits)
            finally:
                if reusable:
                    with lock:idle_connections.append((conn, time.monotonic()))
                else:
                    conn.close()
            with lock:
                metrics[f'upstream_http_{status}'] += 1
                key_metrics[ident][f'http_{status}'] += 1
            if status in (401, 402) or (status == 400 and b'not enough credits' in data.lower()):
                quarantine(ident, 'invalid_key' if status == 401 else 'quota_exhausted', 86400)
                continue
            if status == 429:
                cooldown = max(60, min(3600, float(retry_header))) if retry_header and retry_header.isdigit() else 60
                quarantine(ident, 'rate_limited', cooldown)
                return failure('search_rate_limited', 'Search provider returned HTTP 429; credential cooling down.', True, cooldown)
            if status >= 500:
                return failure('search_upstream_error', f'Search provider HTTP {status}.', True, 5)
            if status >= 400:
                return 400, json.dumps({'success': False, 'error_code': 'search_invalid_request',
                    'error': f'Search provider rejected query: {data.decode(errors="replace")[:500]}', 'retryable': False}).encode()
            return status, data
        except (OSError, http.client.HTTPException) as exc:
            record_usage(ident, 'transport_error', request_id, error_type=type(exc).__name__)
            with lock:
                metrics['transport_error'] += 1
                metrics[f'transport_{type(exc).__name__}'] += 1
                key_metrics[ident]['transport_error'] += 1
            if attempt + 1 < MAX_KEY_ATTEMPTS and deadline - time.monotonic() > 2:
                continue
            return failure('search_transport_error', f'Search transport failed: {type(exc).__name__}.', True, 2)
        finally:
            with lock:
                inflight_keys[ident] -= 1
    return failure('search_credential_attempt_limit', 'Unusable search credentials quarantined; bounded attempt limit reached.', True, 2)


@app.get('/health')
def health():
    keys = search_keys()
    with usage_lock:
        totals = {ident[:12]: dict(v) for ident,v in lifetime_metrics.items()}
    with lock:
        reasons = collections.Counter(key_state.get(key_id(k), {}).get('reason') for k in keys
            if unavailable(key_id(k), time.time()))
        return {'status': 'ok', 'max_concurrency': MAX_CONCURRENCY, 'queue_limit': MAX_QUEUE,
                'requests_per_second': RPS, 'pending': pending, 'total_keys': len(keys),
                'per_key_concurrency': KEY_CONCURRENCY, 'per_key_rps': KEY_RPS,
                'lifetime_usage': totals,
                'usage_log': str(USAGE_PATH),
                'quarantined_keys': dict(reasons), 'eligible_keys': len(keys)-sum(reasons.values()),
                'inflight_keys': sum(v > 0 for v in inflight_keys.values()),
                'inflight_requests': sum(inflight_keys.values()), 'metrics': dict(metrics),
                'per_key_metrics': {key_id(k)[:12]: dict(key_metrics[key_id(k)]) for k in keys if key_metrics[key_id(k)]}}


@app.post('/search')
async def search(request: Request):
    global pending
    payload = await request.json()
    if not isinstance(payload, dict) or not str(payload.get('q') or '').strip():
        return JSONResponse({'success': False, 'error': 'A nonempty q is required.', 'retryable': False}, status_code=400)
    if pending >= MAX_CONCURRENCY + MAX_QUEUE:
        status, data = failure('search_queue_full', 'Search queue is full; retry with lower concurrency.', True, 2)
        return Response(data, status_code=status, media_type='application/json', headers={'Retry-After': '2'})
    pending += 1
    acquired = False
    submitted = False
    try:
        await asyncio.wait_for(slots.acquire(), timeout=QUEUE_TIMEOUT)
        acquired = True
        if await request.is_disconnected():
            return Response(status_code=499)
        future = asyncio.get_running_loop().run_in_executor(executor, upstream_search, payload)
        def completed(_):
            global pending
            slots.release()
            pending -= 1
        future.add_done_callback(completed)
        submitted = True
        status, data = await asyncio.shield(future)
        return Response(data, status_code=status, media_type='application/json')
    except asyncio.TimeoutError:
        status, data = failure('search_queue_timeout', 'Timed out waiting for search capacity.', True, 2)
        return Response(data, status_code=status, media_type='application/json')
    finally:
        if not submitted:
            if acquired:slots.release()
            pending -= 1
