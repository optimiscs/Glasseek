import asyncio
import http.client
import json
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, Response

MAX_CONCURRENCY = max(1, int(os.environ.get('SERPER_PROXY_CONCURRENCY', '16')))
logger = logging.getLogger('serper_proxy')
app = FastAPI()
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY, thread_name_prefix='serper')
slots = asyncio.Semaphore(MAX_CONCURRENCY)
key_lock = threading.Lock()
key_index = int(os.environ.get('SERPER_KEY_START_INDEX', '0'))

def search_keys():
    path = os.environ.get('SERPER_KEY_POOL_FILE')
    if path:
        with open(path, encoding='utf-8') as source:
            keys = json.load(source).get('serper', [])
        if keys:
            return list(dict.fromkeys(k for k in keys if isinstance(k, str) and k))
    return [os.environ['SERPER_KEY_ID']]

@app.get('/health')
def health():
    return {'status': 'ok', 'max_concurrency': MAX_CONCURRENCY}

def upstream_search(payload):
    global key_index
    started = time.monotonic()
    keys = search_keys()
    if not keys:
        raise RuntimeError('Serper key pool is empty')
    with key_lock:
        start_index = key_index % len(keys)
    for offset in range(len(keys)):
        index = (start_index + offset) % len(keys)
        conn = http.client.HTTPSConnection('google.serper.dev', timeout=30)
        try:
            conn.request('POST', '/search', body=json.dumps(payload), headers={
                'Content-Type': 'application/json', 'X-API-KEY': keys[index]})
            response = conn.getresponse()
            status, data = response.status, response.read()
        finally:
            conn.close()
        exhausted = status == 400 and b'not enough credits' in data.lower()
        if exhausted or status in (401, 402, 429):
            with key_lock:
                if key_index % len(keys) == index:
                    key_index = (index + 1) % len(keys)
            logger.warning('serper key index=%d unavailable status=%d; rotating', index + 1, status)
            continue
        logger.info('serper upstream status=%s duration_s=%.3f key_index=%d', status, time.monotonic() - started, index + 1)
        return status, data
    logger.error('all Serper keys unavailable')
    return status, data

@app.post('/search')
async def search(request: Request):
    payload = await request.json()
    queued = time.monotonic()
    await slots.acquire()
    try:
        future = asyncio.get_running_loop().run_in_executor(executor, upstream_search, payload)
    except BaseException:
        slots.release()
        raise
    # Keep the slot occupied if the request coroutine is cancelled: the blocking
    # upstream call still runs until completion in its worker thread.
    future.add_done_callback(lambda _: slots.release())
    status, data = await asyncio.shield(future)
    logger.info('serper request queue_and_upstream_s=%.3f', time.monotonic() - queued)
    return Response(content=data, status_code=status, media_type='application/json')
