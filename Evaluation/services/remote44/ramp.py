"""Run a closed-loop, bounded dual-vLLM embedding concurrency ramp."""
import asyncio
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
import httpx

ROOT = Path('/home/moxu/glasseek-embedding-eval')
PORTS = (18318, 18319)
MODEL = 'QWEN3-EMBEDDING-8B'
INTERVAL = float(os.environ.get('RAMP_WINDOW', '90'))
STEP = 2
MAXIMUM = 64
METRIC = re.compile(r'^vllm:num_preemptions_total(?:\{[^}]*\})?\s+(\S+)', re.M)
raw = (ROOT / 'passages.txt').read_text()
blocks = [raw[i:i+1400] for i in range(0, len(raw), 1400) if len(raw[i:i+1400]) > 100]
passages = [blocks[i % len(blocks)] for i in range(64)]
records = ROOT / 'logs/ramp.jsonl'
state = {'stop': False, 'limit': 2, 'ok': 0, 'errors': 0, 'latencies': []}

def log(event, **fields):
    row = {'epoch': time.time(), 'event': event, **fields}
    with records.open('a', encoding='utf8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)

async def counts(client):
    out = {}
    for port in PORTS:
        r = await client.get(f'http://127.0.0.1:{port}/metrics', timeout=5)
        r.raise_for_status()
        values = METRIC.findall(r.text)
        if not values:
            raise RuntimeError(f'preemption metric missing on port {port}')
        values = [float(value) for value in values]
        if not all(math.isfinite(x) for x in values):
            raise RuntimeError('invalid preemption metric')
        out[str(port)] = sum(values)
    return out

async def worker(index, client):
    endpoint = f'http://127.0.0.1:{PORTS[index % len(PORTS)]}/v1/embeddings'
    payload = {'model': MODEL, 'input': ['Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: Which language did Grace Hopper help develop?'] + passages,
               'truncate_prompt_tokens': 2048}
    while not state['stop']:
        if index >= state['limit']:
            await asyncio.sleep(.5)
            continue
        started = time.monotonic()
        try:
            response = await client.post(endpoint, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()['data']
            if len(data) != len(payload['input']):
                raise ValueError('unexpected embedding count')
            state['ok'] += 1
            state['latencies'].append(time.monotonic()-started)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            state['errors'] += 1
            log('request_error', worker=index, error=type(exc).__name__)
            await asyncio.sleep(2)

async def main():
    limits = httpx.Limits(max_connections=MAXIMUM+4, max_keepalive_connections=MAXIMUM+4)
    async with httpx.AsyncClient(limits=limits, trust_env=False) as client:
        baseline = await counts(client)
        log('baseline', concurrency=2, preemptions=baseline, model=MODEL,
            window_s=INTERVAL, step=STEP, max_concurrency=MAXIMUM,
            requests='1 query + 64 page passages; closed-loop; round-robin across two replicas',
            passage_chars=1400, prefix_caching=False)
        workers = [asyncio.create_task(worker(i, client)) for i in range(MAXIMUM)]
        start = time.monotonic(); old_ok = old_err = 0; lat_start = 0
        try:
            while True:
                await asyncio.sleep(5)
                now = await counts(client)
                if any(now[k] > baseline[k] for k in now):
                    before = state['limit']
                    state['limit'] = max(2, before-STEP)
                    log('preemption', concurrency=before, rollback=state['limit'], preemptions=now)
                    break
                if time.monotonic()-start < INTERVAL:
                    continue
                elapsed = time.monotonic()-start
                latencies = state['latencies'][lat_start:]
                log('window', concurrency=state['limit'], seconds=round(elapsed,2),
                    completed=state['ok']-old_ok, errors=state['errors']-old_err,
                    mean_latency_s=round(statistics.mean(latencies),3) if latencies else None,
                    max_latency_s=round(max(latencies),3) if latencies else None,
                    preemptions=now)
                if state['errors'] > old_err:
                    log('stopped_request_errors', concurrency=state['limit'])
                    break
                if state['ok'] == old_ok:
                    log('stopped_no_completed_requests', concurrency=state['limit'])
                    break
                if state['limit'] >= MAXIMUM:
                    log('max_reached_without_preemption', concurrency=MAXIMUM, preemptions=now)
                    break
                state['limit'] += STEP
                start = time.monotonic(); old_ok = state['ok']; old_err = state['errors']; lat_start = len(state['latencies'])
                baseline = now
                log('ramp_up', concurrency=state['limit'])
        except Exception as exc:
            log('monitor_failure', error=type(exc).__name__, message=str(exc)[:200])
        finally:
            state['stop'] = True
            for task in workers: task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            log('finished', completed=state['ok'], errors=state['errors'])

if __name__ == '__main__':
    asyncio.run(main())
