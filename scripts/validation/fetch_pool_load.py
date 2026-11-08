"""Exercise /fetch using the existing isolated search-pool load infrastructure."""
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import time
import aiohttp
from search_pool_load import Sandbox, ROOT, SERVICES, free_port, percentile


class FetchSandbox(Sandbox):
    def add_replica(self):
        port = free_port()
        config = json.loads((ROOT/'Evaluation/config/fetch_pool.json').read_text())
        mapping = {'concurrency':'SERPER_PROXY_CONCURRENCY','queue_limit':'SERPER_PROXY_QUEUE_LIMIT',
                   'workers':'SEARCH_POOL_WORKERS','per_key_concurrency':'SERPER_KEY_CONCURRENCY',
                   'per_key_rps':'SERPER_KEY_RPS','global_rps':'SERPER_PROXY_RPS',
                   'queue_timeout':'SERPER_PROXY_QUEUE_TIMEOUT','upstream_timeout':'SERPER_UPSTREAM_TIMEOUT',
                   'upstream_budget':'SERPER_UPSTREAM_BUDGET','client_timeout':'SEARCH_SERVER_TIMEOUT',
                   'lease_seconds':'SEARCH_POOL_LEASE_SECONDS','result_ttl':'SEARCH_POOL_RESULT_TTL',
                   'max_upstream_attempts':'SEARCH_POOL_MAX_UPSTREAM_ATTEMPTS'}
        for name, key in mapping.items():
            if key in self.env: config[name] = self.env[key]
        keys = json.loads((self.directory/'keys.json').read_text())
        (self.directory/'fetch-keys.json').write_text(json.dumps({'jina':keys['serper']}))
        (self.directory/'fetch-keys.json').chmod(0o600)
        config.update(redis_url=self.redis_url,namespace=self.env['SEARCH_POOL_NAMESPACE'],
                      upstream_url=self.upstream_url+'/fetch/',keys_file=str(self.directory/'fetch-keys.json'),
                      proxy_url='',dns_safety=False)
        path=self.directory/'fetch-config.json';path.write_text(json.dumps(config))
        env=dict(self.env, GLASSEEK_FETCH_POOL_CONFIG=str(path), GLASSEEK_FETCH_URL=f'http://127.0.0.1:{port}',
                 TOOL_SERVICES_PYTHON=sys.executable)
        proc=self.process([sys.executable,str(SERVICES/'tool_services.py'),'serve','fetch'],env,'fetch-'+str(port))
        url=f'http://127.0.0.1:{port}';self.ready(proc,url)
        self.replicas.append(proc);self.urls.append(url)
        return proc,url


async def load(sandbox, concurrency, page_bytes=65536):
    waits=[];latencies=[];statuses=Counter()
    started=time.monotonic()
    async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=90),
            connector=aiohttp.TCPConnector(limit=concurrency)) as client:
        async def request(i):
            t=time.monotonic()
            async with client.post(sandbox.urls[i%len(sandbox.urls)]+'/fetch',
                    json={'url':f'https://example.com/c{concurrency}-{i}?delay=.1&bytes={page_bytes}'},
                    headers={'Idempotency-Key':f'c{concurrency}-{i}'}) as response:
                await response.read();statuses[response.status]+=1
                latencies.append((time.monotonic()-t)*1000)
                if 'X-Fetch-Queue-Ms' in response.headers:waits.append(float(response.headers['X-Fetch-Queue-Ms']))
        await asyncio.gather(*(request(i) for i in range(concurrency)))
    elapsed=time.monotonic()-started
    rows=[r for r in sandbox.stats()['rows'] if f'/c{concurrency}-' in r['q']]
    bills=Counter(r['q'] for r in rows if r['charged'])
    return dict(concurrency=concurrency,status_counts=dict(statuses),elapsed_seconds=elapsed,
                successful_calls_per_second=statuses[200]/elapsed,tool_success_rate=statuses[200]/concurrency,
                queue_p95_ms=percentile(waits,.95),queue_p99_ms=percentile(waits,.99),
                latency_p95_ms=percentile(latencies,.95),latency_p99_ms=percentile(latencies,.99),
                upstream_attempts=len(rows),duplicate_billed_calls=sum(max(0,n-1) for n in bills.values()),
                page_bytes=page_bytes, workload='mock Reader HTTP, 100ms upstream; DNS and LLM excluded')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='fetch-load-') as directory:
        with FetchSandbox(directory,keys=[f'fake-{i}' for i in range(128)],overrides={
            'SEARCH_POOL_WORKERS':128,'SERPER_PROXY_CONCURRENCY':128,'SERPER_PROXY_QUEUE_LIMIT':1024,
            'SERPER_KEY_CONCURRENCY':2,'SERPER_KEY_RPS':1000,'SERPER_UPSTREAM_TIMEOUT':10,
            'SERPER_UPSTREAM_BUDGET':60,'SEARCH_POOL_LEASE_SECONDS':15,
            'SERPER_PROXY_QUEUE_TIMEOUT':60,'SEARCH_SERVER_TIMEOUT':85}) as sandbox:
            results=[]
            for level in [64,128,256,512,1024]:
                result=asyncio.run(load(sandbox,level));results.append(result);print(json.dumps(result),flush=True)
            args.output.write_text(json.dumps({'levels':results,'health':sandbox.health()},indent=2)+'\n')


if __name__=='__main__':main()
