"""Redis-backed queue, fencing leases, credential admission and durable audit events."""
import collections
import hashlib
import json
from pathlib import Path

import redis.asyncio as redis

SCRIPT = Path(__file__).with_name('search_pool_scripts') / 'state.lua'


def key_id(key):
    return hashlib.sha256(key.encode()).hexdigest()


def read_keys(path, field='serper'):
    keys = json.loads(Path(path).read_text()).get(field)
    if not isinstance(keys, list) or not keys or any(not isinstance(k, str) or not k.strip() for k in keys):
        raise ValueError(f'Configure a nonempty {field} key list')
    return {key_id(k): k for k in keys}


def scheduler_config(c):
    fields = {'concurrency': 'SERPER_PROXY_CONCURRENCY', 'queue': 'SERPER_PROXY_QUEUE_LIMIT',
              'key_concurrency': 'SERPER_KEY_CONCURRENCY', 'key_rps': 'SERPER_KEY_RPS',
              'rps': 'SERPER_PROXY_RPS', 'queue_timeout': 'SERPER_PROXY_QUEUE_TIMEOUT',
              'ttl': 'SEARCH_POOL_RESULT_TTL', 'attempt_limit': 'SEARCH_POOL_MAX_UPSTREAM_ATTEMPTS',
              'upstream': 'SERPER_UPSTREAM_URL', 'timeout': 'SERPER_UPSTREAM_TIMEOUT',
              'budget': 'SERPER_UPSTREAM_BUDGET', 'lease': 'SEARCH_POOL_LEASE_SECONDS',
              'client_timeout': 'SEARCH_SERVER_TIMEOUT',
              'durability': 'SEARCH_POOL_REQUIRE_DURABILITY'}
    result = dict(schema='2', **{k: c[v] for k, v in fields.items()})
    if c.get('POOL_PROVIDER'):
        result.update(provider=c['POOL_PROVIDER'], proxy=c.get('POOL_PROXY', ''),
                      safety=c.get('POOL_DNS_SAFETY', '1'))
    return result


class PoolState:
    def __init__(self, config):
        self.config = config
        self.prefix = '{' + config['SEARCH_POOL_NAMESPACE'] + '}:'
        self.redis = redis.Redis.from_url(config['SEARCH_POOL_REDIS_URL'], decode_responses=True,
                                         socket_connect_timeout=2, socket_timeout=3,
                                         max_connections=2048, retry_on_timeout=False)
        self.script = self.redis.register_script(SCRIPT.read_text())

    async def call(self, op, **args):
        return await self.script(keys=[self.prefix], args=[op, json.dumps(args)])

    async def initialize(self):
        await self.redis.ping()
        if self.config['SEARCH_POOL_REQUIRE_DURABILITY'] == '1':
            for name, required in (('appendonly', 'yes'), ('appendfsync', 'always'), ('maxmemory-policy', 'noeviction')):
                actual = await self.redis.config_get(name)
                if actual.get(name) != required:
                    raise ValueError(f'Redis must configure {name}={required} for durable search admission')
        cfg = scheduler_config(self.config)
        digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        # Only the first initializer reads/imports legacy files. Later replicas consume central keys.
        keys, state, usage = {}, {}, {}
        if not await self.redis.exists(self.prefix + 'config'):
            keys = read_keys(self.config['SERPER_KEY_POOL_FILE'], self.config.get('POOL_KEY_FIELD', 'serper'))
            path = Path(self.config['SERPER_KEY_STATE_FILE'])
            if path.exists():
                state = json.loads(path.read_text())
            history = path.with_name('search-key-usage.jsonl')
            counts = collections.defaultdict(collections.Counter)
            if history.exists():
                for line in history.read_text().splitlines():
                    row = json.loads(line)
                    stats = counts[row['key_id']]
                    if row['event'] == 'attempt':
                        stats['attempts'] += 1
                    elif row['event'] == 'response':
                        stats['responses'] += 1
                        stats[f'http_{row["status"]}'] += 1
                        if isinstance(row.get('credits'), (int, float)) and not isinstance(row['credits'], bool):
                            stats['confirmed_credits'] += row['credits']
                        else:
                            stats['responses_without_credit_field'] += 1
                    elif row['event'] == 'transport_error':
                        stats['transport_errors_unknown_billing'] += 1
            usage = dict(counts)
        await self.call('init', digest=digest, config=cfg, keys=keys, state=state, usage=usage)

    async def job(self, ident):
        return await self.redis.hgetall(self.prefix + 'job:' + ident)

    async def health(self):
        r, p = self.redis, self.prefix
        async with r.pipeline(transaction=False) as pipe:
            pipe.hkeys(p+'keys'); pipe.hgetall(p+'state'); pipe.zcard(p+'queue')
            pipe.zcard(p+'active'); pipe.hgetall(p+'metrics'); pipe.lrange(p+'queue_ms',0,9999); pipe.time()
            keys, states, queued, active, metrics, timings, clock = await pipe.execute()
        now = clock[0]+clock[1]/1e6
        reasons = collections.Counter()
        eligible = []
        async with r.pipeline(transaction=False) as pipe:
            for key in keys:
                state = json.loads(states.get(key, '{}'))
                reason = state.get('reason')
                if reason in ('quota_exhausted','invalid_key','retired') or state.get('until',0)>now:
                    reasons[reason] += 1
                    eligible.append(False)
                else:
                    eligible.append(True)
                pipe.hgetall(p+'usage:'+key)
                pipe.zcount(p+'inflight:'+key, now, '+inf')
            values = await pipe.execute()
        usage = {key[:12]: {k: float(v) for k,v in values[2*i].items()} for i,key in enumerate(keys)}
        inflight = [int(values[2*i+1]) for i in range(len(keys))]
        verified = sum(eligible[i] and float(values[2*i].get('http_200', 0)) > 0 for i in range(len(keys)))
        samples = sorted(float(v) for v in timings)
        def percentile(q):
            return samples[min(len(samples)-1, int((len(samples)-1)*q))] if samples else None
        return dict(total_keys=len(keys), eligible_keys=len(keys)-sum(reasons.values()),
                    verified_eligible_keys=verified,
                    unverified_eligible_keys=len(keys)-sum(reasons.values())-verified,
                    quarantined_keys=dict(reasons), queued=queued, active=active, pending=queued+active,
                    inflight_keys=sum(v>0 for v in inflight), inflight_requests=sum(inflight),
                    metrics={k:int(v) for k,v in metrics.items()}, lifetime_usage=usage,
                    queue_wait_ms={'p95':percentile(.95),'p99':percentile(.99),'samples':len(samples)},
                    state_backend='redis', namespace=self.config['SEARCH_POOL_NAMESPACE'])

    async def close(self):
        await self.redis.aclose()
