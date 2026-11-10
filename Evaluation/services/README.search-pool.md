# Shared asynchronous search pool

Training, evaluation and data production use the same `POST /search` endpoint and
implementation (`serper_proxy.py`). Responses retain the upstream Serper shape;
failures retain `success`, `error_code`, `error`, and `retryable` fields. HTTP I/O
uses an `aiohttp.ClientSession` connection pool. Redis owns the shared queue, admission, credential state,
request results and audit ledger; process-local counters never authorize calls.

## Installation and startup

Install the `search-pool` extra from `Evaluation/pyproject.toml` in the service
interpreter. Redis 6+ is required. Redis is a separate long-running dependency:

```bash
# Terminal/service 1: Redis, bound to localhost, AOF fsync always, no eviction.
bash Evaluation/services/start_search_state.sh

# Terminal/service 2: API and asynchronous workers.
bash Evaluation/services/start_search_pool.sh serve
bash Evaluation/services/start_search_pool.sh check
```

`start_search_state.sh` uses a system `redis-server` or `SEARCH_POOL_REDIS_BIN`.
On this host the Ubuntu Redis binaries/dependencies are extracted under
`.cache/search-pool/redis`; they are dependencies, not another service implementation.
Redis data lives in `Evaluation/data/runtime/search_pool/redis` with private permissions.
The startup script reads `Evaluation/config/search_pool_redis.conf`.

Search configuration is `Evaluation/config/search_pool.json`; environment variables
override matching settings, and `GLASSEEK_SEARCH_POOL_CONFIG` selects a complete
alternative configuration. `SEARCH_POOL_REDIS_URL` may contain Redis credentials;
keep those in private environment configuration, not versioned JSON. For remote
replicas use a reachable Redis with ACL/TLS and a shared namespace. The supplied
localhost Redis configuration is for this host only, not multi-host HA.

`SEARCH_POOL_WORKERS` is local execution capacity (default 128).
`SERPER_PROXY_CONCURRENCY` is **global** execution capacity (default 1024), and
`SERPER_PROXY_QUEUE_LIMIT` is an additional **global** waiting capacity (1024).
A single default replica can execute at most 128 jobs; additional replicas share
the global cap. Actual provider traffic is also bounded by global RPS and each
credential's concurrency/RPS. More replicas do not create more provider quota.
The queue deadline is 45 seconds by default. Queue time and execution share the
original 60-second caller budget (with a response margin); recovering a job does
not grant a fresh full budget. Upstream execution is additionally capped at 45 seconds.

A second process on a different port can share the Redis namespace. Admission
settings must match; incompatible replicas fail startup. Redis must use AOF with
`appendfsync always` and `maxmemory-policy noeviction` by default. State outages
fail closed, without falling back to independent local admission.

For configuration changes: drain traffic, stop all API/worker replicas, edit the
configuration, run `search_pool.py configure`, then restart all replicas. The
command refuses changes while queued/active jobs or credential leases remain.
It preserves credentials, quarantine, completed results and usage history.

## Credential migration and administration

On the namespace's **first startup only**, the service imports credentials from
`Evaluation/data/runtime/search_pool/keys.json`, quarantine from `key-state.json`,
and usage counters from the legacy `search-key-usage.jsonl`. Import is atomic and
idempotent; subsequent replicas use central credentials. The legacy files remain
as migration evidence; no running worker writes them.

To add/remove keys, atomically update the private `keys.json` (`{"serper":[...]}`,
mode 600), then publish it to all replicas:

```bash
bash Evaluation/services/start_search_pool.sh sync-keys
```

Unlike the old implementation, editing the file alone does not update the live
pool. This explicit publish step prevents stale replicas overwriting central state.
Quota-exhausted/invalid/retired credentials remain disabled until explicitly restored.
After recharging a credential, use its full SHA256 fingerprint, never the secret:

```bash
bash Evaluation/services/start_search_pool.sh restore-key --key-id FULL_SHA256
```

This works online. Do not edit the legacy state JSON to restore a live credential.
HTTP 401/402 and recognized insufficient-credit responses quarantine the key and
rotate, at most three attempts per request. HTTP 429 cools the key and returns a
typed failure without sweeping the pool. Timeout/transport ambiguity is **not**
automatically retried: it may already have been billed. Its key slot remains
reserved until the lease expires, to avoid immediately admitting replacement work.

## Durable queue, recovery and idempotency

Redis scripts use the server clock for atomic FIFO admission and key rate/concurrency
limits. Worker leases are periodically renewed. An expired owner cannot renew,
acquire a key, or finalize a job after another owner has taken over. Lease duration
must exceed the whole HTTP request timeout, including socket operations.

* Queued or claimed-but-unsent jobs survive API/worker loss and can be reclaimed.
* A lost worker's dispatched job becomes `search_outcome_unknown`, with
  `retryable=false`; the pool never automatically replays it.
* An active caller may lose its HTTP connection if its API process dies. Durable
  work recovery does not restore that connection.
* Callers may supply `Idempotency-Key` (up to 256 characters). Reusing the same key
  and identical payload waits for or returns the original result across replicas.
  Reusing it with a different payload returns HTTP 409. Completed results remain
  for `SEARCH_POOL_RESULT_TTL` (default one hour); after expiry reuse can be billed
  again. Existing clients need no changes, but without a supplied key their new
  HTTP requests are independent operations.
* `X-Search-Request-ID` is a tracing identifier, not an idempotency token.
* The pool does not claim exactly-once third-party billing. Sending a request and
  storing its response cannot be one transaction with an external provider.

Successful responses include `X-Search-Queue-Ms` (enqueue to upstream admission,
including credential wait). `/health` shows global pending/active counts,
credential reservations, quarantines, aggregated usage, and queue P95/P99 over the
last 10,000 admitted attempts. `/health` readiness means Redis is reachable; it is
not an upstream provider availability/credit guarantee.

Redis Streams store fingerprint-only audit events before dispatch and after response.
The AOF also contains private credentials, queries, and retained results: protect
its directory/backups. Events and usage counters are not automatically trimmed;
monitor storage and archive the ledger before applying retention. Redis HA, ledger
archival automation, account-specific quota grouping, and cross-provider failover
are separate work. `SEARCH_POOL_MAX_UPSTREAM_ATTEMPTS` optionally puts a hard,
namespace-wide lifetime cap on attempts (0 means unlimited).

## Consumers

```bash
source Evaluation/services/search_pool_env.sh
bash Evaluation/services/start_search_pool.sh check
```

Training/evaluation launchers and the data workflow use the same endpoint. Direct
training Python callers must source the environment first. Consumer-side POST
retries remain disabled. Explicit mock/disabled evaluation modes stay isolated.
The service is intended for trusted internal callers; external exposure requires
an authenticated gateway, including a caller scope for idempotency keys.

## Validation

```bash
/root/autodl-tmp/Quest/workspace/env/bin/python -m pytest -q \
  tests/test_search_pool_unified.py tests/test_serper_proxy_concurrency.py \
  tests/test_serper_proxy_rotation.py tests/test_search_pool_recovery.py \
  tests/test_tool_repair.py tests/test_serper_top10.py

/root/autodl-tmp/Quest/workspace/env/bin/python scripts/validation/search_pool_load.py \
  --output outputs/validation/search-pool-load.json
```

The load harness launches isolated Redis/AOF, two replicas, a local mock upstream,
and synthetic trajectories at 64/128/256/512/1024 concurrency. It never loads
production keys. `--levels 1024 --steps 24` runs a longer scenario. The generator
also uses an aiohttp connection pool to avoid HTTP client bookkeeping becoming
the load-generator bottleneck. Synthetic trajectory throughput is not LLM agent throughput;
model utilization is explicitly null. Tests additionally cover global limits,
credit exhaustion, cooldown, budget caps, worker SIGKILL, fencing, queue recovery,
Redis SIGKILL/AOF restart and idempotent result replay.
