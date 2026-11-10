# Shared tool services

Long-lived tool services belong here. Their public defaults live in
`Evaluation/config/`; credentials, PID files, logs, and mutable state belong in
`Evaluation/data/runtime/` or an explicitly configured private location.

| Capability | Implementation / entry | Configuration |
| --- | --- | --- |
| Concurrent search pool | `search_pool.py serve` → `serper_proxy.py` | `../config/search_pool.json` |
| Jina fetch | `tool_services.py serve fetch` → `jina_fetch.py` → shared `provider_pool.py` | `../config/fetch_pool.json` (limits), `../config/tool_services.json` (port) |
| BM25 preparation | `tool_services.py serve bm25` → Quest's existing `bm25_pool_service.py` | `../config/tool_services.json` |
| Embedding adapter | `tool_services.py serve embedding` → Quest's existing `retrieval_vllm_adapter.py` | `../config/tool_services.json` |
| Page retrieval bridge | `tool_services.py serve bridge` → `quest_retrieval_bridge.py` | `../config/tool_services.json` |
| Incremental evaluation judge | `async_judge.py` → existing `browse_comp_eval.judge` classes | `../config/async_judge.json` |

From the repository root:

```bash
source Evaluation/services/tool_services_env.sh
bash Evaluation/services/start_search_pool.sh
# Run each foreground service under the existing supervisor:
python3 Evaluation/services/tool_services.py serve fetch
# Start the existing GPU embedding/retrieval stack only when needed:
bash Evaluation/services/start_quest_retrieval.sh
```

`tool_services.py serve search` (also `serper`) delegates to the same search pool
launcher. It does not maintain another queue, credential pool, or rate limiter.
Training, evaluation, and data production use the shared search settings described
in [README.search-pool.md](README.search-pool.md).

`tool_services.py command SERVICE` prints the resolved command without loading
credentials or starting a process. `tool_services.py env` emits shell-quoted
routing variables. Shell consumers source `tool_services_env.sh`, which also loads
the shared search settings. Experiment names, datasets, model checkpoints and
training hyperparameters remain in experiment configurations.

Defaults can be replaced with `GLASSEEK_TOOL_SERVICES_CONFIG`. Existing routing
environment variables override defaults. The launcher derives bind ports from
the same URLs it exports to clients. Quest's BM25 and embedding adapter currently
bind only to `127.0.0.1`; the launcher rejects other bind hosts for these services.
`start_quest_retrieval.sh` also derives its embedding GPU port from the shared
upstream URL. It retains ownership of GPU startup and PID/log management.

Set `QUEST_ROOT` and `QUEST_PYTHON` for the existing Quest checkout/environment,
or `TOOL_SERVICES_PYTHON` for a service-specific interpreter. The default is the
sibling `Quest/workspace/env/bin/python`. Fetch credentials come from
`JINA_API_KEYS`, then `TOOL_SERVICES_CREDENTIALS`, then `REEVAL_CREDENTIALS`, then
Quest's private `api_key_pool.json` (`jina` list). Secrets are read only on fetch
startup and never emitted by `command` or `env`.

The tool configuration retains this machine's proxy default
`http://127.0.0.1:18200`; edit the config or override `HTTP_PROXY`, `HTTPS_PROXY`,
and `GLASSEEK_DOH_PROXY` together when deploying elsewhere. Localhost is always
added to `NO_PROXY` so pool and retrieval calls stay local.

The incremental judge accepts `--run`, `--dataset`, `--interval`,
`--handoff-lock`, `--handoff-stop`, `--workers`, `--retry-seconds`, and `--config`. `JUDGE_API_KEY`,
`JUDGE_BASE_URL`, and `JUDGE_MODEL` override credentials/endpoint/model;
`REEVAL_CREDENTIALS` remains a compatible private JSON source (`judge` field).
No judge key is stored in the public configuration.

The former local `analysis/` directory, generic experiment service forwarding
scripts, and runtime source links have been removed. Twenty old runtime bootstrap
scripts were verified, archived in `Evaluation/data/runtime/archive/`, then removed
from the runtime entry directory. New service implementations must not be added under
`analysis/`, `experiments/`, or `Evaluation/data/runtime/`.

## Evaluation and model profiles

`eval_run.py PROFILE full|smoke|judge` loads `Evaluation/config/eval_profiles.json`
and the shared search/tool configuration. Profiles are `base9b`, `rl9b`, `sft9b`, and `rl4`.
The small `run_*` scripts are named launch presets that delegate here. Fetch and
retrieval URLs are no longer copied into each launcher. Judge defaults come from
`async_judge.json`; credentials are read only when actually starting a judge.

`model_server.py PROFILE GPU PORT` loads `Evaluation/config/model_profiles.json`;
profiles are `base9b`, `rl9b`, `step39`, and `step76`. It supports model, served-name,
sequence-limit and context-length overrides. GPU caches now default to runtime
cache directories by profile/GPU; existing archived caches are preserved.

Both commands support `--print-command`, which resolves configuration without
starting a service, loading credentials, or using a GPU. Public configuration
files may be replaced using `GLASSEEK_EVAL_PROFILES` and `GLASSEEK_MODEL_PROFILES`.

Supervisors use `incremental_judging.watch_summary` with explicit script and
summary arguments; concurrent runs no longer overwrite module-global watcher
state. Record-level and summary-level scheduling remain distinct policies in
`browse_comp_eval/judge_policy.py`.

## Fetch admission and DNS validation

`/search` and `/fetch` use the same `provider_pool.py` dispatcher and
`search_pool_state.py` / Lua state transitions. The old `SERPER_*` names are an
internal compatibility contract; `provider_services.py` maps service-specific public
settings into it. Fetch uses Redis namespace `glasseek-fetch-v1`, separate
from search credentials, rate limits, queues and audit events. Both fail closed
when central state is unavailable.

The fetch pool keeps the `/fetch` request and success-response contract. Errors
include `error_code` and `retryable`; callers must inspect non-200 bodies. It
supports `Idempotency-Key`, returns `X-Fetch-Request-ID` / `X-Fetch-Queue-Ms`, and
exposes admission, queue, quota and DNS counters on `/health`. A sent request
whose response is lost is not automatically replayed. HTTP 401/402 quarantine
keys with at most three credential attempts per request; 429 cools a key down.
Reader responses are limited to 8 MiB. Credit usage is unknown when Reader does
not return it; response counts are not represented as a credit bill.

The canonical registry is `Evaluation/config/provider_services.json`. Run both
services in one process while retaining the existing ports:

```bash
python Evaluation/services/provider_services.py serve --services serper,jina
python Evaluation/services/provider_services.py status --services serper,jina
python Evaluation/services/provider_services.py sync-keys --services jina
python Evaluation/services/provider_services.py restore-key --services jina --key-id FULL_SHA256
```

`tool_services.py serve search|fetch` and the compatible `search_pool.py serve`
entry delegate to this launcher. Each provider has its own HTTP session,
Redis namespace, credentials and limits, while all use the same pool code.
For a new API, register its protocol adapter and configuration; do not copy
queue, admission, retry or connection-pool code. Health distinguishes eligible
keys with successful responses from keys not yet verified. A new key is not
known healthy just because it was imported. Known exhausted/invalid keys remain
quarantined across restarts and key synchronization; only explicit restoration
can re-enable them. A 429 cooldown expires automatically; this permits future
requests to use that key again, and does not blindly replay an uncertain request.

The launcher imports existing Jina credentials once into the private
`Evaluation/data/runtime/fetch_pool/keys.json` (0600); subsequent startup reads
central Redis state. Changes to that file do not implicitly reset live state.
Do not start a replica with different limits in the same namespace. Reconfigure
only after all replicas stop and active work and key leases drain, using the
shared state's `configure` transition. The Redis service must retain AOF
`appendfsync always` and `noeviction`.

`/validate` centralizes the existing external-fetch URL checks. Redis caches
successful checks for 5 seconds, confirmed NXDOMAIN for 30 seconds, and other
failures for 3 seconds. Per-host Redis leases coalesce concurrent lookups;
each replica has eight DNS workers and at most 128 waiting leaders. Private
addresses remain blocked. DNS NXDOMAIN, SERVFAIL and transport failures are
reported separately; unavailable validation never permits a fetch.
`tool_services_env.sh` routes Hermes prechecks to `/validate` and enables a
single client attempt with an idempotency token. `/fetch` validates again at
execution time, so direct callers cannot bypass the check. This is a Reader
proxy: redirect handling and final destination resolution also depend on the
Reader upstream's own protections.

Validation commands (isolated Redis and fake keys, no production credits):

```bash
python -m pytest tests/test_fetch_pool.py tests/test_search_pool_recovery.py
python scripts/validation/fetch_pool_load.py --output /tmp/fetch-load.json
```

The 64–1024 mock runs exercise the queue with a 100 ms upstream and 64 KiB
pages. They do not establish the real Jina quota, full trajectory throughput,
or model utilization. Production begins conservatively at global concurrency
64, queue 1024, 2 in-flight requests and 1 RPS per key. The evaluation remains
paused until explicitly resumed.

### Base 9B on BrowseComp 128

`base9b` uses the `step39_128` evaluation protocol: the same 128 questions and
system prompt, 131072 context, 16384 output tokens, compression at 0.5,
200 maximum rounds, one retry, no total time limit, and a shared 120-tool budget.
Its evaluator has eight workers and its vLLM profile has eight scheduled sequences.
The model profile explicitly enables thinking. Sampling overrides remain unset,
as in the step39 evaluation command.

```bash
python Evaluation/services/model_server.py base9b 0 19109
python Evaluation/services/eval_run.py base9b full --run-name YOUR_RUN
python Evaluation/services/eval_run.py base9b watch-judge --run-name YOUR_RUN
```

The evaluation automatically registers and publishes its trajectories to the
existing viewer at `http://127.0.0.1:6080/trace-viewer-zh/?dataset=YOUR_RUN`.
This is enabled by default for the Python evaluator, `run_eval.sh`, profile
launcher, and four URL mode launcher. No separate `publish` command is needed.
The publisher refreshes every 20 seconds, serializes concurrent imports, and
performs a final refresh before stopping with the evaluation. Viewer failures
are logged and retried independently of inference. See `viewer.json` and
`viewer.log` inside the evaluation output directory. The shared local Quest web service is reused, or started automatically via its
existing `serve.py` entrypoint when absent. A configured remote viewer is not
started locally. The database, viewer environment, and frontend build must already
be installed. The shared web service stays available after evaluation ends.

Use `run_eval.sh --no-viewer` to opt out or `--viewer-interval 10` to change the
refresh interval. Python accepts the same options; environment equivalents are
`GLASSEEK_VIEWER_ENABLED=0` and `GLASSEEK_VIEWER_INTERVAL=10`.
The explicit `publish` action remains available for historical runs.

Run the model, evaluation, and optional `watch-judge` under supervised processes.
`watch-judge` uses the existing incremental judge; supply `JUDGE_API_KEY` or
`REEVAL_CREDENTIALS` in its environment and use the evaluation's run name.
Check shared tools, remote embedding, the model endpoint, and viewer database
before starting.

### Four URL mode experiments

The archived `archieve/scripts/trajectory_benchmark/run_url_modes_live.py` recorded an incremental
DeepSeek judge for each mode by default, in addition to automatic live viewing.
Use `--no-judge` only for an intentionally unscored experiment. Scores are written
to each run's `judge_official/` and appear in the viewer. The launcher waits for
evaluation and judge completion; judge failures are reported in `status.json`
and `<mode>.judge.log`. Existing prediction fingerprints prevent duplicate scoring.
The judge drains terminal predictions after its evaluator exits and publishes
the final scores before stopping.

Credentials use `JUDGE_API_KEY`, then an explicit `REEVAL_CREDENTIALS` or
`GLASSEEK_API_CONFIG` file. Without an explicit file, the legacy credentials file
and existing Quest private API configuration are checked in that order. No key
is written to experiment manifests or command lines.

### Experimental CLM working context

Evaluation and RL/SFT launchers accept `--context-manager hermes|clm|clm-notes` and
`--context-manager-config /absolute/path/config.json`. The default is `hermes`;
no CLM schemas or prompts are injected on that path. JSON configuration accepts
`max_note_tokens` (4096).
Environment equivalents are `GLASSEEK_CONTEXT_MANAGER` and
`GLASSEEK_CONTEXT_MANAGER_CONFIG`; Ray rollout workers receive both variables.

The legacy `clm-notes` adaptation adds `context_edit(base_version, content)`. Only a standalone edit turn is
accepted. The original task and system instructions stay pinned, source IDs must
already occur in tool observations, and stale/empty/oversized edits are rejected.
Hermes preserves the latest complete tool action across edits; RL closes the
current segment and renders a fresh prompt after a valid edit. Original sampled
IDs, behavior logprobs, action masks and prior segments are preserved. Budget
pressure without a usable edit uses the existing summarizer. This is a constrained
working-note adaptation, not the paper's unrestricted file-editing harness.
Hermes uses its existing rough message-token estimator and hard request preflight;
RL uses its actual tokenizer. The working-note state is currently in memory;
full CLM version/state restoration across a fresh process is not implemented.

SFT `--context-manager clm` requires canonical trajectories containing
`context_edit` tool calls. It does not synthesize training labels from ordinary
summaries. The viewer shows context edits and their receipts as ordinary tool
steps. Pilot reports are in `archieve/reports/clm-pilot-20261108/`; the four-prefix
comparison measures note generation, not end-to-end answer quality or an RL run.

### Official CLM: per-task local paths

The independent official runner (`glasseek_shared.context.official.run`) defaults
to the existing local terminal backend. Every output workspace owns
`.clm/context/LIVE_CTX_MAIN.txt`, `.clm/bash/`, and `.clm/paths.json`. The official
agent is loaded into a private module per task; only the path constants and their
literals in the system template are bound to these paths. Its budget hints and
mirror I/O read the same bindings. Upstream source files are not patched.

Independent workspaces can run concurrently. A per-workspace nonblocking lock
rejects duplicate launches into the same workspace; there is no global CLM lock
and no implicit job queue. The workspace must have a shell-safe absolute path
(letters, digits, slash, underscore, dot, hyphen), because upstream interpolates
these paths into commands and Python snippets without quoting.

Official checkpoints include the `.clm` directories. On upload into a different
workspace, the local adapter rebinds stored bash cwd/env paths using `paths.json`.
Historical trajectory messages are not rewritten. Provenance records prompt
hashes and the exact substitutions. These bindings apply to the official backend selected by the existing evaluation
and RL launchers with `--context-manager clm`. The older `context_edit` adaptation
is selected explicitly with `--context-manager clm-notes`.

Validation: `PYTHONPATH=Shared .cache/clm-official-env/bin/python -m pytest -q
tests/test_official_clm_task_paths.py`. The test drives two upstream agents with
deterministic LM replies, applies mirror edits, checks shell separation, and
restores both checkpoints into new workspaces. It makes no model/API requests.

### Glasseek entrypoints for official CLM

Use the existing `Evaluation/scripts/run_eval.sh` or
`Evaluation/services/eval_run.py` with `--context-manager clm`; no alternate user
entrypoint or Docker is required. HermesClient launches an internal official
worker under its existing cancellation/timeout tracking. Results use the existing
conv schema, incremental judge, native session exports, and live viewer publisher.
Official bash commands call a task-local helper for web_search/web_extract; its
loopback bridge starts and stops with the worker. The URL registry is checkpointed
inside `.clm/web-state.json` and loaded after workspace restoration.

RL uses the same upstream agent loop and local environment, substituting only
its model transport with the existing verl generator. Every actual sampled call
records its exact prompt IDs, action IDs, behavior logprobs and mask as a segment;
context edits never rewrite recorded actions. This per-call segmentation differs
from the legacy compaction-based segment boundaries. No efficiency reward or
full GPU training experiment has been introduced or validated here. SFT CLM data
must contain canonical bash actions rather than context_edit notes.

The default `hermes` path is unchanged. For CLM install optional dependencies in
the executing environment via `pip install -e third_party/context-language-models`
and `pip install mcp` (required by the resolved LiteLLM package). Evaluation defaults
to `.cache/clm-official-env/bin/python`, overridable by `GLASSEEK_CLM_PYTHON`. Ray
workers must have the pinned source checkout and dependencies available.

The existing generation and context limits are forwarded to the official worker;
the upstream BCP defaults supply remaining settings. A custom Hermes system prompt
is not used as the CLM system prompt: CLM retains the upstream template with task
path substitutions. Model results must be written to the task's answer.txt.

For `--context-manager clm`, the shared JSON config may contain an `official`
object with upstream reserve/nudge/retry/checkpoint settings. Unknown keys fail
fast. Model output/context/step caps continue to come from the existing launcher
arguments; no caller must invoke the diagnostic standalone module.

### Shared Hermes recording schema

All new official CLM producers (evaluation and RL) write native Hermes sessions
and messages through SessionDB, with standard API-usage and tool-call logs.
Context rewrites create parent/child sessions and inherited messages; original
actions are not overwritten. Derived answer files are result metadata, not fake
assistant generations. ATIF-CTX export is disabled in the Glasseek path.

`message_timings` associates measured durations and optional wall-clock anchors
with message IDs and phases. Legacy imports preserve null anchors.
`request_contexts` stores request messages, actual tool schemas and classification
metadata; API input totals include cache, while session input counters exclude it.
Classification is explicitly an estimate of serialized request text excluding
chat-template overhead. The viewer uses these common records for every producer.
Old transcripts without tool definitions retain an unknown tool count.

The terminal-only legacy importer backs up predictions and verifies unchanged
scoring inputs before rebinding cached judgment fingerprints to metadata-only
changes. It does not invent missing event times or import unfinished attempts.
Reports and backfill coverage: `archieve/reports/clm-hermes-schema-20261108/`.


### Isolated scientific computation

The evaluation service sets `HERMES_REMOTE_PURE_COMPUTE=Lab3090`. This route
uses `tools/remote_pure_compute.py` and the trusted
`tools/scientific_container_worker.py` supervisor. It never falls back to the
host interpreter or the unrestricted terminal backend. The legacy local
filesystem-denied launcher is not used by this evaluation route.

`Evaluation_backend/tools/scientific_runtime.json` pins the Docker image ID,
Python 3.8 scientific distributions, wheel hashes and resource budgets. Build
with `scientific_runtime.Dockerfile` using a directory containing that manifest
and the exact wheels. Download Python 3.8 conditional dependencies explicitly
when provisioning from a newer Python. Build offline (`docker build
--network=none`), record the resulting immutable image ID, then run the opt-in
integration tests before updating cleaning artifacts. No package installation
or image pulling occurs during dataset execution.

Each call has 2 CPU equivalents, 2 GiB memory with no additional swap, 64 tasks,
a 128 MiB private `/tmp`, 16 MiB shared memory, an 8 MiB per-file limit and a
200 KB combined output budget. At most four jobs from this runtime run on the
host at once. The root is read-only, network is disabled, no host volumes or
Docker sockets are mounted, capabilities are dropped, and no-new-privileges is
set. Code runs as UID 65534. BLAS defaults to two threads; cgroups enforce the
actual resource bounds. Private temporary files and all process state disappear
when the supervisor removes the container.

The remote supervisor enforces the requested deadline (capped at 300 seconds),
terminates output floods, distinguishes OOM from execution errors and confirms
container removal. Transport cancellation also requests removal by an unguessable
job ID. A missing receipt or unverified cleanup is an error, never a pass.

Run `GLASSEEK_TEST_REMOTE_SANDBOX=1` with `tests/test_remote_pure_compute.py` to
validate the actual scientific stack, cgroup limits, threads, memory/output
limits, network/host-file isolation and fresh temporary directories. Ordinary
unit tests alone do not certify remote deployment.
