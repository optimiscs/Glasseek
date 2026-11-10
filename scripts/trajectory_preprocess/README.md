# Trajectory curation and human review

`pipeline.py annotate` now uses rules-first curation. Its configuration must set
`canonical_input` to the corresponding safety/export JSONL, in addition to the
normalized `--input`. The gate verifies message/evidence binding and reruns the
existing safety scanner before model calls. Placeholder and unfinished answers,
missing observations, summary-only continuations and direct answers needing a
separate proof route remain separate from `annotation_error`. No old safety,
protocol, source/parse_type, evidence or independent acceptance gate is relaxed.

Active annotation modules are `eligibility.py`, `task_routing.py`,
`claim_audit.py`, `segment_audit.py`, and `selection.py`; `pipeline.py` remains
the orchestrator. `policy.py` remains the single implementation of the existing
evidence/behavior validators. Shared contracts live under
`Shared/glasseek_shared/curation/`. There is no second versioned runner.

The model receives strict dynamic JSON Schema with program-owned claim, edge,
step and segment IDs. Evidence is still checked against the complete recorded
observations. An independent whole-trajectory acceptance gate remains in place.
Semantic segments group complete assistant messages; causal assessments receive
only each action's prefix, excluding its own return and all future instructions.
Contribution assessment is separate and may see later outcomes. Reasonable
exploration and recovery are not rejected merely because they failed.

Behavior-label multiplicity and collisions introduced by normalization are
recorded separately from repeated tool actions. Every witness is validated;
equivalent labels share one score and retain their provenance, without making
the whole trajectory fail merely for duplicate labels. Invalid witnesses and
evidence-backed negative behavior still block acceptance. Task instructions live
in `task_requirements`, not world-factual constraints, and remain subject to the
independent requirements-satisfied gate. Review artifacts distinguish
`model_verdict/model_reason` from a program gate's final verdict/reason. Initial
positive or contrary judgments require located evidence. Semantic prompts retain
complete observations once rather than repeating migration metadata and per-claim
copies of the same source passage. Remaining annotation issues are reported as
`annotation_review`, separately from established quality issues.

The program builds `learn/context_only/review` supervision. It keeps all context
and binds the plan into the release content hash. The actual verl loader zeros
loss for context-only assistant messages. Pending plans and all-zero plans cannot
be published; safety, quality, real replay and human review remain mandatory.
Topic/difficulty labels are unverified model metadata, never acceptance criteria.
Eligible candidates are ordered across strata; family IDs keep split assignment
consistent. `report.py --input RUN --baseline OLD_RUN` compares the exact same
input records and separates routing, annotation failures and quality holds.

For local-only runs set `local_only: true`, use explicit loopback providers and
model IDs, and disable proxy inheritance. `execution.workers` and each provider's
`max_concurrent_requests` bound producer concurrency independently of vLLM's
`max_num_seqs`. The 128-row rerun uses eight producer requests per GPU (16 total),
while each vLLM instance keeps `max_num_seqs=16`. Set `record_first_prompts: true`
to retain first-attempt request bodies and their full schemas under `prompts/`;
privacy-mode clients never write these raw prompts. Use new output directories
and caches after policy changes. Outputs remain review candidates, not training
releases.

The read-only stage dashboard is served by the existing Quest viewer:

```bash
GLASSEEK_CURATION_RUN=/absolute/path/to/run \
  bash scripts/trajectory_review/start.sh --host 127.0.0.1 --port 6080
```

Open `http://127.0.0.1:6080/api/curation/pipeline` (forward port 6080 when
accessing a remote host). It refreshes every ten seconds, distinguishes executed
requests from quality passes, and exposes per-stage P95, dataset/outcome filters,
evidence verdict provenance, supervision decisions and complete recorded context.
It reads only the configured run's root GPU outputs, excluding archived attempts.
Multiple verification witnesses are displayed as evidence aggregation, not as
conflicts. Candidate counts never imply human approval or training publication.

Install the shared package and preprocessing dependencies from the repository root:

```bash
python -m pip install -e Shared -r scripts/trajectory_preprocess/requirements.txt
```

The pipeline normalizes XML tool calls, preserves Qwen `user` / `<tool_response>` messages, records call/response identities, removes runtime telemetry, performs exact deduplication and flags similar questions. `glasseek_contract.py` checks the evaluation launcher when present; a clean checkout uses the tracked standard `Evaluation/scripts/run_eval.sh` profile. Both are static checks, not execution receipts. OpenResearcher is excluded.

`web_search` and `web_extract` share the same batch policy. Search accepts a string or an array of up to five queries in `query`; extraction accepts up to five `urls`. Independent calls in one assistant message combine only when their other supported options match (including extraction goal and search result limit). Original batches retain their text or JSON without mandatory per-query/page splitting or inferred success. Different turns, intervening user/assistant context and other tool types prevent merging; original calls are never truncated or split to fit a batch.

Historical web arguments must be preserved by an explicit equivalent mapping. Search accepts only `query`, `limit`, and its `topn` alias; extraction accepts `url`/`urls` and the reading objective `goal`/`query`. A direct HTTP(S) `browser.open.id` becomes `web_extract.urls`; conflicting URL aliases, cursors, local resources and viewport/session options remain held. Unknown parameters such as `source` and `parse_type` hold the entire trajectory even when null or empty. They are never removed to pass validation. Held trajectories cannot be materialized.

Native IDs or explicit identifiers still establish individual response links. When individual links are ambiguous, a closed source turn containing only one compatible tool batch can instead establish a whole-batch association. This preserves opaque combined returns without inventing individual call/result pairs. Mixed tools/goals, conflicting identifiers and known missing native-ID responses remain held. Original call IDs, positions, return components and hashes remain in metadata and the source archive. Safety scans each source component independently for answer leakage and scans the full return for credentials and injection; publishing still requires quality, safety, replay and human acceptance.

The current local Qwen3.5/Qwen3.8 templates use `<tool_call><function=...><parameter=...>`, with no `<tool_calls_begin>` outer frame. A balanced legacy frame containing only calls is removed; incomplete or ambiguous frames are held. Source tool schemas and explicit JSON-call examples in instruction messages are reconciled with the runtime schema/XML format, while task requirements and source text remain archived. Training validation rejects legacy outer frames that survive normalization. JSON arrays/objects in parameter values remain typed data.

Every linked search response records `search_return_audit`: batch association, query-label coverage and historically reported execution status are independent. A shared array title is not per-query attribution; partial/rewritten labels stay marked for review even when the batch binding is unique. Error returns remain errors, and unreported execution never becomes success. Return bodies stay unchanged. Use `scripts/trajectory_preprocess/unify_filtered_search.py` to normalize an existing filtered pool, and `audit_xml.py` for seeded samples rendered with local tokenizer templates.

Run `full_prepare.py --help`, `migrate_searchswarm.py --help`, `safety_pipeline.py --help`, and `publish_release.py --help` for independent batch entrypoints. Use a new output directory whenever rules or inputs change. `config.example.json` maps stages to OpenAI-compatible API or local providers. Supply credentials through environment variables or local key files; do not commit credentials or datasets. The default safety policy ignores email/phone and checks credentials, answer leakage, URLs, code and tool contracts.

The launcher `start_full_prepare.sh OUTPUT` accepts `PREPROCESS_PYTHON` and `PREPROCESS_WORKERS`. Its AutoDL defaults are conveniences; override them on other hosts. Input patterns are configured in `inputs.datasets.json`.

Scientific imports (`numpy`, `scipy`, `pandas`, `matplotlib`, `sympy`, `dateutil`)
are available in the pinned Python 3.8 container specified by
`Evaluation_backend/tools/scientific_runtime.json`. Every invocation gets a
fresh container: no network or host mounts, a read-only root, private bounded
`/tmp`, dropped capabilities, no-new-privileges and CPU/memory/task limits.
Runtime and supervisor hashes participate in the cleaning contract. In-memory
buffers and literal private-temp paths are distinguished from host/network I/O;
static checks remain conservative and are not the security boundary.
Use new output directories after rule changes.

`replay_compute.py --input SAFETY_EXPORT --output NEW_DIRECTORY` executes only
recorded code calls in that runtime, compares stdout with the recorded return,
and retains exact matches in `recovered.jsonl`. Failures and differing output
stay in the replay ledger. Web calls and factual quality are not replayed or
approved. Receipts are resumable only for identical input/runtime/code hashes.
This recovery audit uses 30 seconds per code call and compares stdout after
removing outer whitespace only. A timeout or stochastic/formatting difference
is a review hold, not a dangerous-code verdict. Later calls in a trajectory are
skipped after its first failed or differing call; they are never marked passed.
The production runtime independently accepts deadlines up to 300 seconds.

For the offline-passed pool, retain source IDs, hashes and parent/child groups.
Apply evidence-span annotation and behavior labels with `pipeline.py`, followed
by independent acceptance with `quality_gate.py`. Calibrate stratified samples
before scaling model calls; never silently truncate long trajectories. Separate
uncertain, negative and positive outcomes. Deduplicate and split by task and agent
family before training. `publish_release.py` additionally requires safety,
quality, replay and human-review receipts; offline pass is not training approval.

## Human review

The UI lives in the companion [QUEST viewer](https://github.com/optimiscs/QUEST). Check out both repositories as sibling folders, or set `QUEST_ROOT` and `GLASSEEK_ROOT`. The viewer backend mounts the review API only when the Glasseek review package exists. Set `GLASSEEK_REVIEW_DB` to a prepared batch database, then run:

```bash
bash scripts/trajectory_review/start.sh --host 127.0.0.1 --port 6080
```

Open `/curation.html`. Prepare a bounded batch with `python scripts/trajectory_review/prepare.py --help`, and attach redacted source snapshots with `python scripts/trajectory_review/attach_sources.py --help`. Source browsing offers role-based messages, original JSON, downloads and full-record search. Source coordinates and hash mismatches remain visible. Review identities are self-reported; this is a local review tool, not an authenticated multi-user service.

The 12-question rubric is scored server-side. Critical failures override totals; unknown answers cannot pass. Program labels are revealed after submission, with per-check agreement. Single reviews are not adjudicated gold labels. No human approvals, tool replay receipts or training readiness are fabricated. Publishing requires hash-bound safety, quality, replay and review acceptance; the SFT loader verifies the release manifest before loading.

## Offline tests

`normalize_training_text.py` is the single post-rubric text normalizer used by
pilot selection and checked again by the strict publisher. It fixes associated
response labels, exact unquoted tool aliases in complete reasoning, known task
wrappers, exactly duplicated adjacent questions, answer/report envelopes, and
empty or redundant thinking openers only when the recorded calls validate.
Requested `FINAL ANSWER:` prefixes and child `<report>` contracts are retained.
Nonempty reasoning with missing boundaries and native line citations without a
verified source mapping remain held. Tool return bodies and source IDs stay
unchanged; deleting an exact duplicate reindexes masks and valid supervision
plans without changing their learn/context decisions. Existing safety holds
remain holds. Content changes invalidate previous acceptance and safety hashes.

Normalize canonical JSONL into a fresh directory, then compare every output ID
against the original stratified audit sample and rerun the existing safety scan:

```bash
PYTHONPATH=Shared:scripts/trajectory_preprocess python scripts/trajectory_preprocess/normalize_training_text.py \
  --input INPUT.candidates.jsonl --output CLEANED
PYTHONPATH=Shared:scripts/trajectory_preprocess python scripts/trajectory_preprocess/audit_training_repairs.py \
  --verify-cleaned CLEANED --sample AUDIT/sample.jsonl --output VERIFICATION \
  --contract CONTRACT/tools.json
```

The verification writes per-trajectory checks, safety findings and a summary;
it exits unsuccessfully for mismatched bodies, bindings, protocol, routing,
idempotence or safety. These artifacts do not grant runtime replay or release
acceptance and do not overwrite existing training Parquets.

```bash
export PYTHONPATH=Shared:scripts/trajectory_preprocess:scripts
python -m unittest discover -s scripts/trajectory_preprocess -p 'test_*.py' -q
python -m unittest discover -s scripts/trajectory_review -p 'test_*.py' -q
```

These tests use synthetic records and mocked transports; no paid model calls or live tool execution are required. Generated datasets, review databases, credentials and local reports stay outside this change.

## Canonical entrypoints

`pipeline.py` uses `policy.py` for evidence-ID annotation, source-scoped review and deterministic scores. There is one policy implementation; omitted `policy_version` selects it. The `evidence-id-v4` configuration identifier remains accepted for existing configurations. Historical pilot policy selectors fail explicitly; use a new output directory when switching policy. Code hashes continue to bind outputs to their implementation.

Reusable exports, candidate unification and validation live in this directory. The SearchSwarm launcher runs migration, review export and full-corpus validation; the four-example comparison and local-model mask demo are archived. These offline checks do not certify runtime replay or training acceptance.

`report.py safety --run RUN --key-file KEY` generates redacted review reports. Input examples use `data/inputs/trajectory-samples.jsonl`; supply your own input or pass `--inputs`. New artifacts belong in `data/processed/` or `outputs/`. Historical reports and pilot code live only under `archieve/` and are not runtime dependencies.

For a full raw-data cleaning run, pass the seven non-SearchSwarm input specifications to `full_prepare.py` without a sample manifest, and expand all SearchSwarm bundles with `migrate_searchswarm.py --compress-archive`. Unify with `unify_trajectory_candidates.py`, run `safety_pipeline.py scan`, then `safety_pipeline.py export --verify-protocol`. The export validates canonical XML with the runtime parser before propagating held/missing children to parents. `report.py cleaning --run RUN` reconciles `intake/`, `searchswarm/`, `unified/` and `safety/` into `final/` and `report/`; it fails if any raw trajectory is missing from the outcomes. `report/rule_matrix.csv` uses datasets as rows and rule memberships as columns. Each cell is final migrated/held trajectory counts; columns overlap, quarantine is included in held, and exact duplicates are counted separately. Unparseable records remain explicitly unattributed to tool families. These are offline outcomes, not runtime replay or training approval.

### 单次整轨迹 rubric 初筛

trajectory_rubric.py 使用 prompts/whole_trajectory.zh.txt 中的完整中文 rubric。
模型只返回六个维度的等级、四个硬门槛和原始片段引用；程序按
30/20/20/15/10/5 加权。unknown 不当作零分质量样本，高分不抵消门槛失败。
同一条记录的多次验证不扣分；结构化判定之间的冲突记为待复核。
所有输出均为初筛结果，training_ready=false，不代替原来的独立验收与训练发布流程。

rubric_batch.py 在发送数据前复用 eligibility.assess 和现有安全 Scanner；
限定本地 qwen3.5-9b-local，每卡生产并发 16，生成和传输均仅允许一次尝试。
失败独立记为 annotation_error，不能推断原数据无效。中断后不自动重发结果未知的请求。
SQLite 保存输入绑定与进度，结果追加到 results.jsonl，可排序记录输出到 ranking.jsonl。
同分记录按 ID 稳定排列，不表示它们有质量差异。

以 PYTHONPATH=Shared:scripts/trajectory_preprocess 运行 rubric_batch.py，
传入 --input、--canonical、--config、--output；128 条配置保存在
data/processed/qwen35-9b-rubric128-20261109/config.json。
已有输出只有在策略和配置指纹相同的情况下才能续跑，不应删除失败记录来重新请求。
使用同一 Python 环境运行 -m reporting.rubric --input 新版目录 --baseline 多阶段目录生成对比。

扩大到全量前必须验证同一批输入、规则分流一致、每条至多一次请求、引用与程序计分、
测试结果，并抽查语义准确性和排名区分度。程序检查通过不等于语义验证通过。
--prepare-only 只建立输入索引，不会调用模型；它不表示全量推理已经启动。
