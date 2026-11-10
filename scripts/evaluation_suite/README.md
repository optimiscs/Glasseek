# Python evaluation CLI

The next fixed cohort is `data/evaluation/suite5pct-v1/round8-seed42-v3/`:
seven benchmarks × eight fresh tasks, seed 42, excluding the previous two-task
cohort, current34 and the existing training inventory. Mind2Web2 is excluded at
the user's request. Sampling uses each pinned complete population because the
original 5% GAIA/DRBench/LRB pools contain fewer than eight unused tasks.

```bash
python scripts/evaluation_suite/run.py \
  --suite data/evaluation/suite5pct-v1/round8-seed42-v3 \
  --phase all --seed 42 --workers 5 --profile base9b \
  --context-manager clm --url-mode raw --viewer --max-tool-calls 120 \
  --score-config scripts/evaluation_suite/score.deepseek.json \
  --corpus-config scripts/evaluation_suite/corpus.local.json \
  --run-id suite_round8_raw_clm_c5_s42
```

The 56 tasks are already frozen; `--seed` must match that release. It does not
resample a frozen release or change model decoding randomness. The viewer and
native graders start automatically. The fixed-corpus service is also managed by
the Python entrypoint, with no separate user startup command.

BC+ uses all 100,195 documents at the pinned corpus revision and an explicitly
declared SQLite FTS5 BM25 retriever, not the author's Lucene implementation.
Search and extraction record task-scoped docids; absent URLs have no live-web
fallback. CLM bash descendants can create Unix sockets only and reach a per-task
Unix web bridge. Normal live-web evaluations retain their existing routing.

Audited fixes preserve pinned author files: WideSearch's empty DataFrame
intersection becomes a typed empty Series before native metric calculation;
DRBench's zero-denominator FACT ratios are explicitly undefined with citation
coverage recorded. BC+ uses its original rubric/parser with an explicit
Responses-to-Chat-Completions transport for DeepSeek Flash. Source/runtime hashes
and each scoring attempt remain recorded. Two LRB tasks in this cohort lack
author depth references; they retain the other four dimensions with depth marked
unscored rather than replacing the sampled tasks or inventing reference reports.

Run from the repository root. This entrypoint samples the frozen evaluation
release, schedules inference, publishes live trajectories and scores completed
benchmarks with their own protocols. Inference launches the existing
`browse_comp_eval.run` Python module through `Evaluation/services/native_eval.py`;
no shell launcher is required. The native entry retains the fetch health and
functional watchdog, isolated worker cleanup and active-case failure recording.

```bash
python scripts/evaluation_suite/run.py \
  --phase all \
  --per-benchmark 2 --seed 20261108 \
  --workers 5 --profile base9b \
  --context-manager clm --url-mode raw \
  --viewer --max-tool-calls 120 \
  --score-config scripts/evaluation_suite/score.deepseek.json \
  --allow-blocked \
  --run-id suite_smoke2_raw_clm_c5
```

Without `--affinity-endpoints`, `--workers` is a global inference limit across benchmarks and endpoints. Scoring
uses the separate configured judge service and retains native metric scales.
`--viewer` defaults to enabled; `--no-viewer` disables publishing. Input token
accounting includes cached input once, output includes reasoning and final text,
and judge token usage is reported separately. There is no wall-clock task timeout.

Use `--phase plan` to verify inputs and print commands without model or judge
requests. The default phase is `plan`; running inference must be explicit.
`--benchmarks gaia,hle` selects named families. `--eval-python PATH` and
`--score-python PATH` select interpreters without shell environment setup; by
default the existing Hermes runtime and suite score-venv are used. The outer
Python CLI needs only the standard library. Credentials are loaded by the judge
from its local configuration, never passed as CLI arguments.

Reports are written to `Evaluation/output/suite5pct/<run-id>/`: `summary.csv`,
`report.json`, `cases.jsonl`, `status.json`, and `scores/`. Predictions live in
`Evaluation/output/suite5pct/predictions/<run-id>_<benchmark>/suite_<benchmark>/`.
The same run-id can resume only with matching code, inputs and settings; use a
new run-id when those change.

BrowseComp-Plus remains blocked without `--corpus-config` for its fixed-corpus
backend; `--allow-blocked` records this and continues other families. Mind2Web2
native evidence may require a vision-capable judge; unsupported modalities are
reported as unscored. Failed judges and missing measurements are never filled
with invented scores or token counts.

Scoring defaults to `--score-schedule trajectory` during `--phase infer` or
`--phase all`: completed valid predictions enter a separate two-worker native
grading queue while inference continues. Use `--score-config` to select the
judge (for example `score.deepseek.json`); no judge provider is changed implicitly.
`--score-schedule off` disables automatic grading; `--phase all
--score-schedule dataset` retains dataset-end grading. Per-task packets, native
outputs and durable claims live in `trajectory-scores/`. Existing grades and
claims are skipped on resume; interrupted or failed claims require inspection
before an explicit retry, to prevent duplicate paid calls. Single-task native
metrics retain their scope; aggregate accuracy uses the full assigned count.

Attach to existing inference without restarting it using
`python scripts/evaluation_suite/trajectory_scoring.py --run-id RUN
--benchmarks browsecomp,hle --score-config scripts/evaluation_suite/score.deepseek.json
--eval-python Evaluation/data/runtime/venv/bin/python`. It scans every two
seconds and publishes each completed grade to the existing viewer.

Override only the research system prompt with
`--system-prompt-file Evaluation/config/prompts/research-compact.txt`.
HLE always retains its official system prompt; user templates and tool schemas
are unchanged. The override path and content hash are recorded in launch.json,
and changing the prompt requires a new run ID.
Use `compare_prompts.py --before OLD_GPU0 OLD_GPU1 --after NEW_GPU0 NEW_GPU1
--out comparison.json` for paired task IDs, native grades, timing and tool counts.
Missing scores are not interpreted as incorrect answers. Live web variation and
unfixed model sampling mean a single comparison is observational, not proof of
statistical significance.

## AREX native research and dataset affinity

Use a **new run ID** when switching protocols. `--research-prompt arex-native`
switches the system and user templates together and enables the native `finish`
contract. The original `legacy` mode remains the default so running experiments
and the existing compact-prompt comparison do not silently change protocol.
AREX native currently supports Hermes with raw URLs; fixed-corpus BC+ is still
blocked and opaque-URL modes require a separate finish adapter.

```bash
PYTHONPATH=Shared:scripts/evaluation_suite .cache/clm-official-env/bin/python \
  scripts/evaluation_suite/run.py \
  --phase all --profile base9b \
  --benchmarks browsecomp,gaia,hle,drbench,liveresearchbench,widesearch \
  --research-prompt arex-native --context-manager hermes --url-mode raw \
  --affinity-endpoints http://127.0.0.1:19109/v1,http://127.0.0.1:19110/v1 \
  --workers 16 --max-tool-calls 120 \
  --max-reviews 2 --protocol-corrections 3 \
  --accept-confidence 95 --medium-confidence 90 \
  --score-schedule trajectory --score-config scripts/evaluation_suite/score.deepseek.json \
  --viewer --run-id base9b_197_arex_native
```

With `--affinity-endpoints`, `--workers` means **per endpoint** (16 × 2 = 32).
Both endpoints must serve the same model configured by `--profile`. Dataset
preferences initially follow size-balanced placement. Each endpoint prioritizes
its current dataset, but does not reserve the whole GPU: if it has only two
remaining trajectories, other datasets immediately receive its other 14 slots.
A pending whole dataset is preferred over spreading another GPU's active dataset
when both can fill the gap. Dispatch is recalculated each second and workers
refresh grants every 0.5 seconds; completion/waiter checks run every 0.1–0.5 s.

Every dataset process acquires a kernel-backed lease from the same per-endpoint
slot pool before starting a trajectory. Current allocation is rechecked during
acquisition; overlapping/stale registry updates cannot exceed 16 active leases
on one endpoint. A lease spans the entire trajectory, including its tool waits,
and is released after completion or process exit. Changing quotas to zero only
pauses new claims, never cancels or moves in-flight work. This limit covers the
single evaluation run, not unrelated experiments using the same model server.

`affinity.json` records per-dataset endpoint quotas and active leases;
`scheduling/*.json` records queued/in-flight work. There remains one runner/writer
per dataset. No tasks are copied or restarted, and viewers and per-trajectory
graders retain their existing identities. Fewer than 32 remaining trajectories,
unavailable endpoints, or tool waits can still mean fewer than 32 active model
decodes. Use a new run ID for changed scheduler code; old-run migration remains
a separate operation. Either `legacy` or `arex-native` prompts work with this
scheduler.

`finish` accepts exactly `answer`, `evidences: [{evidence, url}]`, and finite
numeric `confidence` in [0,100]. One native tool is allowed per turn, enforced
both in API requests and before execution. Mixed calls have no side effects.
Invalid/ordinary-text submissions receive at most three corrections, then fail
explicitly. Program-observed URLs survive automatic compression. Research
submissions require evidence; self-contained HLE/GAIA calculations may omit URLs.
Report/table answer bodies are preserved verbatim. HLE and BrowseComp receive
their established Explanation / Answer (or Exact Answer) / Confidence envelope;
GAIA's native answer normalizer remains unchanged.

Candidates below `--accept-confidence` undergo a separate same-model review
without any reference answers. Medium confidence verifies the current candidate;
below `--medium-confidence` challenges it and proposes alternatives. Review
calls consume the same iteration budget and use the same endpoint; at most
`--max-reviews` reviews are allowed. Unresolved candidates, exhausted budgets,
and malformed reviews are not silently accepted. Confidence only controls
stopping, never benchmark correctness. Per-attempt `research-result-aN.json`
retains candidates, review inputs/outputs, token usage, durations and failure
reasons; the prediction includes this record for host-side inspection. The
existing sandbox request/result wire schema is unchanged.

The original BM25 + embedding selector returns source passages, not generated
summaries. Reader extraction is forced to text, and the old LLM summarizer is
bypassed. Automatic Hermes compression remains the sole context-management
entry; tool/network/file isolation and native benchmark graders are unchanged.
Prompt provenance and the upstream Apache-2.0 license are in
`Shared/glasseek_shared/research/NOTICE` and `LICENSE-AREX-2`.

Before-change rollback checkpoints (already pushed):
`checkpoint/pre-arex-start-20261110` and
`checkpoint/pre-arex-complete-20261110`. Each merged module PR is a merge commit;
use `git revert -m 1 <merge-sha>` in a new PR to undo one module without rewriting
main history. The complete tag is a reproducible baseline including all four
checkpoint PRs; do not force-reset shared main.

To continue compatible legacy-prompt experiments after changing the scheduler,
use a new coordinator `--run-id` and repeat `--resume-run-id OLD_ID` for the
previous GPU runs. Stop their producers first. Keep the same frozen suite,
context/URL protocol, system-prompt override, model, tool budget and grader config.
The CLI validates these settings and retains the old per-dataset save-name,
workspace, profile/session, predictions and viewer identity. Successful results
are skipped; interrupted attempts can resume their existing checkpoints. Paid
score claims and receipts are adopted, including unresolved/error claims, so a
coordinator change does not silently resubmit grading calls. The launch manifest
records the source identities and their hashes. Newly introduced datasets use
the new run ID; report and prompt-comparison tools follow this mapping. This is
explicit migration rather than bypassing the old launch manifest's code checks.

### Balanced incremental samples

The suite builder accepts `per_benchmark` as either one integer or a mapping
from benchmark name to nonnegative quota. With `exclude_suites`, this freezes
only new tasks, preserving existing immutable releases and results. Set each
quota to the desired total minus the old assigned count; select both prompt
variants from the same incremental release. Training-overlap exclusions still
apply, and unavailable quotas are reported as shortfalls rather than refilled
with excluded questions. Run the release through the existing `run.py --suite`
entrypoint and combine old and incremental run IDs with `compare_prompts.py`.

### Harness correction provenance

Runs now record `harness_revision` and each trajectory records its termination
reason independently of correctness. Tool-budget exhaustion does not imply the
task is solved; native judges still score the preserved final answer. Legacy
system variants share task-appropriate user guidance; HLE keeps its official
system and unwrapped question. Compare only explicitly paired task IDs and
report runtime revisions when combining historical and new runs.

Vision is pinned to the evaluated model and endpoint, with an actual image
preflight before inference. Retrieval rejects recognized challenge pages and
empty source content. The leak guard retains multilingual question tokens and
blocks matching solution excerpts in agent-evaluation papers; network and host
file restrictions remain unchanged.

DRBench executes a clean checkout of the pinned author commit; tracked source
changes still fail verification. Structured judge responses are saved before
validation, with at most one schema-correction request using the unchanged
rubric. Invalid/conflicting verdicts remain unscored. `scoring_status_counts`
and `dimension_coverage` distinguish partial evaluation from complete scores.
For targeted replay, `smoke.prepare(..., selected_ids={...})` freezes an explicit
subset without modifying the parent release or existing predictions.
