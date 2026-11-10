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

`--workers` is a global inference limit across benchmarks and endpoints. Scoring
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
