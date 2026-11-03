# Evaluation

BrowseComp runner for the local Hermes backend and AGS sandboxes. Only the local launcher, AGS launcher, and judge launcher are included. Input datasets and run results are not distributed.

## Setup

Requires Python 3.11–3.13. Install the sibling [`Evaluation_backend`](../Evaluation_backend/README.md) first, then from this directory:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[ags]'
cp config/credentials.env.example config/credentials.env
# Edit config/credentials.env with your own credentials and service URLs.
set -a; source config/credentials.env; set +a
```

Provide a licensed JSONL dataset outside Git, for example `data/browsecomp/questions.jsonl`. Each row needs `question`, `answer`, and `type`; `id` is optional. The answer is used only by the offline judge.

## Local run

```bash
scripts/run_eval.sh \
  --model YOUR_MODEL --base-url "$HERMES_BASE_URL" \
  --dataset browsecomp --data-path data/browsecomp/questions.jsonl \
  --save-name local_run --workers-per-endpoint 1
```

The launcher uses `../Evaluation_backend/.venv/bin/hermes` and `.venv/bin/python` by default. Set `HERMES_BIN` or `EVAL_PYTHON` if your install differs. Set `HERMES_HOME` to a private directory containing your Hermes `config.yaml`.

For one vLLM endpoint, optional preemption-aware agent concurrency can be enabled with:

```bash
scripts/run_eval.sh --model YOUR_MODEL --base-url http://127.0.0.1:18080/v1 \
  --dataset browsecomp --data-path data/browsecomp/questions.jsonl \
  --save-name local_run --workers-per-endpoint 16 --workers-per-key 16 \
  --ramp-metrics-url http://127.0.0.1:18080/metrics
```

The default limit starts at 2, increases by 2 after 90 seconds without preemption, and stops at 16. The endpoint and key limits must also allow the desired concurrency. On each increase of the preemption counter, the limit falls by 2 (minimum 1) and further increases are frozen. Running agents finish normally; no new agents start until active work falls below the new limit. Missing metrics prevent starting before the first baseline and pause increases thereafter; a decreasing counter freezes increases. Events are recorded in `output/preds/<save-name>/<dataset>/concurrency.jsonl`. The ramp controls active evaluation runs, not the number of instantaneous model requests.

For a dynamic endpoint registry, use `--endpoint-registry PATH --ramp-registry` in place of `--base-url` and `--ramp-metrics-url`. The ramp follows the evaluator's applied registry snapshot and derives each metrics URL by replacing its trailing `/v1` with `/metrics` (or appending `/metrics` when there is no `/v1`). It pauses dispatch when the snapshot changes or any metrics endpoint fails, establishes a baseline for newly added endpoints, and restarts the stability window. Any monitored endpoint's preemption triggers a backoff. Removing an endpoint uses the evaluator's existing route-retirement behavior, which can cancel its active runs; do not remove routes merely to reduce concurrency.

The optional `HERMES_REMOTE_PURE_COMPUTE=Lab3090` backend executes `execute_code` on the explicitly configured SSH host using the native namespace and seccomp launcher. It requires `code_execution.mode=strict` and filesystem, network and tool access all set to `none`; SSH or sandbox failures return an error without falling back. The remote job has no outbound network and cannot spawn processes or threads, and uses the remote system Python. The SSH host must be trusted and support the native launcher. Run the optional remote integration checks with `GLASSEEK_TEST_REMOTE_SANDBOX=1 python -m unittest discover -s ../tests -p test_remote_pure_compute.py` from this directory.

## AGS sandbox run

Supply `E2B_DOMAIN`, `E2B_API_KEY`, `AGS_TEMPLATE`, `SEARCH_SERVER_ENDPOINT`, `SEARCH_SERVER_API_KEY`, and `AGS_FETCH_SERVER_SOURCE_DIR` in `config/credentials.env`. The Fetch Server source directory must contain `server.py` and `requirements.lock.txt` or `requirements.txt`. The AGS template and Fetch Server are external prerequisites.

```bash
scripts/run_eval_ags.sh \
  --model YOUR_MODEL --base-url "$HERMES_BASE_URL" \
  --dataset browsecomp --data-path data/browsecomp/questions.jsonl \
  --save-name ags_run --workers-per-endpoint 1
```

## Judge

```bash
scripts/judge.sh \
  --save-name local_run --dataset browsecomp \
  --judge-base-url "$JUDGE_BASE_URL" --judge-model YOUR_JUDGE_MODEL
```

Predictions go to `output/preds/<save-name>/<dataset>/conv/`. Judge results and metrics are written alongside them. Use `--help` on each script for all options. `--dry-run` is available for validation without model or judge API calls. Keep `config/credentials.env`, input data, `output/`, and `logs/` out of Git.

### Optional Quest webpage evidence selection

For local experiments, `services/start_quest_retrieval.sh` starts Quest's existing
BM25 process pool and Qwen3-Embedding-8B scoring code, plus a small Glasseek HTTP
adapter. It reads Quest's existing weights and code without modifying either.
The supplied launcher runs 8B BF16 on GPU 1 with a 2,048-token input limit,
maximum 16 model sequences and a 0.80 GPU memory target. GPU 1 must be free;
the launcher does not stop other GPU jobs. The services listen on loopback.
The script does not enable the feature for an evaluation automatically.

To opt a run in, set `GLASSEEK_QUEST_PAGE_RETRIEVAL=1` and
`GLASSEEK_QUEST_RETRIEVAL_URL=http://127.0.0.1:18217` in the evaluation
process environment. The adapter uses the per-task `BROWSE_COMP_QUESTION`
as its retrieval goal, not Quest's page-specific visit goal. Defaults match
Quest's 64 lexical candidates, 16 selected chunks and 8,000-token packing
budget. Failed retrieval is returned as a tool error rather than silently
mixing another extraction method into the evaluation. Set
`GLASSEEK_QUEST_RETRIEVAL_TIMEOUT` to control the client timeout in seconds
(default 180); long CPU queues can exceed it. Configure and benchmark capacity
before using it for high-concurrency evaluations.

### Dataset-specific answer prompts

The evaluator selects final-answer instructions from
`config/answer_prompts.json` using the configured dataset name. Exact names
win over glob patterns; an ambiguous glob match is an error. Each registry
value is a UTF-8 text file path relative to the registry. Unmapped datasets
use `default`. Set `GLASSEEK_ANSWER_PROMPT_REGISTRY` to an alternative JSON
registry to add or replace prompts without editing evaluator code. The Quest
BrowseComp subset uses the `Explanation / Exact Answer / Confidence` template.
These are model instructions rather than a guarantee of format compliance;
existing completed answers are not rewritten.
