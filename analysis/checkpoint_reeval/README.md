# RL9B trajectory checkpoint comparison

`rl9b_judged_complete.jsonl` contains judged, successful (not timed out)
inference sessions, with labels `success` (correct answer) and `failure`
(incorrect answer). It excludes infrastructure failures and unjudged cases.
Session messages, original question, judge reference and prediction SHA-256
are retained. This is a selected subset, not a new benchmark score.

`adapters/step_*/` contain safetensors converted from single-rank verl LoRA
checkpoints. Load each on the Qwen3.5-9B base, not on the RL9B model. All
source and destination tensor values were checked before copying. Verify the
base model weights, model-level outputs, long-context rendering and tool-message
masking on the evaluation host before interpreting scores.

The trajectory scoring program is not yet implemented. Do not interpret the
25/9 source labels as model performance for the SFT checkpoints: these labels
describe the original RL9B answers. In particular, log-likelihood on a fixed
trace and rerunning the search agent answer different questions.

## Restart the baseline on a migrated host

The 34 questions live in `browsecomp_rl9b_non_timeout_34.jsonl`. The eight
entries in `base-registry.json` preserve the original 32-request global
concurrency and use one model replica per GPU. `run_base_eval.sh` runs the
original evaluation and `supervise_base_eval.py` runs preflight and then the
official judge.

Before running `start_portable.sh`, install Quest's vLLM environment and the
Evaluation Hermes environment; provide the Qwen3.5-9B base model; and start the
Quest BM25/embedding adapter/bridge on ports 18215/18213/18217, respectively.
Port 18218 must be forwarded to a running embedding server on host 44. Configure
the `Lab3090` SSH alias and a working code-execution tunnel, and provide an HTTP
proxy at 127.0.0.1:18200. The launcher checks these prerequisites; it does not
create the remote-host tunnels or install models and dependencies.

Store credentials outside the repository in a mode-600 JSON file containing
`serper` and `jina` arrays and a `judge` string. Set environment variables:

```bash
export GLASSEEK_ROOT=/path/to/Glasseek
export GLASSEEK_EVAL_ROOT="$GLASSEEK_ROOT/Evaluation"
export BASE_MODEL_PATH=/path/to/Qwen3.5-9B
export VLLM_BIN=/path/to/Quest/workspace/env/bin/vllm
export REEVAL_CREDENTIALS=/private/path/eval-credentials.json
export HERMES_REMOTE_PURE_COMPUTE=Lab3090
bash "$GLASSEEK_ROOT/analysis/checkpoint_reeval/start_portable.sh"
```

`REEVAL_RUN_NAME` selects a new output namespace; use a new name for an
independent rerun. The starter refuses to start a second supervisor. The
supervisor will wait at most 30 minutes for all model and tool health checks,
tests actual search/fetch requests, and checks for at least 1 GiB free.
The model cache defaults to `/tmp`; ensure that partition has enough space.

`sync_remote_results.sh` runs on the source machine and copies the 8-GPU host's
new evaluation output and code over SSH. SQLite session data are copied using
SQLite's online backup API, not by blindly copying a live database. Set
`REEVAL_SSH_HOST`, `REEVAL_SSH_PORT`, `REEVAL_SSH_KEY` and
`REEVAL_REMOTE_ROOT` for a different host. `--once` performs one snapshot;
without it, the script repeats every 60 seconds until stopped. It does not copy
private credentials. Copies of logs and predictions may contain user data.
