#!/usr/bin/env bash
set -euo pipefail
E="${GLASSEEK_EVAL_ROOT:-/root/autodl-tmp/Glasseek/Evaluation}"
R="$E/data/runtime"
P="${REEVAL_DIR:-/root/autodl-tmp/Glasseek/checkpoint-reeval/analysis/checkpoint_reeval}"
export HERMES_HOME="$R/hermes" EVAL_PYTHON="$R/venv/bin/python" HERMES_BIN="$R/venv/bin/hermes" HERMES_API_KEY=local
export SEARCH_SERVER_ENDPOINT=http://127.0.0.1:18083/search SEARCH_SERVER_API_KEY=rotation
export HERMES_REMOTE_PURE_COMPUTE="${HERMES_REMOTE_PURE_COMPUTE:-Lab3090}" GLASSEEK_SEARCH_TOP10=1 GLASSEEK_FETCH_DOH_SAFETY=1
export GLASSEEK_ALLOW_KV_PREEMPTION=1 GLASSEEK_RESUME_INTERRUPTED=1
export GLASSEEK_QUEST_PAGE_RETRIEVAL=1 GLASSEEK_QUEST_RETRIEVAL_URL=http://127.0.0.1:18217 GLASSEEK_QUEST_RETRIEVAL_TIMEOUT=180
cd "$E"
args=(--model glasseek-base-9b --endpoint-registry "$P/base-registry.json" --endpoint-registry-refresh-interval 5
 --workers-per-endpoint 8 --ramp-registry --ramp-start 32 --ramp-max 32 --ramp-step 2 --ramp-window 90 --ramp-interval 5
 --dataset browsecomp_rl9b_non_timeout_34 --data-path "$P/browsecomp_rl9b_non_timeout_34.jsonl"
 --save-name "${REEVAL_RUN_NAME:-glasseek_base9b_rl9b_subset}" --numbered-output --tool-whitelist web,code_execution,delegation,vision
 --enable-custom-system-prompt true --custom-system-prompt-file "$P/fixed_system_prompt.txt"
 --max-rounds 200 --max-retries 1 --timeout-seconds 3600 --context-length 131072
 --fetch-server-url http://127.0.0.1:18081 --fetch-watchdog-target https://example.com)
if [[ "${1:-}" == smoke ]]; then args+=(--limit 1); fi
exec scripts/run_eval.sh "${args[@]}"
