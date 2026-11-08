#!/usr/bin/env bash
set -euo pipefail
step="${1:?checkpoint step required}"
case "$step" in 10|20|30|40|50|60|70|80) ;; *) exit 2;; esac
E="${GLASSEEK_EVAL_ROOT:-/root/autodl-tmp/Glasseek/Evaluation}"
R="$E/data/runtime"
P="${REEVAL_DIR:-/root/autodl-tmp/Glasseek/checkpoint-reeval/analysis/checkpoint_reeval}"
export HERMES_HOME="$R/hermes" EVAL_PYTHON="$R/venv/bin/python" HERMES_BIN="$R/venv/bin/hermes" HERMES_API_KEY=local
source "$E/services/search_pool_env.sh"
export HERMES_REMOTE_PURE_COMPUTE="${HERMES_REMOTE_PURE_COMPUTE:-Lab3090}" GLASSEEK_SEARCH_TOP10=1 GLASSEEK_FETCH_DOH_SAFETY=1
export GLASSEEK_ALLOW_KV_PREEMPTION=1 GLASSEEK_RESUME_INTERRUPTED=1 NO_TOTAL_TIMEOUT=1
export GLASSEEK_QUEST_PAGE_RETRIEVAL=1 GLASSEEK_QUEST_RETRIEVAL_URL=http://127.0.0.1:18217 GLASSEEK_QUEST_RETRIEVAL_TIMEOUT=180
cd "$E"
exec scripts/run_eval.sh --model "step$step" --endpoint-registry "$P/step${step}-registry.json" --endpoint-registry-refresh-interval 5 \
 --workers-per-endpoint 8 --dataset browsecomp_rl9b_non_timeout_34 \
 --data-path "$P/browsecomp_rl9b_non_timeout_34.jsonl" --save-name "glasseek_step${step}_rl9b_subset" \
 --numbered-output --tool-whitelist web,code_execution,delegation,vision --enable-custom-system-prompt true --custom-system-prompt-file "$P/fixed_system_prompt.txt" --max-rounds 200 --max-retries 1 \
 --context-length 131072 --fetch-server-url http://127.0.0.1:18081 --fetch-watchdog-target https://example.com
