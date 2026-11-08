P=/root/autodl-tmp/Glasseek
E="$P/Evaluation"
D="$P/experiments/clean2899-full64k"
export HERMES_HOME="$E/data/runtime/hermes" EVAL_PYTHON="$E/data/runtime/venv/bin/python" HERMES_BIN="$E/data/runtime/venv/bin/hermes" HERMES_API_KEY=local
source "$E/services/tool_services_env.sh"
export HERMES_REMOTE_PURE_COMPUTE=Lab3090
export GLASSEEK_SEARCH_TOP10=1 GLASSEEK_FETCH_DOH_SAFETY=1 GLASSEEK_ALLOW_KV_PREEMPTION=1 GLASSEEK_RESUME_INTERRUPTED=1 NO_TOTAL_TIMEOUT=1
export REEVAL_CREDENTIALS=/root/autodl-tmp/migration-24704/Glasseek/checkpoint-reeval/private/eval-credentials.json
export GLASSEEK_VIEWER_RUN="${GLASSEEK_VIEWER_RUN:-glasseek_clean2899_full64k_step39_128}"
export GLASSEEK_VIEWER_DATASET=quest_qwen27b_128
export GLASSEEK_VIEWER_WORKSPACE="${GLASSEEK_VIEWER_WORKSPACE:-glasseek-clean2899-full64k-step39-128}"
export GLASSEEK_VIEWER_MODEL=clean2899-step39
export GLASSEEK_VIEWER_LABEL="${GLASSEEK_VIEWER_LABEL:-Qwen3.5-9B 清洗去重全参数 step39 · BrowseComp 128}"
export GLASSEEK_MAX_TOOL_CALLS=120
