#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/Glasseek/Evaluation/data/runtime
E=/root/autodl-tmp/Glasseek/Evaluation
RUN=glasseek_rl4_quest128_ssh44_c16
DATASET=quest_qwen27b_128
export HERMES_HOME="$R/hermes" EVAL_PYTHON="$R/venv/bin/python" HERMES_BIN="$R/venv/bin/hermes" HERMES_API_KEY=local
export SEARCH_SERVER_ENDPOINT=http://127.0.0.1:18083/search SEARCH_SERVER_API_KEY=rotation
export HERMES_REMOTE_PURE_COMPUTE=Lab3090 GLASSEEK_SEARCH_TOP10=1 GLASSEEK_FETCH_DOH_SAFETY=1
export GLASSEEK_QUEST_PAGE_RETRIEVAL=1 GLASSEEK_QUEST_RETRIEVAL_URL=http://127.0.0.1:18217 GLASSEEK_QUEST_RETRIEVAL_TIMEOUT=180
export GLASSEEK_ALLOW_KV_PREEMPTION=1 GLASSEEK_RESUME_INTERRUPTED=1
cd "$E"
case "${1:-}" in
  full)
    exec scripts/run_eval.sh --model glasseek-rl-4b --endpoint-registry "$R/remote44-rl4-registry.json" \
      --endpoint-registry-refresh-interval 5 --workers-per-endpoint 4 --ramp-registry \
      --ramp-start 4 --ramp-max 4 --ramp-step 2 --ramp-window 90 --ramp-interval 5 \
      --dataset "$DATASET" --data-path "$R/$DATASET.jsonl" --save-name "$RUN" \
      --numbered-output --tool-whitelist web,code_execution,delegation,vision \
      --max-rounds 200 --max-retries 1 --timeout-seconds 3600 --context-length 131072 \
      --fetch-server-url http://127.0.0.1:18081 --fetch-watchdog-target https://example.com
    ;;
  judge)
    export GLASSEEK_JUDGE_TERMINAL_ONLY=1
    export JUDGE_API_KEY="$($EVAL_PYTHON -c 'import json,sys;print(json.load(open(sys.argv[1]))["common"]["JUDGE_OPENAI_API_KEY"])' /root/autodl-tmp/Quest/workspace/strict-reproduction/.secrets/api_config.json)"
    exec scripts/judge.sh --save-name "$RUN" --dataset "$DATASET" --judge-provider openai \
      --judge-base-url https://api.deepseek.com/v1 --judge-model deepseek-flash \
      --prompt-version official --token-param max_tokens --no-json-schema --temperature 0
    ;;
  *) echo 'Usage: run_rl4_c16.sh full|judge' >&2; exit 2 ;;
esac
