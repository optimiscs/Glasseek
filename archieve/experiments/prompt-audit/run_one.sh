#!/usr/bin/env bash
set -uo pipefail
E=/root/autodl-tmp/Glasseek/Evaluation
P=/root/autodl-tmp/Glasseek/checkpoint-reeval/analysis/checkpoint_reeval
A=/root/autodl-tmp/Glasseek/checkpoint-reeval/analysis/prompt-audit
R="$E/data/runtime"
export HERMES_HOME="$R/hermes" EVAL_PYTHON="$R/venv/bin/python" HERMES_BIN="$R/venv/bin/hermes" HERMES_API_KEY=local
export SEARCH_SERVER_ENDPOINT=http://127.0.0.1:18083/search SEARCH_SERVER_API_KEY=rotation
export HERMES_REMOTE_PURE_COMPUTE=Lab3090 GLASSEEK_SEARCH_TOP10=1 GLASSEEK_FETCH_DOH_SAFETY=1 GLASSEEK_ALLOW_KV_PREEMPTION=1 GLASSEEK_RESUME_INTERRUPTED=1 NO_TOTAL_TIMEOUT=1
export GLASSEEK_QUEST_PAGE_RETRIEVAL=1 GLASSEEK_QUEST_RETRIEVAL_URL=http://127.0.0.1:18217 GLASSEEK_QUEST_RETRIEVAL_TIMEOUT=180
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
cd "$E"
bash scripts/run_eval.sh --model glasseek-base-9b --base-url http://127.0.0.1:18101/v1 --workers-per-endpoint 1 --dataset browsecomp_rl9b_non_timeout_34 --data-path "$P/browsecomp_rl9b_non_timeout_34.jsonl" --save-name glasseek_base9b_prompt_audit_20261105 --numbered-output --tool-whitelist web,code_execution,delegation,vision --enable-custom-system-prompt true --custom-system-prompt-file "$P/fixed_system_prompt.txt" --max-rounds 200 --max-retries 1 --context-length 131072 --fetch-server-url http://127.0.0.1:18081 --fetch-watchdog-target https://example.com --limit 1
status=$?
echo "evaluation_exit=$status"
if ((status != 0));then exit "$status";fi
export HTTPS_PROXY=http://127.0.0.1:18200 HTTP_PROXY=http://127.0.0.1:18200
export JUDGE_API_KEY="$($EVAL_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["judge"])' /root/autodl-tmp/Glasseek/checkpoint-reeval/private/eval-credentials.json)"
bash scripts/judge.sh --save-name glasseek_base9b_prompt_audit_20261105 --dataset browsecomp_rl9b_non_timeout_34 --judge-provider openai --judge-base-url https://api.deepseek.com/v1 --judge-model deepseek-flash --prompt-version official --token-param max_tokens --no-json-schema --temperature 0 --allow-partial
status=$?
echo "judge_exit=$status"
exit "$status"
