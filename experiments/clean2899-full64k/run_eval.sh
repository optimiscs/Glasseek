#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/Glasseek/experiments/clean2899-full64k/eval_env.sh
cd "$E"
exec bash scripts/run_eval.sh --model clean2899-step39 --base-urls http://127.0.0.1:19139/v1,http://127.0.0.1:19239/v1 --workers-per-endpoint "${EVAL_PER_GPU_CONCURRENCY:-32}" --dataset quest_qwen27b_128 --data-path "$D/browsecomp128.jsonl" --expected-questions 128 --save-name "$GLASSEEK_VIEWER_RUN" --numbered-output --tool-whitelist web,code_execution,delegation,vision --enable-custom-system-prompt true --custom-system-prompt-file "$P/data/glasseek/sft-eval-fixed-128k/eval_system_body.txt" --max-rounds 200 --max-retries 1 --max-tokens 16384 --context-length 131072 --fetch-server-url http://127.0.0.1:18081 --fetch-watchdog-target https://example.com "$@"
