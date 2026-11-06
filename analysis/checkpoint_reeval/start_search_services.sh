#!/usr/bin/env bash
set -euo pipefail
root="${GLASSEEK_ROOT:-/root/autodl-tmp/Glasseek}"
runtime="${GLASSEEK_EVAL_ROOT:-$root/Evaluation}/data/runtime"
credentials="${REEVAL_CREDENTIALS:-$root/checkpoint-reeval/private/eval-credentials.json}"
python="$runtime/venv/bin/python"
export SERPER_KEY_POOL_FILE="$credentials"
export HTTPS_PROXY=http://127.0.0.1:18200 HTTP_PROXY=http://127.0.0.1:18200
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export JINA_API_KEYS="$($python -c 'import json,sys;print(",".join(json.load(open(sys.argv[1]))["jina"]))' "$credentials")"
case "$1" in
 search) cd "$root/Evaluation/services"; exec "$python" -m uvicorn serper_proxy:app --host 127.0.0.1 --port 18083 ;;
 fetch) cd "$runtime/services"; exec "$python" -m uvicorn jina_fetch:app --host 127.0.0.1 --port 18081 ;;
 *) exit 2 ;;
esac
