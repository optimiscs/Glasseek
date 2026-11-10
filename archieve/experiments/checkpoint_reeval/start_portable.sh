#!/usr/bin/env bash
# Start from the destination host; external SSH/embedding tunnels must be ready.
set -euo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REEVAL_DIR="$P"
export GLASSEEK_ROOT="${GLASSEEK_ROOT:-$(cd "$P/../.." && pwd)}"
export GLASSEEK_EVAL_ROOT="${GLASSEEK_EVAL_ROOT:-$GLASSEEK_ROOT/Evaluation}"
export REEVAL_CREDENTIALS="${REEVAL_CREDENTIALS:?Set REEVAL_CREDENTIALS to the private credential JSON}"
export VLLM_BIN="${VLLM_BIN:?Set VLLM_BIN to the Quest vllm executable}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to Qwen3.5-9B}"
export HERMES_REMOTE_PURE_COMPUTE="${HERMES_REMOTE_PURE_COMPUTE:-Lab3090}"
test -r "$REEVAL_CREDENTIALS"
test -x "$VLLM_BIN"
test -f "$BASE_MODEL_PATH/config.json"
test -x "$GLASSEEK_EVAL_ROOT/data/runtime/venv/bin/hermes"
test "$(nvidia-smi -L | wc -l)" -ge 8
ssh -o BatchMode=yes -o ConnectTimeout=5 "$HERMES_REMOTE_PURE_COMPUTE" true
curl -fsS --max-time 5 http://127.0.0.1:18218/health >/dev/null || { echo '44 embedding tunnel on 18218 unavailable' >&2; exit 1; }
curl -fsS --max-time 5 http://127.0.0.1:18217/health >/dev/null || { echo 'Quest retrieval bridge on 18217 unavailable' >&2; exit 1; }
curl -fsS --max-time 5 -x http://127.0.0.1:18200 https://example.com -o /dev/null || { echo 'HTTPS proxy on 18200 unavailable' >&2; exit 1; }
if screen -ls | grep -q 'glasseek-base-supervisor'; then echo 'Baseline supervisor already exists' >&2; exit 1; fi
for service in search fetch; do
  if ! screen -ls | grep -q "glasseek-$service"; then
    screen -dmS "glasseek-$service" bash -c 'exec "${PYTHON_BIN:-python3}" "$GLASSEEK_EVAL_ROOT/services/tool_services.py" serve "$2" >>"$1/$2.log" 2>&1' _ "$P" "$service"
  fi
done
for pair in 0:18080 1:18087 2:18082 3:18085 4:18086 5:18088 6:18089 7:18090; do
  gpu=${pair%:*}; port=${pair#*:}
  if ! screen -ls | grep -q "glasseek-base-$gpu"; then
    screen -dmS "glasseek-base-$gpu" bash -c 'exec bash "$1/start_base_replica.sh" "$2" "$3" >>"$1/base-$2.log" 2>&1' _ "$P" "$gpu" "$port"
  fi
done
screen -dmS glasseek-base-supervisor bash -c 'exec "${PYTHON_BIN:-python3}" -u "$1/supervise_base_eval.py" >>"$1/supervisor.log" 2>&1' _ "$P"
echo "Supervisor started; inspect $P/supervisor.log"
