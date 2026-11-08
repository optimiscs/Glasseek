#!/usr/bin/env bash
set -euo pipefail
evaluation="${GLASSEEK_EVAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
base_runtime="${GLASSEEK_RUNTIME_DIR:-$evaluation/data/runtime}"
runtime="$base_runtime/retrieval"
quest="${QUEST_ROOT:-$(dirname "$(dirname "$evaluation")")/Quest}"
service="$evaluation/services"
python="${QUEST_PYTHON:-$quest/workspace/env/bin/python}"
mkdir -p "$runtime"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${MKL_NUM_THREADS:-16} TOKENIZERS_PARALLELISM=false
source "$service/tool_services_env.sh"
start() {
  local name=$1
  shift
  if [[ -s "$runtime/$name.pid" ]] && kill -0 "$(cat "$runtime/$name.pid")" 2>/dev/null; then
    echo "$name already running"
    return
  fi
  nohup "$@" >>"$runtime/$name.log" 2>&1 </dev/null &
  echo "$!" >"$runtime/$name.pid"
  echo "$name started pid=$!"
}
start bm25 "$python" "$service/tool_services.py" serve bm25
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.80} MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048} MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
export HF_HOME=$base_runtime/cache/huggingface
export VLLM_CACHE_ROOT=$base_runtime/cache/vllm-embedding-gpu1
export TRITON_CACHE_DIR=$base_runtime/cache/triton-embedding-gpu1
embedding_port="$("$python" -c 'import os; from urllib.parse import urlsplit; u=urlsplit(os.environ["GLASSEEK_EMBEDDING_UPSTREAM"]); assert u.hostname == "127.0.0.1" and u.port; print(u.port)')"
start embedding_gpu "$quest/inference/scripts/serve_single_gpu.sh" embedding 1 "$embedding_port"
start embedding_adapter "$python" "$service/tool_services.py" serve embedding
start bridge "$python" "$service/tool_services.py" serve bridge
