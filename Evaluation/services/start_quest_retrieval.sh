#!/usr/bin/env bash
set -euo pipefail
runtime=/root/autodl-tmp/Glasseek/Evaluation/data/runtime/retrieval
quest=/root/autodl-tmp/Quest
service=/root/autodl-tmp/Glasseek/Evaluation/services
python="$quest/workspace/env/bin/python"
mkdir -p "$runtime"
export VISIT_BM25_POOL_URL=http://127.0.0.1:18215 VISIT_EMBEDDING_URL=http://127.0.0.1:18213
export VISIT_RETRIEVAL_CANDIDATES=${VISIT_RETRIEVAL_CANDIDATES:-64} VISIT_RETRIEVAL_TOP_K=${VISIT_RETRIEVAL_TOP_K:-16}
export GLASSEEK_RETRIEVAL_CONCURRENCY=${GLASSEEK_RETRIEVAL_CONCURRENCY:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${MKL_NUM_THREADS:-16} TOKENIZERS_PARALLELISM=false
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
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
start bm25 "$python" "$quest/inference/bm25_pool_service.py" --workers 4 --queue-limit 16 --port 18215
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.80} MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048} MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
export HF_HOME=/root/autodl-tmp/Glasseek/Evaluation/data/runtime/cache/huggingface
export VLLM_CACHE_ROOT=/root/autodl-tmp/Glasseek/Evaluation/data/runtime/cache/vllm-embedding-gpu1
export TRITON_CACHE_DIR=/root/autodl-tmp/Glasseek/Evaluation/data/runtime/cache/triton-embedding-gpu1
start embedding_gpu "$quest/inference/scripts/serve_single_gpu.sh" embedding 1 18218
start embedding_adapter "$python" "$quest/inference/retrieval_vllm_adapter.py" --role embedding --model QWEN3-EMBEDDING-8B --upstream http://127.0.0.1:18218 --port 18213
(cd "$service" && start bridge "$python" -m uvicorn quest_retrieval_bridge:app --host 127.0.0.1 --port 18217)
