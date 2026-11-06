#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && "$1" =~ ^[0-9]+$ && "$2" =~ ^[0-9]+$ ]] || { echo 'Usage: start_vllm_replica.sh GPU_INDEX PORT' >&2; exit 2; }
E="${GLASSEEK_EVAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${GLASSEEK_RUNTIME_DIR:-$E/data/runtime}"
QUEST_ROOT="${QUEST_ROOT:-$(dirname "$(dirname "$E")")/Quest}"
VLLM_BIN="${VLLM_BIN:-$QUEST_ROOT/workspace/env/bin/vllm}"
GPU_INDEX="$1"
PORT="$2"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PATH="$R/venv/bin:$PATH" HF_HOME="$R/cache/huggingface"
export VLLM_CACHE_ROOT="$R/cache/vllm-rl9b-gpu$GPU_INDEX" TRITON_CACHE_DIR="$R/cache/triton-gpu$GPU_INDEX"
export TORCHINDUCTOR_CACHE_DIR="$R/cache/torchinductor-gpu$GPU_INDEX" OMP_NUM_THREADS=4
exec "$VLLM_BIN" serve "$R/models/Glasseeker-9B-RL" \
  --served-model-name glasseek-rl-9b --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 --max-model-len 131072 --gpu-memory-utilization 0.92 \
  --kv-cache-dtype fp8 --max-num-seqs 16 --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --enable-prompt-tokens-details
