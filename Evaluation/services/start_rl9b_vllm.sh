#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && "$1" =~ ^[0-9]+$ && "$2" =~ ^[0-9]+$ ]] || { echo 'Usage: start_vllm_replica.sh GPU_INDEX PORT' >&2; exit 2; }
R=/root/autodl-tmp/Glasseek/Evaluation/data/runtime
GPU_INDEX="$1"
PORT="$2"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PATH="$R/venv/bin:$PATH" HF_HOME="$R/cache/huggingface"
export VLLM_CACHE_ROOT="$R/cache/vllm-rl9b-gpu$GPU_INDEX" TRITON_CACHE_DIR="$R/cache/triton-gpu$GPU_INDEX"
export TORCHINDUCTOR_CACHE_DIR="$R/cache/torchinductor-gpu$GPU_INDEX" OMP_NUM_THREADS=4
exec /root/autodl-tmp/Quest/workspace/env/bin/vllm serve "$R/models/Glasseeker-9B-RL" \
  --served-model-name glasseek-rl-9b --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 --max-model-len 131072 --gpu-memory-utilization 0.92 \
  --kv-cache-dtype fp8 --max-num-seqs 16 --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --enable-prompt-tokens-details
