#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?}; port=${2:?}
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 PATH=/root/miniconda3/envs/ms-swift/bin:$PATH CUDA_HOME=/usr/local/cuda
export VLLM_CACHE_ROOT="/root/autodl-tmp/Glasseek/analysis/clean2899-full64k/vllm-cache-$gpu" TRITON_CACHE_DIR="/root/autodl-tmp/Glasseek/analysis/clean2899-full64k/triton-cache-$gpu"
exec /root/autodl-tmp/Quest/workspace/env/bin/vllm serve /root/autodl-tmp/Glasseek/analysis/clean2899-full64k/step39-hf --served-model-name clean2899-step39 --host 127.0.0.1 --port "$port" --tensor-parallel-size 1 --max-model-len 131072 --gpu-memory-utilization 0.92 --kv-cache-dtype fp8 --max-num-seqs "${EVAL_PER_GPU_CONCURRENCY:-32}" --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --enable-prompt-tokens-details --default-chat-template-kwargs '{"enable_thinking":true}'
