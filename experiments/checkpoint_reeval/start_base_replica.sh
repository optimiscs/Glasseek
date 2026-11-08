#!/usr/bin/env bash
set -euo pipefail
gpu="$1"; port="$2"
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4
export PATH="${NINJA_BIN_DIR:-/root/miniconda3/envs/ms-swift/bin}:$PATH"
export HF_HOME="${EVAL_HF_HOME:-/root/autodl-tmp/Glasseek/Evaluation/data/runtime/cache/huggingface}"
export VLLM_CACHE_ROOT="/tmp/glasseek-base-vllm-$gpu" TRITON_CACHE_DIR="/tmp/glasseek-base-triton-$gpu"
export TORCHINDUCTOR_CACHE_DIR="/tmp/glasseek-base-inductor-$gpu"
exec "${VLLM_BIN:-/root/autodl-tmp/Quest/workspace/env/bin/vllm}" serve "${BASE_MODEL_PATH:-/root/autodl-fs/models/Qwen3.5-9B}" \
 --served-model-name glasseek-base-9b --host 127.0.0.1 --port "$port" --tensor-parallel-size 1 \
 --max-model-len 131072 --gpu-memory-utilization 0.92 --kv-cache-dtype fp8 --max-num-seqs 16 \
 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --enable-prompt-tokens-details
