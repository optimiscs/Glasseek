#!/usr/bin/env bash
set -euo pipefail
gpu="$1";port="$2"
export PATH="/root/miniconda3/envs/ms-swift/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT="/tmp/glasseek-no-tool-vllm-$gpu" TRITON_CACHE_DIR="/tmp/glasseek-no-tool-triton-$gpu" TORCHINDUCTOR_CACHE_DIR="/tmp/glasseek-no-tool-inductor-$gpu"
exec /root/autodl-tmp/Quest/workspace/env/bin/vllm serve /root/autodl-fs/models/Qwen3.5-9B --served-model-name glasseek-base-9b --host 127.0.0.1 --port "$port" --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.86 --kv-cache-dtype fp8 --max-num-seqs 8 --max-num-batched-tokens 8192 --reasoning-parser qwen3
