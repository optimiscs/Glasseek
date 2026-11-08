#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Glasseek
export PYTHONPATH="$PWD/.cache/experiments/pro6000-lora/flash-wheel:$PWD/RL_training${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR=/root/autodl-tmp/pro6000-triton-cache TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/pro6000-torchinductor-cache TMPDIR=/root/autodl-tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
exec /root/autodl-tmp/glasseek-verl091-env/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 -m glasseek_training.lora_probe --profile offload "$@"
