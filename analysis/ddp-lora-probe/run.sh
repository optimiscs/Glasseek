#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Glasseek
export PYTHONPATH="$PWD/RL_training${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR=/root/autodl-tmp/glasseek-triton-cache
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/glasseek-torchinductor-cache
export TMPDIR=/root/autodl-tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
exec /root/autodl-tmp/glasseek-verl091-env/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 analysis/ddp-lora-probe/probe.py "$1"
