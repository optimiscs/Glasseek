#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Glasseek
export PYTHONPATH="$PWD/analysis/pro6000-lora/flash-wheel:$PWD/RL_training${PYTHONPATH:+:$PYTHONPATH}"
export TENSORBOARD_DIR="$PWD/analysis/sft128k/tensorboard_comparison/clean2899_full64k_2gpu"
export TRITON_CACHE_DIR=/root/autodl-tmp/pro6000-triton-cache TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/pro6000-torchinductor-cache TMPDIR=/root/autodl-tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1
exec /root/autodl-tmp/glasseek-verl091-env/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 -m glasseek_training.turn_curriculum_sft \
 --config-path="$PWD/RL_training/configs" --config-name=sft_64k_full_fsdp2 \
 engine.use_no_sync_for_gradient_accumulation=false \
 "data.train_files=$PWD/data/glasseek/sft-clean2899-fixed/train.parquet" \
 "data.val_files=$PWD/data/glasseek/sft-clean2899-fixed/val.parquet" \
 trainer.experiment_name=sft_clean2899_64k_full_2gpu_turns \
 "trainer.default_local_dir=$PWD/checkpoints/glasseek/sft_clean2899_64k_full_2gpu_turns" "$@"
