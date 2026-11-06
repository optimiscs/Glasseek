#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Glasseek
export PYTHONPATH="$PWD/analysis/pro6000-lora/flash-wheel:$PWD/RL_training${PYTHONPATH:+:$PYTHONPATH}"
export SFT_LENGTH_AUDIT="$PWD/data/glasseek/sft-128k/selection.jsonl"
export TENSORBOARD_DIR="$PWD/analysis/sft128k/tensorboard/length_curriculum"
export TRITON_CACHE_DIR=/root/autodl-tmp/pro6000-triton-cache TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/pro6000-torchinductor-cache TMPDIR=/root/autodl-tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
exec /root/autodl-tmp/glasseek-verl091-env/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 -m "${SFT_TRAIN_MODULE:-glasseek_training.curriculum_sft}" \
 --config-path="$PWD/RL_training/configs" --config-name=sft \
 model.path=/root/autodl-tmp/models/Qwen3.5-9B \
 "data.train_files=$PWD/data/glasseek/sft-128k/train.parquet" "data.val_files=$PWD/data/glasseek/sft-128k/val.parquet" \
 data.train_batch_size=64 data.micro_batch_size_per_gpu=1 data.max_length=131072 data.max_token_len_per_gpu=131072 data.use_dynamic_bsz=false \
 engine.strategy=fsdp2 engine.ulysses_sequence_parallel_size=1 trainer.nnodes=1 trainer.n_gpus_per_node=1 \
 trainer.experiment_name=sft_length_curriculum trainer.total_epochs=3 trainer.save_freq=10 trainer.test_freq=after_each_epoch trainer.resume_mode=auto \
 "trainer.default_local_dir=$PWD/checkpoints/glasseek/sft_128k_length_curriculum_20261104" \
 '+checkpoint.save_lora_only=true' 'trainer.logger=[console,tensorboard]' "$@"
