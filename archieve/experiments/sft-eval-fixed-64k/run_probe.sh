#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Glasseek
export PYTHONPATH="$PWD/RL_training${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR=/root/autodl-tmp/glasseek-fsdp4-triton TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/glasseek-fsdp4-inductor TMPDIR=/root/autodl-tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec /root/autodl-tmp/glasseek-verl091-env/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 experiments/sft-eval-fixed-64k/probe.py \
 --config-path="$PWD/RL_training/configs" --config-name=sft_64k_full_fsdp2 \
 model.path=/root/autodl-fs/models/Qwen3.5-9B model.lora_rank=0 \
 "data.train_files=$PWD/data/glasseek/sft-eval-fixed-64k/${PROBE_DATA:-probe-longest-4.parquet}" data.val_files=null \
 data.train_batch_size=4 data.micro_batch_size_per_gpu=1 data.max_length=65536 data.max_token_len_per_gpu=65536 data.use_dynamic_bsz=false data.num_workers=0 \
 engine.strategy=fsdp2 engine.ulysses_sequence_parallel_size="${SP_SIZE:-1}" engine.use_no_sync_for_gradient_accumulation=false \
 trainer.nnodes=1 trainer.n_gpus_per_node=4 trainer.total_training_steps=1 trainer.total_epochs=1 trainer.save_freq=-1 trainer.test_freq=-1 trainer.resume_mode=disable \
 "trainer.default_local_dir=$PWD/experiments/sft-eval-fixed-64k/probe-checkpoints" 'trainer.logger=[console]'
