#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Glasseek
export PYTHONPATH="$PWD/.cache/experiments/pro6000-lora/flash-wheel:$PWD/RL_training${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR=/root/autodl-tmp/pro6000-triton-cache TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/pro6000-torchinductor-cache TMPDIR=/root/autodl-tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
exec /root/autodl-tmp/glasseek-verl091-env/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 experiments/32k-parallel-benchmark/probe.py \
 --config-path="$PWD/RL_training/configs" --config-name=sft model.path=/root/autodl-tmp/models/Qwen3.5-9B \
 "data.train_files=$PWD/data/experiments/sft128k/four_gpu_longest_probe.parquet" data.val_files=null data.train_batch_size=4 data.micro_batch_size_per_gpu=1 \
 data.max_length=131072 data.max_token_len_per_gpu=131072 data.use_dynamic_bsz=false data.num_workers=0 \
 engine.strategy=fsdp2 engine.ulysses_sequence_parallel_size=1 trainer.nnodes=1 trainer.n_gpus_per_node=4 \
 trainer.total_training_steps=1 trainer.total_epochs=1 trainer.save_freq=-1 trainer.test_freq=-1 trainer.resume_mode=disable \
 "trainer.default_local_dir=$PWD/experiments/sft128k/four-gpu-probe-checkpoints" 'trainer.logger=[console]'
