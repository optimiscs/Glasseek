#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Glasseek
export PYTHONPATH="$PWD/RL_training${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/autodl-tmp/glasseek-verl091-env/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TRITON_CACHE_DIR=/root/autodl-tmp/glasseek-triton-cache
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/glasseek-torchinductor-cache
export TMPDIR=/root/autodl-tmp
line_id=$1
exec /root/autodl-tmp/glasseek-verl091-env/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  experiments/lora-sp4-probe/run_probe.py \
  --config-path=/root/autodl-tmp/Glasseek/RL_training/configs --config-name=sft \
  "model.path=/root/autodl-tmp/models/Qwen3.5-9B" model.lora_rank=16 model.lora_alpha=32 \
  model.target_modules=all-linear model.use_fused_kernels=true model.fused_kernel_options.impl_backend=torch \
  engine.strategy=fsdp2 engine.model_dtype=bf16 engine.ulysses_sequence_parallel_size=2 \
  "data.train_files=$PWD/experiments/sp4-30g/sp2-line-${line_id}.parquet" data.val_files=null \
  data.train_batch_size=2 data.micro_batch_size_per_gpu=1 data.use_dynamic_bsz=false \
  data.max_length=262144 data.max_token_len_per_gpu=65536 data.num_workers=0 \
  trainer.nnodes=1 trainer.n_gpus_per_node=4 trainer.total_epochs=1 trainer.total_training_steps=1 \
  trainer.save_freq=-1 trainer.test_freq=-1 trainer.resume_mode=disable \
  "trainer.default_local_dir=$PWD/experiments/sp4-30g/checkpoints" 'trainer.logger=[console]'
