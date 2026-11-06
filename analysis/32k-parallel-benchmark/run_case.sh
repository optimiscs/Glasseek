#!/usr/bin/env bash
set -uo pipefail
cd /root/autodl-tmp/Glasseek
export PYTHONPATH="$PWD/RL_training${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/autodl-tmp/glasseek-verl091-env/bin:$PATH"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
export TRITON_CACHE_DIR=/root/autodl-tmp/glasseek-triton-cache TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/glasseek-torchinductor-cache TMPDIR=/root/autodl-tmp
case "$1" in A) sp=1; reshard=true;; B) sp=1; reshard=false;; C) sp=2; reshard=true;; D) sp=4; reshard=true;; *) exit 2;; esac
batch=$2
n=$((batch*2))
if [ "$batch" = 64 ]; then data_file=representative-128.parquet; else data_file=data-${n}.parquet; fi
exec timeout --signal=TERM --kill-after=60s 7200s python -m torch.distributed.run --standalone --nproc_per_node=4 analysis/32k-parallel-benchmark/probe.py \
 --config-path="$PWD/RL_training/configs" --config-name=sft \
 model.path=/root/autodl-tmp/models/Qwen3.5-9B model.lora_rank=16 model.lora_alpha=32 \
 model.target_modules=all-linear model.use_fused_kernels=true model.fused_kernel_options.impl_backend=torch \
 engine.strategy=fsdp2 engine.model_dtype=bf16 "engine.ulysses_sequence_parallel_size=$sp" "engine.reshard_after_forward=$reshard" \
 "data.train_files=$PWD/analysis/32k-parallel-benchmark/${data_file}" data.val_files=null \
 "data.train_batch_size=$batch" data.micro_batch_size_per_gpu=1 data.use_dynamic_bsz=false \
 data.max_length=32768 data.max_token_len_per_gpu=32768 data.num_workers=0 \
 trainer.nnodes=1 trainer.n_gpus_per_node=4 trainer.total_epochs=1 "trainer.total_training_steps=${BENCH_STEPS:-2}" \
 trainer.save_freq=-1 trainer.test_freq=-1 trainer.resume_mode=disable \
 "trainer.default_local_dir=$PWD/analysis/32k-parallel-benchmark/checkpoints-$1-$batch" 'trainer.logger=[console]'
