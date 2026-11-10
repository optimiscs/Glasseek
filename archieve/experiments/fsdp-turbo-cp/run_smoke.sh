#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Glasseek
export PYTHONPATH="/root/autodl-tmp/FSDPTurbo:$PWD/RL_training${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/autodl-tmp/glasseek-verl091-env/bin:$PATH"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
export TRITON_CACHE_DIR=/root/autodl-tmp/glasseek-triton-cache TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/glasseek-torchinductor-cache TMPDIR=/root/autodl-tmp
exec python -m torch.distributed.run --standalone --nproc_per_node=4 experiments/lora-sp4-probe/run_probe.py \
 --config-path="$PWD/RL_training/configs" --config-name=sft \
 model.path=/root/autodl-tmp/models/Qwen3.5-9B model.lora_rank=16 model.lora_alpha=32 model.target_modules=all-linear \
 model.use_fused_kernels=true model.fused_kernel_options.impl_backend=torch \
 engine.strategy=fsdp_turbo engine.model_dtype=bf16 engine.ulysses_sequence_parallel_size=1 \
 engine.turbo_config.distributed.fully_shard_parallel_size=4 engine.turbo_config.distributed.ulysses_parallel_size=2 \
 '+engine.turbo_config.distributed.fsdp_plan.apply_modules={model.visual.blocks.\{*\}:{},model.visual.patch_embed:{},model.visual.pos_embed:{},model.language_model.embed_tokens:{},model.language_model.layers.\{*\}:{},lm_head:{}}' \
 "+engine.turbo_config.distributed.fsdp_plan.hook_modules=['model.language_model.layers.{*}']" \
 "+engine.turbo_config.distributed.cp_plan.ulysses_function_patches=[{target_functions:['transformers.models.qwen3_5.modeling_qwen3_5.eager_attention_forward'],type:full_attention},{target_functions:['transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward'],type:gated_delta_net}]" \
 "+engine.turbo_config.distributed.cp_plan.loss_function_patches=[{target_functions:['transformers.loss.loss_utils.ForCausalLMLoss'],type:causal_lm_loss}]" \
 "+engine.turbo_config.module_patches=[{target:transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Model.forward,replacement:fsdp_turbo.models.qwen.qwen3_5.qwen3_5_model_forward}]" \
 "data.train_files=$PWD/data/experiments/32k-parallel-benchmark/data-8.parquet" data.val_files=null \
 data.train_batch_size=4 data.micro_batch_size_per_gpu=1 data.use_dynamic_bsz=false data.max_length=32768 data.max_token_len_per_gpu=32768 data.num_workers=0 \
 trainer.nnodes=1 trainer.n_gpus_per_node=4 trainer.total_epochs=1 trainer.total_training_steps=1 trainer.save_freq=-1 trainer.test_freq=-1 trainer.resume_mode=disable \
 "trainer.default_local_dir=$PWD/experiments/fsdp-turbo-cp/checkpoints" 'trainer.logger=[console]'
