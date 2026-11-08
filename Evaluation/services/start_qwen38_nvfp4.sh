#!/usr/bin/env bash
# Local trajectory judge: use checkpoint-native mixed NVFP4 quantization.
set -euo pipefail
quest_root="${QUEST_ROOT:-/root/autodl-tmp/Quest}"
qwen_env="${QWEN_VLLM_ENV:-$quest_root/workspace/env}"
qwen_model="${QWEN_MODEL_PATH:-$quest_root/workspace/models/Qwen3.8-27B-NVFP4}"
gpu="${1:-0}"
port="${2:-19227}"
mode="${3:-}"
[[ $gpu =~ ^[0-9]+$ && $port =~ ^[0-9]+$ && $port -ge 1 && $port -le 65535 ]] || exit 2
[[ -z $mode || $mode == --dry-run ]] || exit 2
[[ -f "$qwen_model/config.json" && -x "$qwen_env/bin/vllm" ]] || { echo 'Missing local checkpoint or vLLM executable' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME="${QWEN_CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$qwen_env/bin:$PATH"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export MAX_JOBS="${MAX_JOBS:-2}"
export VLLM_USE_FLASHINFER_SAMPLER=0
cmd=("$qwen_env/bin/vllm" serve "$qwen_model"
  --host 127.0.0.1 --port "$port" --served-model-name QWEN38-27B-NVFP4
  --tensor-parallel-size 1 --dtype bfloat16 --language-model-only
  --max-model-len "${MAX_MODEL_LEN:-131072}" --max-num-seqs "${MAX_NUM_SEQS:-16}"
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-4096}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}" --kv-cache-dtype fp8
  --attention-backend "${ATTENTION_BACKEND:-TRITON_ATTN}" --enable-prefix-caching --enable-chunked-prefill
  --reasoning-parser qwen3 --enable-prompt-tokens-details
  --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"medium"}')
# User-reviewed defaults: 128K context and CUDA Graph enabled.
# Eager remains an explicit diagnostic override, never the default.
if [[ ${QWEN_ENFORCE_EAGER:-0} == 1 ]]; then
  cmd+=(--enforce-eager)
else
  cmd+=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}')
fi
if [[ $mode == --dry-run ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"; printf '%q ' "${cmd[@]}"; printf '\n'
else
  exec "${cmd[@]}"
fi
