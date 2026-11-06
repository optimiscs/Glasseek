#!/usr/bin/env bash
set -euo pipefail
root="${GLASSEEK_REMOTE_ROOT:-/home/moxu/glasseek-rl4-eval}"
python="${GLASSEEK_REMOTE_PYTHON:-/home/moxu/intentops-qwen35-20261017/venv/bin/python}"
log(){ echo "$(date -Is) $*" | tee -a "$root/logs/supervisor.log"; }
trap 'log "error at line $LINENO"' ERR
log 'waiting for verified RL-4B download'
while [[ ! -f "$root/download.success" ]]; do
 if ! kill -0 "$(cat "$root/download.pid")" 2>/dev/null; then log 'download stopped before verification'; exit 1; fi
 sleep 15
done
[[ $(cat "$root/download.success") == c7fcb9a8bb478e75b46e04fa1d68c82287c047dd ]] || exit 1
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '2p' | tr -d ' ')
if (( used > 500 )); then log "GPU 1 is occupied (${used} MiB); refusing to start"; exit 1; fi
log 'verified model ready; starting vLLM on GPU 1'
export PATH="$(dirname "$python"):$PATH"
export CUDA_VISIBLE_DEVICES=1 HF_HOME="$root/cache/hf" VLLM_CACHE_ROOT="$root/cache/vllm" TRITON_CACHE_DIR="$root/cache/triton" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
exec "$python" -m vllm.entrypoints.openai.api_server \
 --model "$root/model" --tokenizer "$root/tokenizer-compat" --served-model-name glasseek-rl-4b --host 127.0.0.1 --port 18320 \
 --tensor-parallel-size 1 --dtype bfloat16 --gpu-memory-utilization 0.92 \
 --max-model-len 131072 --kv-cache-dtype auto --max-num-seqs 16 --attention-backend TRITON_ATTN --enforce-eager \
 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
 --enable-prompt-tokens-details
