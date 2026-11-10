#!/usr/bin/env bash
set -euo pipefail
export PATH="/root/miniconda3/envs/ms-swift/bin:$PATH"
P=/root/autodl-tmp/Glasseek/experiments/no-tool-strong-audit
V=/root/autodl-tmp/Quest/workspace/env/bin
mkdir -p "$P"
for gpu in 0 1 2 3; do
 port=$((18300+gpu))
 CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VLLM_CACHE_ROOT="/tmp/strong-audit-vllm-$gpu" "$V/vllm" serve /root/autodl-fs/models/Qwen3.5-9B --served-model-name glasseek-base-9b --host 127.0.0.1 --port "$port" --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --max-num-seqs 4 --max-num-batched-tokens 8192 --reasoning-parser qwen3 >"$P/vllm-$gpu.log" 2>&1 &
done
for port in 18300 18301 18302 18303; do
 ready=0
 for ((i=0;i<180;i++)); do
  if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null; then ready=1; break; fi
  sleep 2
 done
 if ((ready==0)); then echo "server not ready: $port"; exit 1; fi
done
"$V/python" -u "$P/pipeline.py" >"$P/pipeline.log" 2>&1

"$V/python" -u "$P/export.py" >"$P/export.log" 2>&1
