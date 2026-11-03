#!/usr/bin/env bash
set -euo pipefail
root=/home/moxu/glasseek-embedding-eval
python=/home/moxu/intentops-qwen35-20261017/venv/bin/python
mkdir -p "$root/logs" "$root/cache"
log(){ echo "$(date -Is) $*" | tee -a "$root/logs/supervisor-events.log"; }
trap 'log "supervisor failed line=$LINENO"' ERR
log 'waiting for direct model download'
while [[ ! -f "$root/download.exit" ]]; do
  if ! kill -0 "$(cat "$root/download.pid")" 2>/dev/null; then log 'download process stopped without receipt'; exit 1; fi
  sleep 30
done
[[ "$(cat "$root/download.exit")" == 0 ]] || { log 'download failed'; exit 1; }
log 'verifying all model shards against original'
(cd "$root/model" && sha256sum -c "$root/model.sha256") >>"$root/logs/hash-verification.log" 2>&1
log 'model hashes verified'
while IFS=, read -r gpu used; do
  used=${used//[[:space:]]/}
  if (( used > 500 )); then log "GPU $gpu already in use: ${used} MiB, refusing to start"; exit 1; fi
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
for i in 0 1; do
  port=$((18318+i))
  CUDA_VISIBLE_DEVICES="$i" VLLM_CACHE_ROOT="$root/cache/vllm-$i" HF_HOME="$root/cache/hf" \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
    nohup "$python" -m vllm.entrypoints.openai.api_server \
    --model "$root/model" --served-model-name QWEN3-EMBEDDING-8B \
    --host 127.0.0.1 --port "$port" --runner pooling --tensor-parallel-size 1 \
    --dtype bfloat16 --gpu-memory-utilization 0.80 --max-model-len 2048 \
    --max-num-seqs 64 --no-enable-prefix-caching \
    >"$root/logs/vllm-gpu$i.log" 2>&1 </dev/null &
  echo "$!" >"$root/vllm-gpu$i.pid"
  log "started gpu=$i port=$port pid=$!"
done
for i in 0 1; do
  port=$((18318+i));ready=0
  for attempt in $(seq 1 180); do
    if ! kill -0 "$(cat "$root/vllm-gpu$i.pid")" 2>/dev/null; then log "GPU $i vLLM exited"; exit 1; fi
    if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null; then ready=1; break; fi
    sleep 5
  done
  ((ready==1)) || { log "GPU $i readiness timed out"; exit 1; }
done
log 'both endpoints ready; starting ramp'
"$python" "$root/ramp.py" >>"$root/logs/ramp-console.log" 2>&1
log 'ramp process exited; inspect ramp.jsonl for outcome'
