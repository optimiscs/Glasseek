#!/usr/bin/env bash
set -uo pipefail
BASE=/home/moxu/glasseek-sft-translation
export PYTHONPATH="$BASE/deps" CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_CACHE_ROOT="$BASE/vllm-cache" OMP_NUM_THREADS=4
while true; do
  available=$(df -Pk "$BASE" | awk 'NR==2 {print $4}')
  if (( available < 2097152 )); then echo "$(date -Is) less than 2 GiB free; stopping"; exit 2; fi
  /home/moxu/intentops-qwen35-20261017/venv/bin/python -u "$BASE/translate_offline.py" --backend vllm --batch-size 32 --model "$BASE/model" --input "$BASE/source.jsonl" --output "$BASE/output"
  status=$?
  echo "$(date -Is) worker exited status=$status"
  if (( status == 0 )); then exit 0; fi
  echo "Retrying after failure in 120 seconds; completed records and segment cache are retained."
  sleep 120
done
