#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Glasseek
D="$PWD/experiments/clean2899-full64k"
exec 9>"$D/train.lock"
flock -n 9 || exit 10
printf 'training_started=%s\n' "$(date -Is)" > "$D/pipeline-status.txt"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv,noheader -l 10 > "$D/gpu.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
if bash "$D/run_train.sh" >> "$D/train.log" 2>&1; then
 printf 'training_completed=%s\n' "$(date -Is)" >> "$D/pipeline-status.txt"
else
 code=$?; printf 'training_failed=%s exit=%s\n' "$(date -Is)" "$code" >> "$D/pipeline-status.txt"; exit "$code"
fi
[[ "$(cat checkpoints/glasseek/sft_clean2899_64k_full_2gpu_turns/latest_checkpointed_iteration.txt)" == 39 ]]
printf 'evaluation_preparing=%s\n' "$(date -Is)" >> "$D/pipeline-status.txt"
if bash "$D/after_train_eval.sh" >> "$D/evaluation-pipeline.log" 2>&1; then
 printf 'evaluation_completed=%s\n' "$(date -Is)" >> "$D/pipeline-status.txt"
else
 code=$?; printf 'evaluation_failed=%s exit=%s\n' "$(date -Is)" "$code" >> "$D/pipeline-status.txt"; exit "$code"
fi
