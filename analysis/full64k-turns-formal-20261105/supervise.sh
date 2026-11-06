#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/Glasseek
D="$PWD/analysis/full64k-turns-formal-20261105"
exec 9>"$D/train.lock"
flock -n 9 || exit 10
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv,noheader -l 5 > "$D/gpu.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
printf "started=%s\n" "$(date -Is)" > "$D/status.txt"
bash analysis/sft128k/run_train_64k_turns.sh >> "$D/train.log" 2>&1
code=$?
printf "finished=%s\nexit_code=%s\n" "$(date -Is)" "$code" >> "$D/status.txt"
exit "$code"
