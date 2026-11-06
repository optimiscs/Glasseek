#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/Glasseek/analysis/clean2899-full64k/eval_env.sh
cd "$P"
export PYTHONPATH="$P/RL_training${PYTHONPATH:+:$PYTHONPATH}"
TRAIN_PY=/root/autodl-tmp/glasseek-verl091-env/bin/python
"$TRAIN_PY" -m verl.model_merger merge --backend fsdp --local_dir "$P/checkpoints/glasseek/sft_clean2899_64k_full_2gpu_turns/global_step_39" --target_dir "$D/step39-hf" --use_cpu_initialization > "$D/export.log" 2>&1
timeout 600 "$TRAIN_PY" "$D/verify_weights.py" > "$D/verify-weights.log" 2>&1
rm -f "$D/judge.stop"
python "$D/preflight.py"
for gpu in 0 1; do
 port=$((19139+gpu*100))
 screen -dmS "clean2899-vllm-$gpu" bash -c 'exec bash "$1" "$2" "$3" > "$4" 2>&1' _ "$D/start_model.sh" "$gpu" "$port" "$D/vllm-$gpu.log"
done
python "$D/check_models.py"
"$EVAL_PYTHON" "$P/analysis/checkpoint_reeval/async_judge.py" --run "$GLASSEEK_VIEWER_RUN" --dataset quest_qwen27b_128 --handoff-stop "$D/judge.stop" > "$D/judge.log" 2>&1 &
judge=$!
bash "$D/publish_viewer.sh" > "$D/viewer.log" 2>&1 &
viewer=$!
trap 'kill "$viewer" 2>/dev/null || true' EXIT
python "$D/register_viewer.py"
set +e
bash "$D/run_eval.sh" > "$D/eval.log" 2>&1
code=$?
set -e
touch "$D/judge.stop"
wait "$judge"
export JUDGE_API_KEY="$("$EVAL_PYTHON" -c 'import json,os;print(json.load(open(os.environ["REEVAL_CREDENTIALS"]))["judge"])')"
cd "$E"
bash scripts/judge.sh --save-name "$GLASSEEK_VIEWER_RUN" --dataset quest_qwen27b_128 --judge-provider openai --judge-base-url https://api.deepseek.com/v1 --judge-model deepseek-flash --prompt-version official --token-param max_tokens --no-json-schema --temperature 0 > "$D/final-judge.log" 2>&1
bash "$D/publish_viewer.sh" once
exit "$code"
