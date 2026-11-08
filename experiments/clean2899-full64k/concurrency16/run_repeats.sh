#!/usr/bin/env bash
set -euo pipefail
P=/root/autodl-tmp/Glasseek
D="$P/experiments/clean2899-full64k"
C="$D/concurrency16"
exec 8>"$C/repeats.lock"
flock -n 8 || exit 10
cd "$P"
python "$D/check_models.py" >> "$C/model-check.log" 2>&1
python "$D/preflight.py" >> "$C/tool-check.log" 2>&1
for rep in 1 2 3; do
 export GLASSEEK_VIEWER_RUN=glasseek_clean2899_full64k_step39_128
 export GLASSEEK_VIEWER_WORKSPACE=glasseek-clean2899-full64k-step39-128
 if [[ "$rep" != 1 ]]; then
  export GLASSEEK_VIEWER_RUN="${GLASSEEK_VIEWER_RUN}_rep$rep"
  export GLASSEEK_VIEWER_WORKSPACE="${GLASSEEK_VIEWER_WORKSPACE}-rep$rep"
 fi
 export GLASSEEK_VIEWER_LABEL="Qwen3.5-9B step39 · BrowseComp 128 · repeat $rep · tool cap 120"
 source "$D/eval_env.sh"
 printf 'repeat=%s started=%s\n' "$rep" "$(date -Is)" >> "$C/status.log"
 stop="$C/judge-rep$rep.stop"
 rm -f "$stop"
 "$EVAL_PYTHON" "$P/Evaluation/services/async_judge.py" --run "$GLASSEEK_VIEWER_RUN" --dataset quest_qwen27b_128 --handoff-stop "$stop" >> "$C/judge-rep$rep.log" 2>&1 &
 judge=$!
 bash "$D/publish_viewer.sh" >> "$C/viewer-rep$rep.log" 2>&1 &
 viewer=$!
 trap 'kill "$viewer" "$judge" 2>/dev/null || true' EXIT
 python "$C/register_repeat.py"
 set +e
 bash "$D/run_eval.sh" >> "$C/eval-rep$rep.log" 2>&1
 code=$?
 set -e
 touch "$stop"
 wait "$judge"
 if [[ "$code" != 0 && "$code" != 1 ]]; then echo "Evaluation interrupted or infrastructure failed: $code"; exit "$code"; fi
 export JUDGE_API_KEY="$("$EVAL_PYTHON" -c 'import json,os;print(json.load(open(os.environ["REEVAL_CREDENTIALS"]))["judge"])')"
 (cd "$E" && bash scripts/judge.sh --save-name "$GLASSEEK_VIEWER_RUN" --dataset quest_qwen27b_128 --judge-provider openai --judge-base-url https://api.deepseek.com/v1 --judge-model deepseek-flash --prompt-version official --token-param max_tokens --no-json-schema --temperature 0) > "$C/final-judge-rep$rep.log" 2>&1 || echo "Final judging requires review for repeat $rep"
 kill "$viewer" 2>/dev/null || true
 wait "$viewer" 2>/dev/null || true
 bash "$D/publish_viewer.sh" once
 python "$C/avg3.py"
 printf 'repeat=%s ended=%s\n' "$rep" "$(date -Is)" >> "$C/status.log"
done
