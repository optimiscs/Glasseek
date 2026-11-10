#!/usr/bin/env bash
# Installs score-only dependencies in a separate environment. Does not touch vLLM.
set -euo pipefail
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
score_env="${SUITE_SCORE_ENV:-$root_dir/data/evaluation/suite5pct-v1/score-venv}"
lock_file="$root_dir/data/evaluation/suite5pct-v1/score-requirements.lock.txt"
"${SUITE_PYTHON:-python}" -m venv "$score_env"
if [[ -f "$lock_file" ]]; then
  "$score_env/bin/python" -m pip install -r "$lock_file"
  # New transport features can add dependencies without modifying the frozen
  # baseline lock. Installed versions are retained unless a requirement changes.
  "$score_env/bin/python" -m pip install -r "$root_dir/scripts/evaluation_suite/score-requirements.txt"
else
  "$score_env/bin/python" -m pip install -r "$root_dir/scripts/evaluation_suite/score-requirements.txt"
fi
"$score_env/bin/python" -m pip freeze > "$score_env/requirements.lock.txt"
if [[ ! -f "$lock_file" ]]; then
  cp "$score_env/requirements.lock.txt" "$lock_file"
fi
printf 'SUITE_SCORE_PYTHON=%s\n' "$score_env/bin/python"
# Mind2Web2 requires a browser for evidence/attribution. Install only when used:
# "$score_env/bin/python" -m patchright install chromium
