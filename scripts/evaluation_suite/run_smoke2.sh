#!/usr/bin/env bash
# Prepare/run only when the user invokes this entrypoint.
set -euo pipefail
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$root_dir/scripts/evaluation_suite/launch.sh" \
  --phase all --per-benchmark 2 --seed 20261108 \
  --workers 5 --context-manager clm --url-mode raw --allow-blocked \
  --score-config "$root_dir/scripts/evaluation_suite/score.deepseek.json" \
  --run-id suite_smoke2_raw_clm_c5 "$@"
