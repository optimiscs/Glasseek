#!/usr/bin/env bash
set -euo pipefail
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"
python_bin="${SUITE_PYTHON:-python}"
base_dir="$root_dir/data/evaluation/suite5pct-v1"
config_path="$root_dir/scripts/evaluation_suite/config.json"
source /etc/network_turbo >/dev/null 2>&1
"$python_bin" scripts/evaluation_suite/acquire.py --base "$base_dir"
if [[ ! -f "$base_dir/training/manifest.json" ]]; then
  # A nonzero status requires the independent non-task audit below. Data/schema
  # errors remain fatal there; they are never treated as completed exclusions.
  "$python_bin" scripts/evaluation_suite/training_index.py --config "$config_path" --output "$base_dir/training" || index_status=$?
fi
"$python_bin" scripts/evaluation_suite/training_index.py --config "$config_path" --output "$base_dir/training" --audit-non-tasks
"$python_bin" scripts/evaluation_suite/suite.py build --config "$config_path" --output "${1:-$base_dir/release}"
