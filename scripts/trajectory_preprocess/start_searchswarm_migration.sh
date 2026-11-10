#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
OUT="${1:-$ROOT/data/processed/searchswarm-glasseek-migration-v3}"
PYTHON="${PREPROCESS_PYTHON:-/root/autodl-tmp/glasseek-verl091-env/bin/python}"
mkdir -p "$OUT"
if [[ -f "$OUT/manifest.json" ]]; then
  echo "Migration output already exists; use a new output directory: $OUT" >&2
  exit 1
fi
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
exec 9>"$OUT/launcher.lock"
flock -n 9 || exit 1
nohup setsid bash -c '
    "$1" -u "$2/migrate_searchswarm.py" --output "$3" &&
    "$1" -u "$2/export_migrated_review_sft.py" --migration "$3" &&
    "$1" -u "$2/audit_searchswarm_migration.py" --migration "$3"
' _ "$PYTHON" "$SCRIPT_DIR" "$OUT" >>"$OUT/run.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$OUT/run.pid"
echo "PID=$pid OUTPUT=$OUT"
