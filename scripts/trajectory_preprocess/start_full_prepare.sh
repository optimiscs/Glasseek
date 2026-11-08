#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
OUT="${1:-$ROOT/data/processed/trajectory-glasseek-v7}"
PYTHON="${PREPROCESS_PYTHON:-/root/autodl-tmp/glasseek-verl091-env/bin/python}"
WORKERS="${PREPROCESS_WORKERS:-8}"
mkdir -p "$OUT"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
"$PYTHON" -c 'import requests, jsonschema, yaml; import pyarrow.parquet' || {
    echo "Preflight failed: selected Python must provide requests and pyarrow: $PYTHON" >&2
    exit 1
}
# The Python process also locks the directory, protecting direct CLI launches.
exec 9>"$OUT/launcher.lock"
if ! flock -n 9; then
    echo "Already running: $OUT" >&2
    exit 1
fi
nohup setsid bash -c '
    "$1" -u "$2/full_prepare.py" --output "$3" --workers "$4" &&
    "$1" -u "$2/../trajectory_benchmark/audit_glasseek_contract.py" --output "$3"
' _ "$PYTHON" "$SCRIPT_DIR" "$OUT" "$WORKERS" >>"$OUT/run.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$OUT/run.pid"
sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
    if wait "$pid"; then
        echo "COMPLETED=$OUT"
        echo "LOG=$OUT/run.log"
        exit 0
    fi
    echo "Startup exited; inspect $OUT/run.log" >&2
    exit 1
fi
echo "PID=$pid"
echo "WORKERS=$WORKERS"
echo "OUTPUT=$OUT"
echo "LOG=$OUT/run.log"
