#!/usr/bin/env bash
set -euo pipefail
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${SUITE_PYTHON:-python}" "$root_dir/scripts/evaluation_suite/run.py" "$@"
