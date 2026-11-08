#!/usr/bin/env bash
set -euo pipefail
search_pool_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
search_pool_python="${SEARCH_POOL_PYTHON:-${EVAL_PYTHON:-$search_pool_dir/../data/runtime/venv/bin/python}}"
if [[ ! -x "$search_pool_python" ]] && ! command -v "$search_pool_python" >/dev/null 2>&1; then
  search_pool_python=python3
fi
exec "$search_pool_python" "$search_pool_dir/search_pool.py" "${@:-serve}"
