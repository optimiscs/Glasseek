# Source this file in consumers; never source an experiment-specific config.
_search_pool_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_search_pool_exports="$("${SEARCH_POOL_PYTHON:-python3}" "$_search_pool_dir/search_pool.py" env)" || return
eval "$_search_pool_exports"
unset _search_pool_dir _search_pool_exports
