#!/usr/bin/env bash
set -euo pipefail
search_service_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
search_root="$(cd "$search_service_dir/../.." && pwd)"
search_redis_dir="${SEARCH_POOL_REDIS_DIR:-$search_root/Evaluation/data/runtime/search_pool/redis}"
mkdir -p "$search_redis_dir"
chmod 700 "$search_redis_dir"
umask 077
search_redis_bin="${SEARCH_POOL_REDIS_BIN:-$(command -v redis-server || true)}"
if [[ -z "$search_redis_bin" ]]; then
  search_redis_bin="$search_root/.cache/search-pool/redis/usr/bin/redis-server"
  export LD_LIBRARY_PATH="$search_root/.cache/search-pool/redis/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export LUA_CPATH="$search_root/.cache/search-pool/redis/usr/lib/x86_64-linux-gnu/lua/5.1/?.so;;"
fi
exec "$search_redis_bin" "$search_root/Evaluation/config/search_pool_redis.conf" --dir "$search_redis_dir" "$@"
