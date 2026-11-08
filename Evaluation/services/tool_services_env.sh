# Shared tool routing; experiment names and training settings belong to callers.
_tool_services_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_tool_services_exports="$("${TOOL_CONFIG_PYTHON:-python3}" "$_tool_services_dir/tool_services.py" env)" || return
eval "$_tool_services_exports"
source "$_tool_services_dir/search_pool_env.sh"
unset _tool_services_dir _tool_services_exports
