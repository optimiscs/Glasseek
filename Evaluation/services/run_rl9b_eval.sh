#!/usr/bin/env bash
set -euo pipefail
service="$(cd "$(dirname "${BASH_SOURCE[0]}")/." && pwd)"
exec "${TOOL_CONFIG_PYTHON:-python3}" "$service/eval_run.py" rl9b "$@"
