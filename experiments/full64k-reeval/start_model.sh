#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${TOOL_CONFIG_PYTHON:-python3}" "$root/Evaluation/services/model_server.py" step76 "$@"
