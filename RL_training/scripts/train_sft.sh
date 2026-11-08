#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${SFT_LAUNCH_PYTHON:-python3}" "$root/RL_training/glasseek_training/sft_launch.py" "$@"
