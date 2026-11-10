#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/Glasseek/experiments/clean2899-full64k/eval_env.sh
cd "$P"
exec "$EVAL_PYTHON" -u experiments/faircompare120-20261106/controller.py
