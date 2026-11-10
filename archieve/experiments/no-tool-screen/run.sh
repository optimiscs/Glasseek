#!/usr/bin/env bash
set -euo pipefail
P=/root/autodl-tmp/Glasseek/experiments/no-tool-screen
cd "$P"
/root/autodl-tmp/Glasseek/Evaluation/data/runtime/venv/bin/python -u rule_screen.py infer --gpus 4 --per-gpu 4
/root/autodl-tmp/Glasseek/Evaluation/data/runtime/venv/bin/python -u rule_screen.py audit
