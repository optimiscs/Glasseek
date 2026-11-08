#!/usr/bin/env bash
set -euo pipefail
REVIEW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QUEST_REVIEW_ROOT="${QUEST_ROOT:-$(dirname "$REVIEW_ROOT")/Quest}"
export GLASSEEK_ROOT="$REVIEW_ROOT"
export GLASSEEK_REVIEW_DB="${GLASSEEK_REVIEW_DB:-$REVIEW_ROOT/data/processed/trajectory-human-review-v1/reviews.sqlite}"
export VIEWER_PYTHON="${VIEWER_PYTHON:-$QUEST_REVIEW_ROOT/workspace/viewer/venv/bin/python}"
cd "$QUEST_REVIEW_ROOT/web"
exec "$VIEWER_PYTHON" serve.py "$@"
