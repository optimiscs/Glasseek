#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/Glasseek/analysis/clean2899-full64k/eval_env.sh
file="/root/autodl-tmp/Quest/web/trace-viewer/datasets/$GLASSEEK_VIEWER_WORKSPACE/traces.json"
last=''
while true; do
 "$EVAL_PYTHON" "$E/quest_viewer_checkpoint_adapter.py" --output "$file" --once
 if [[ -f "$file" ]]; then
  hash=$(sha256sum "$file" | cut -d' ' -f1)
  if [[ "$hash" != "$last" ]]; then
   (cd /root/autodl-tmp/Quest/web && /root/autodl-tmp/Quest/workspace/viewer/venv/bin/quest-viewer import --site trace-viewer)
   last=$hash
  fi
 fi
 [[ "${1:-}" == once ]] && break
 sleep 20
done
