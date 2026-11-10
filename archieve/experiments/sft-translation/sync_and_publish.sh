#!/usr/bin/env bash
set -uo pipefail
P=/root/autodl-tmp/Glasseek/experiments/sft-translation
REMOTE=/home/moxu/glasseek-sft-translation/output
mkdir -p "$P/translated"
while true; do
  if rsync -az --partial --include='trace-*.json' --exclude='*' -e 'ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=10' "44:$REMOTE/" "$P/translated/"; then
    result=$(/root/miniconda3/bin/python "$P/publish_viewer.py")
    printf '%s %s\n' "$(date -Is)" "$result"
    if [[ "$result" != *'"updated_workspaces": 0'* ]]; then touch "$P/.needs-import"; fi
    if [[ -e "$P/.needs-import" ]]; then
      (cd /root/autodl-tmp/Quest/web && /root/autodl-tmp/Quest/workspace/viewer/venv/bin/quest-viewer import --site trace-viewer) && rm -f "$P/.needs-import" || echo 'import failed; retry next cycle'
    fi
  else
    printf '%s rsync failed; retry\n' "$(date -Is)"
  fi
  sleep 30
done
