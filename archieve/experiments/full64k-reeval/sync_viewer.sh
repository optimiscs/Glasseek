#!/usr/bin/env bash
set -uo pipefail
P=/root/autodl-tmp/Glasseek/experiments/full64k-reeval
SITE=/root/autodl-tmp/Quest/web/trace-viewer
NAME=glasseek-full64k-step76-34
REMOTE=/root/autodl-tmp/Glasseek/checkpoint-reeval/viewer/$NAME.json
mkdir -p "$SITE/datasets/$NAME"
last=''
while true; do
 if rsync -az --partial -e 'ssh -i /root/.ssh/seetacloud_westc_40809 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=8 -p 24704' "root@connect.westc.seetacloud.com:$REMOTE" "$SITE/datasets/$NAME/traces.json"; then
  hash=$(sha256sum "$SITE/datasets/$NAME/traces.json" | cut -d' ' -f1)
  if [[ "$hash" != "$last" ]]; then
   /root/miniconda3/bin/python "$P/register_viewer.py" && (cd /root/autodl-tmp/Quest/web && /root/autodl-tmp/Quest/workspace/viewer/venv/bin/quest-viewer import --site trace-viewer > "$P/viewer-import.log" 2>&1) && last=$hash && printf '%s synced %s\n' "$(date -Is)" "$hash"
  fi
 else printf '%s sync failed; retry\n' "$(date -Is)"; fi
 sleep 20
done
