#!/usr/bin/env bash
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${REEVAL_SNAPSHOT_DIR:-$P/remote_snapshot}"
HOST="${REEVAL_SSH_HOST:-root@connect.westc.seetacloud.com}"
PORT="${REEVAL_SSH_PORT:-24704}"
KEY="${REEVAL_SSH_KEY:-/root/.ssh/seetacloud_westc_40809}"
REMOTE="${REEVAL_REMOTE_ROOT:-/root/autodl-tmp/Glasseek}"
SITE="${QUEST_VIEWER_SITE:-/root/autodl-tmp/Quest/web/trace-viewer}"
ssh_cmd="ssh -i $KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -p $PORT"
for step in 10 20 30 40 50 60 70 80; do
  name="glasseek_step${step}_rl9b_subset"
  viewer="glasseek-step${step}-rl9b-34"
  mkdir -p "$DEST/checkpoints/$name" "$SITE/datasets/$viewer" "$DEST/checkpoints/$name/sessions"
  rsync -az --partial -e "$ssh_cmd" "$HOST:$REMOTE/Evaluation/output/preds/$name/" "$DEST/checkpoints/$name/output/" 2>/dev/null || continue
  ssh -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -p "$PORT" "$HOST" /root/miniconda3/bin/python - \
    "$REMOTE/Evaluation/output/preds/$name/browsecomp_rl9b_non_timeout_34/summary.jsonl" \
    "$REMOTE/Evaluation/data/runtime/hermes/profiles" < "$P/export_session_backups.py" | tar -xzf - -C "$DEST/checkpoints/$name/sessions" || true
  rsync -az --partial -e "$ssh_cmd" "$HOST:$REMOTE/checkpoint-reeval/viewer/$viewer.json" "$SITE/datasets/$viewer/traces.json" 2>/dev/null || continue
  /root/miniconda3/bin/python "$P/register_viewer_dataset.py" "$SITE" "$step"
done
