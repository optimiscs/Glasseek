#!/usr/bin/env bash
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${REEVAL_SNAPSHOT_DIR:-$P/remote_snapshot}"
HOST="${REEVAL_SSH_HOST:-root@connect.westc.seetacloud.com}"
PORT="${REEVAL_SSH_PORT:-24704}"
KEY="${REEVAL_SSH_KEY:-/root/.ssh/seetacloud_westc_40809}"
REMOTE="${REEVAL_REMOTE_ROOT:-/root/autodl-tmp/Glasseek}"
mkdir -p "$DEST/code" "$DEST/output"
ssh_cmd="ssh -i $KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -p $PORT"
sync_once() {
  rsync -az --partial -e "$ssh_cmd" --include='*/' --include='*.py' --include='*.sh' --include='*.json' --include='*.jsonl' --include='*.md' --include='*.log' --exclude='adapters/***' --exclude='*' "$HOST:$REMOTE/checkpoint-reeval/analysis/checkpoint_reeval/" "$DEST/code/" || return 1
  rsync -az --partial -e "$ssh_cmd" "$HOST:$REMOTE/Evaluation/output/preds/glasseek_base9b_rl9b_subset/" "$DEST/output/" || return 1
  mkdir -p "$DEST/sessions"
  # SQLite's online backup API gives consistent snapshots while Hermes is writing.
  ssh -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -p "$PORT" "$HOST" \
    /root/miniconda3/bin/python - \
    "$REMOTE/Evaluation/output/preds/glasseek_base9b_rl9b_subset/browsecomp_rl9b_non_timeout_34/summary.jsonl" \
    "$REMOTE/Evaluation/data/runtime/hermes/profiles" < "$P/export_session_backups.py" | tar -xzf - -C "$DEST/sessions" || return 1
  viewer_dir="${QUEST_VIEWER_SITE:-/root/autodl-tmp/Quest/web/trace-viewer}/datasets/glasseek-base9b-rl9b-34"
  mkdir -p "$viewer_dir"
  rsync -az -e "$ssh_cmd" "$HOST:$REMOTE/checkpoint-reeval/viewer/traces.json" "$viewer_dir/traces.json" || return 1
}
if [[ "${1:-}" == --once ]]; then sync_once; exit $?; fi
while true; do
  sync_once && date '+sync successful %F %T %z' || date '+sync failed %F %T %z'
  sleep "${REEVAL_SYNC_SECONDS:-60}"
done
