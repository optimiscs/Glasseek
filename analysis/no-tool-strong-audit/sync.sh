#!/usr/bin/env bash
set -u
P=/root/autodl-tmp/Glasseek/analysis/no-tool-strong-audit
SSH='ssh -i /root/.ssh/seetacloud_westc_40809 -p 24704 -o BatchMode=yes -o ConnectTimeout=10'
REMOTE=root@connect.westc.seetacloud.com
while true; do
 $SSH "$REMOTE" "/root/autodl-tmp/Quest/workspace/env/bin/python -c \"import sqlite3,os; p='$P'; s=p+'/results.sqlite3'; os.path.exists(s) and sqlite3.connect(s).backup(sqlite3.connect(p+'/snapshot.sqlite3'))\"" >/dev/null 2>&1 || true
 rsync -az --ignore-missing-args -e "$SSH" "$REMOTE:$P/snapshot.sqlite3" "$REMOTE:$P/pipeline.log" "$REMOTE:$P/decisions.jsonl" "$REMOTE:$P/retained.jsonl" "$REMOTE:$P/manifest.json" "$REMOTE:$P/export.log" "$P/" >/dev/null 2>&1 || true
 sleep 120
done
