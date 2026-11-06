#!/usr/bin/env bash
set -u
P=/root/autodl-tmp/Glasseek/analysis/no-tool-screen
HOST=root@connect.westc.seetacloud.com
SSH='ssh -i /root/.ssh/seetacloud_westc_40809 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=8 -p 24704'
mkdir -p "$P/remote" /root/autodl-tmp/Glasseek/data/glasseek/sft-no-tool-screened-64k
while true; do
 rsync -az --partial -e "$SSH" --include='progress.json' --include='snapshot.sqlite3' --include='full.log' --exclude='*' "$HOST:/root/autodl-tmp/Glasseek/analysis/no-tool-screen/" "$P/remote/" 2>/dev/null || true
 rsync -az --partial -e "$SSH" "$HOST:/root/autodl-tmp/Glasseek/data/glasseek/sft-no-tool-screened-64k/" /root/autodl-tmp/Glasseek/data/glasseek/sft-no-tool-screened-64k/ 2>/dev/null || true
 sleep 30
done
