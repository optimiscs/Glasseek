#!/usr/bin/env bash
set -u
P=/root/autodl-tmp/Glasseek/experiments/prompt-audit
REMOTE=root@connect.westc.seetacloud.com
SSH='ssh -i /root/.ssh/seetacloud_westc_40809 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=8 -p 24704'
while true; do
 mkdir -p "$P/remote-output" "$P/remote-audit"
 rsync -az --partial -e "$SSH" "$REMOTE:/root/autodl-tmp/Glasseek/Evaluation/output/preds/glasseek_base9b_prompt_audit_20261105/" "$P/remote-output/" 2>/dev/null || true
 rsync -az --partial -e "$SSH" --include='report.json' --include='requests.jsonl' --include='first-request.json' --include='evaluation.log' --exclude='*' "$REMOTE:/root/autodl-tmp/Glasseek/checkpoint-reeval/analysis/prompt-audit/" "$P/remote-audit/" 2>/dev/null || true
 sleep 30
done
