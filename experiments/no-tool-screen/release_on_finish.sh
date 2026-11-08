#!/usr/bin/env bash
set -u
P=/root/autodl-tmp/Glasseek/data/glasseek/sft-no-tool-screened-64k/manifest.json
while [[ ! -s "$P" ]]; do sleep 30; done
for i in 0 1 2 3; do screen -S "glasseek-no-tool-vllm-$i" -X quit >/dev/null 2>&1 || true; done
printf '%s all inference replicas stopped after export\n' "$(date -Is)"
