#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/RL_training${PYTHONPATH:+:$PYTHONPATH}"
: "${MODEL_PATH:?Set MODEL_PATH to a base model or local checkpoint}"
python -c 'from glasseek_training.validation import require_verl_version; require_verl_version()'
exec torchrun \
  --nnodes="${NNODES:-1}" --nproc_per_node="${NPROC_PER_NODE:-4}" \
  --node_rank="${NODE_RANK:-0}" --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" \
  -m verl.trainer.sft_trainer \
  --config-path="$ROOT/RL_training/configs" --config-name=sft \
  "trainer.nnodes=${NNODES:-1}" "trainer.n_gpus_per_node=${NPROC_PER_NODE:-4}" "$@"
