#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
source "$ROOT/Evaluation/services/search_pool_env.sh"
export PYTHONPATH="$ROOT/RL_training${PYTHONPATH:+:$PYTHONPATH}"
export GLASSEEK_AGENT_CONFIG="$ROOT/RL_training/configs/agent_loop.yaml"
: "${MODEL_PATH:?Set MODEL_PATH to the exported SFT checkpoint}"
exec python -m glasseek_training.main_rl \
  "trainer.nnodes=${NNODES:-1}" "trainer.n_gpus_per_node=${NPROC_PER_NODE:-8}" "$@"
