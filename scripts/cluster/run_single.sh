#!/usr/bin/env bash
# Execute one Slurm array task → one atomic result shard via src.run_single.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/cluster/env.sh"

TASK_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-id) TASK_ID="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${TASK_ID}" ]]; then
  echo "--task-id required" >&2
  exit 2
fi

WAVE="${SSL_WAVE:?SSL_WAVE not set}"
TASK_MAP="${SSL_TASK_MAP:?SSL_TASK_MAP not set}"

read -r METHOD_NAME DATASET SEED BUDGET < <(
  python3 - <<PY
import json
from pathlib import Path
tasks = json.loads(Path("${TASK_MAP}").read_text())
task = next(t for t in tasks if int(t["task_id"]) == int("${TASK_ID}"))
print(task["method"], task["dataset"], task["seed"], task["n_labeled"])
PY
)

ENV_PREFIX="${SSL_ENV_PREFIX:-}"
if [[ -z "$ENV_PREFIX" ]]; then
  case "$METHOD_NAME" in
    tabpfn*|tabicl*|tfm_*|*_geometric*|*_laplacian_adapter|stunt|seba|d2r2*|geometric_attention*|laplacian_*|prototype_*|retrieval_attention*|vime*|scarf|sslae)
      ENV_PREFIX="$SSL_TFM_PREFIX"
      ;;
    *)
      ENV_PREFIX="$SSL_CORE_PREFIX"
      ;;
  esac
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  # Fall back to ssl-core when TFM env is missing for classical methods.
  if [[ -x "${SSL_CORE_PREFIX}/bin/python" ]]; then
    ENV_PREFIX="$SSL_CORE_PREFIX"
  else
    echo "ERROR: env python missing at ${ENV_PREFIX}/bin/python" >&2
    exit 3
  fi
fi

cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

set +e
"${ENV_PREFIX}/bin/python" -m src.run_single \
  --wave "$WAVE" \
  --dataset "$DATASET" \
  --method "$METHOD_NAME" \
  --seed "$SEED" \
  --label-budget "$BUDGET" \
  --resume
rc=$?
set -e

if [[ $rc -ne 0 ]]; then
  echo "run_single failed rc=$rc method=$METHOD_NAME dataset=$DATASET seed=$SEED budget=$BUDGET" >&2
fi
exit "$rc"
