#!/usr/bin/env bash
# Create / update ssl-core env under LAB storage.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/cluster/env.sh"

if [ ! -x "$MAMBA_EXE" ]; then
  echo "micromamba missing; run bootstrap_micromamba.sh first" >&2
  exit 1
fi

YAML="$ROOT/environment/core/environment.yml"
LOCK_DIR="$ROOT/environment/core/lock"
mkdir -p "$LOCK_DIR" "$SSL_CORE_PREFIX"

echo "[bootstrap_ssl_core] prefix=$SSL_CORE_PREFIX"
"$MAMBA_EXE" create -y -p "$SSL_CORE_PREFIX" -f "$YAML" || \
  "$MAMBA_EXE" install -y -p "$SSL_CORE_PREFIX" -f "$YAML"

# Validate imports
"$MAMBA_EXE" run -p "$SSL_CORE_PREFIX" python - <<'PY'
import numpy, pandas, sklearn, yaml, matplotlib
import xgboost, lightgbm, catboost
print("ssl-core OK", numpy.__version__, pandas.__version__, sklearn.__version__)
PY

"$MAMBA_EXE" run -p "$SSL_CORE_PREFIX" python -m pip freeze > "$LOCK_DIR/pip-freeze.txt"
"$MAMBA_EXE" env export -p "$SSL_CORE_PREFIX" > "$LOCK_DIR/environment-explicit.yml" || true
echo "[bootstrap_ssl_core] locks written to $LOCK_DIR"
