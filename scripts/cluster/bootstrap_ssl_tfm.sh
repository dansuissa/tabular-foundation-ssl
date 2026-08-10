#!/usr/bin/env bash
# Create / update ssl-tfm env: conda base deps, then CUDA torch, then tabpfn+tabicl.
# Prefer running under: salloc/sbatch -p p_uriofir --gres=gpu:1 -A ug_uri_ofir
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/cluster/env.sh"

if [ ! -x "$MAMBA_EXE" ]; then
  echo "micromamba missing; run bootstrap_micromamba.sh first" >&2
  exit 1
fi

YAML="$ROOT/environment/tfm/environment.yml"
LOCK_DIR="$ROOT/environment/tfm/lock"
mkdir -p "$LOCK_DIR" "$SSL_TFM_PREFIX"

# Default CUDA wheel index — override if needed after preflight.
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"

echo "[bootstrap_ssl_tfm] prefix=$SSL_TFM_PREFIX"
"$MAMBA_EXE" create -y -p "$SSL_TFM_PREFIX" -f "$YAML" || \
  "$MAMBA_EXE" install -y -p "$SSL_TFM_PREFIX" -f "$YAML"

echo "[bootstrap_ssl_tfm] installing torch from $TORCH_CUDA_INDEX"
"$MAMBA_EXE" run -p "$SSL_TFM_PREFIX" python -m pip install --upgrade pip
"$MAMBA_EXE" run -p "$SSL_TFM_PREFIX" python -m pip install \
  --index-url "$TORCH_CUDA_INDEX" \
  torch torchvision

echo "[bootstrap_ssl_tfm] installing tabpfn + tabicl"
"$MAMBA_EXE" run -p "$SSL_TFM_PREFIX" python -m pip install \
  "tabpfn>=2.0" "tabicl>=2.0" huggingface_hub

# GPU validation
"$MAMBA_EXE" run -p "$SSL_TFM_PREFIX" python - <<'PY'
import torch, tabpfn, tabicl, sklearn, pandas, numpy
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda.is_available() is False — aborting lock write")
x = torch.randn(1024, 1024, device="cuda")
y = x @ x.T
print("gpu_matmul_ok", float(y.mean().detach().cpu()), "device", torch.cuda.get_device_name(0))
print("tabpfn", getattr(tabpfn, "__version__", "?"))
print("tabicl", getattr(tabicl, "__version__", "?"))
print("ssl-tfm OK")
PY

"$MAMBA_EXE" run -p "$SSL_TFM_PREFIX" python -m pip freeze > "$LOCK_DIR/pip-freeze.txt"
"$MAMBA_EXE" env export -p "$SSL_TFM_PREFIX" > "$LOCK_DIR/environment-explicit.yml" || true
"$MAMBA_EXE" run -p "$SSL_TFM_PREFIX" python - <<'PY' > "$LOCK_DIR/torch-build.txt"
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cudnn", torch.backends.cudnn.version())
print("device0", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
echo "[bootstrap_ssl_tfm] locks written to $LOCK_DIR"
echo "[bootstrap_ssl_tfm] NOTE: TabPFN weights still require TABPFN_TOKEN + warm_model_cache.py"
