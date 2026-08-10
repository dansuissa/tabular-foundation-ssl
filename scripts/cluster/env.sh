# Shared cluster paths and thread settings for ssl_tabular_benchmark
# Usage: source scripts/cluster/env.sh

# Do not set secrets here.

export SSL_PROJECT_ROOT="${SSL_PROJECT_ROOT:-/home/eng/suissad4/projects/ssl-foundation-models/ssl_tabular_benchmark}"
export LAB_ROOT="${LAB_ROOT:-/private/ofirlin-lab/suissad4}"

export SSL_ENVS_ROOT="${SSL_ENVS_ROOT:-$LAB_ROOT/envs}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$SSL_ENVS_ROOT/micromamba}"
export MAMBA_EXE="${MAMBA_EXE:-$MAMBA_ROOT_PREFIX/bin/micromamba}"

export SSL_CORE_PREFIX="${SSL_CORE_PREFIX:-$SSL_ENVS_ROOT/ssl-core}"
export SSL_TFM_PREFIX="${SSL_TFM_PREFIX:-$SSL_ENVS_ROOT/ssl-tfm}"

export SSL_CACHE_ROOT="${SSL_CACHE_ROOT:-$LAB_ROOT/caches}"
export HF_HOME="${HF_HOME:-$SSL_CACHE_ROOT/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$SSL_CACHE_ROOT/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SSL_CACHE_ROOT/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SSL_CACHE_ROOT/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SSL_CACHE_ROOT/uv}"
export OPENML_CACHE_DIR="${OPENML_CACHE_DIR:-$SSL_CACHE_ROOT/openml}"
# sklearn fetch_openml uses ~/scikit_learn_data by default; redirect if supported:
export SCIKIT_LEARN_DATA="${SCIKIT_LEARN_DATA:-$OPENML_CACHE_DIR}"

export SSL_RESULTS_ROOT="${SSL_RESULTS_ROOT:-$LAB_ROOT/results}"
export SSL_SHARD_ROOT="${SSL_SHARD_ROOT:-$SSL_RESULTS_ROOT/raw_shards}"
export SSL_LOG_ROOT="${SSL_LOG_ROOT:-$SSL_RESULTS_ROOT/logs}"
export SSL_COMBINED_ROOT="${SSL_COMBINED_ROOT:-$SSL_RESULTS_ROOT/combined}"
export SSL_TMP="${SSL_TMP:-$LAB_ROOT/tmp}"

# Headless PriorLabs / HF behavior
export TABPFN_NO_BROWSER="${TABPFN_NO_BROWSER:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

# Threading: prefer Slurm allocation if present
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
  _NT="$SLURM_CPUS_PER_TASK"
else
  _NT="${SSL_DEFAULT_THREADS:-8}"
fi
export OMP_NUM_THREADS="$_NT"
export MKL_NUM_THREADS="$_NT"
export OPENBLAS_NUM_THREADS="$_NT"
export NUMEXPR_NUM_THREADS="$_NT"
export TORCH_NUM_THREADS="$_NT"

mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" \
  "$OPENML_CACHE_DIR" "$SSL_SHARD_ROOT" "$SSL_LOG_ROOT" "$SSL_COMBINED_ROOT" "$SSL_TMP" \
  "$SSL_CACHE_ROOT/tabpfn" "$SSL_CACHE_ROOT/tabicl"

# Optional activation helpers
ssl_activate_core() {
  # shellcheck disable=SC1091
  eval "$("$MAMBA_EXE" shell hook -s bash)"
  micromamba activate "$SSL_CORE_PREFIX"
}

ssl_activate_tfm() {
  # shellcheck disable=SC1091
  eval "$("$MAMBA_EXE" shell hook -s bash)"
  micromamba activate "$SSL_TFM_PREFIX"
}
