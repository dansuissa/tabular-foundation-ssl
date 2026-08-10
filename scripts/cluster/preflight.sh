#!/usr/bin/env bash
# Read-only / light preflight for cluster readiness. No heavy downloads.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/cluster/env.sh"

echo "=== HOST ==="
hostname -f || hostname
uname -a
echo "=== SLURM ==="
command -v sbatch srun sinfo >/dev/null
echo "job_id=${SLURM_JOB_ID:-none} partition=${SLURM_JOB_PARTITION:-none} gpus=${SLURM_GPUS:-none}"
sinfo -N -p p_uriofir,A100-4h -o '%N %P %a %G %c %m %l' || true
echo "=== GPU ==="
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
else
  echo "nvidia-smi missing"
fi
echo "=== PATHS ==="
echo "SSL_PROJECT_ROOT=$SSL_PROJECT_ROOT"
echo "LAB_ROOT=$LAB_ROOT"
echo "MAMBA_EXE=$MAMBA_EXE (exists=$( [[ -x ${MAMBA_EXE} ]] && echo yes || echo no ))"
echo "SSL_CORE_PREFIX=$SSL_CORE_PREFIX (exists=$( [[ -d ${SSL_CORE_PREFIX} ]] && echo yes || echo no ))"
echo "SSL_TFM_PREFIX=$SSL_TFM_PREFIX (exists=$( [[ -d ${SSL_TFM_PREFIX} ]] && echo yes || echo no ))"
echo "=== AUTH (presence only) ==="
"$ROOT/scripts/cluster/check_auth.py" || true
echo "=== DISK ==="
df -h "$LAB_ROOT" "$SSL_PROJECT_ROOT" /tmp | sed -n '1,10p'
echo "=== NET (light) ==="
curl -I --max-time 5 https://pypi.org 2>/dev/null | head -1 || echo "pypi unreachable"
curl -I --max-time 5 https://huggingface.co 2>/dev/null | head -1 || echo "hf unreachable"
echo "=== PREFLIGHT DONE ==="
