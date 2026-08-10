#!/usr/bin/env bash
# Idempotent user-space micromamba install into LAB storage.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/cluster/env.sh"

echo "[bootstrap_micromamba] MAMBA_ROOT_PREFIX=$MAMBA_ROOT_PREFIX"
mkdir -p "$MAMBA_ROOT_PREFIX" "$SSL_TMP"

if [ -x "$MAMBA_EXE" ]; then
  echo "[bootstrap_micromamba] already present: $MAMBA_EXE"
  "$MAMBA_EXE" --version
  exit 0
fi

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) PLATFORM="linux-64" ;;
  aarch64) PLATFORM="linux-aarch64" ;;
  *) echo "Unsupported arch: $ARCH" >&2; exit 1 ;;
esac

URL="https://micro.mamba.pm/api/micromamba/${PLATFORM}/latest"
TMP="$SSL_TMP/micromamba_${PLATFORM}.tar.bz2"

echo "[bootstrap_micromamba] downloading $URL"
curl -L --fail --retry 3 --retry-delay 2 -o "$TMP" "$URL"
mkdir -p "$MAMBA_ROOT_PREFIX"
tar -xjf "$TMP" -C "$MAMBA_ROOT_PREFIX" --strip-components=1 bin/micromamba
chmod +x "$MAMBA_EXE"
"$MAMBA_EXE" --version
echo "[bootstrap_micromamba] OK"
