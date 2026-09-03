"""Atomic shard I/O for cluster-parallel benchmark runs."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def shard_root(wave: str) -> Path:
    base = Path(os.environ.get("SSL_SHARD_ROOT", "results/raw_shards"))
    path = base / wave
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_id(
    dataset: str,
    method: str,
    seed: int,
    n_labeled: int,
    method_config_hash: str,
    split_protocol_version: str,
    code_version: str,
) -> str:
    raw = "|".join(
        [
            dataset,
            method,
            str(int(seed)),
            str(int(n_labeled)),
            method_config_hash,
            split_protocol_version,
            code_version,
        ]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    safe = f"{dataset}__{method}__s{seed}__n{n_labeled}__{digest}"
    return safe.replace("/", "_")


def shard_path(wave: str, run_id: str) -> Path:
    return shard_root(wave) / f"{run_id}.json"


def write_shard_atomic(wave: str, run_id: str, payload: dict[str, Any]) -> Path:
    dest = shard_path(wave, run_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{run_id}.",
        suffix=".json.tmp",
        dir=str(dest.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, dest)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return dest


def read_shard(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def list_shards(wave: str) -> list[Path]:
    root = shard_root(wave)
    skip = {"task_map.json"}
    out = []
    for path in sorted(root.glob("*.json")):
        if path.name in skip or path.name.startswith("."):
            continue
        out.append(path)
    return out


def shard_success_exists(wave: str, run_id: str) -> bool:
    path = shard_path(wave, run_id)
    if not path.exists():
        return False
    try:
        return read_shard(path).get("status") == "success"
    except Exception:
        return False
