#!/usr/bin/env python3
"""Prefetch OpenML datasets for the low_class_wave group into a shared cache.

Uses sklearn.datasets.fetch_openml. Redirect SCIKIT_LEARN_DATA / project cache
via scripts/cluster/env.sh before running.
"""
from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from pathlib import Path

import yaml
from sklearn.datasets import fetch_openml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def acquire_lock(path: Path, timeout_s: float = 3600.0) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    start = time.time()
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.time() - start > timeout_s:
                os.close(fd)
                raise TimeoutError(path)
            time.sleep(2)


def main() -> int:
    root = project_root()
    cfg = yaml.safe_load((root / "configs" / "datasets.yaml").read_text(encoding="utf-8"))
    group = (cfg.get("dataset_groups") or {}).get("low_class_wave") or []
    by_name = {e["name"]: e for e in cfg["datasets"]}
    cache_root = Path(os.environ.get("OPENML_CACHE_DIR", root / ".cache" / "openml"))
    cache_root.mkdir(parents=True, exist_ok=True)
    # Help sklearn if supported by this version
    os.environ.setdefault("SCIKIT_LEARN_DATA", str(cache_root))

    lock_fd = acquire_lock(cache_root / ".warm_openml.lock")
    report = []
    try:
        for name in group:
            entry = by_name[name]
            oid = int(entry["openml_id"])
            print(f"fetching {name} openml_id={oid}", flush=True)
            t0 = time.time()
            bunch = fetch_openml(data_id=oid, as_frame=True, parser="auto")
            X = bunch.data
            y = bunch.target
            report.append(
                {
                    "name": name,
                    "openml_id": oid,
                    "n_rows": int(len(X)),
                    "n_features": int(X.shape[1]),
                    "n_classes": int(y.nunique()) if hasattr(y, "nunique") else None,
                    "seconds": round(time.time() - t0, 3),
                }
            )
            print("  ok", report[-1])
        out = cache_root / "openml_cache_manifest.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(tmp, out)
        print("wrote", out)
        return 0
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ModuleNotFoundError as exc:
        print("Missing dependency:", exc, file=sys.stderr)
        print("Run inside ssl-core or ssl-tfm env.", file=sys.stderr)
        raise SystemExit(1)
