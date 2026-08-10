#!/usr/bin/env python3
"""Serially warm TabPFN-3 and TabICLv2 checkpoints with a file lock.

Run inside ssl-tfm after bootstrap, preferably once on an internet-capable node:
  micromamba run -p $SSL_TFM_PREFIX python scripts/cluster/warm_model_cache.py

Never launch hundreds of array tasks that each download independently.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path


def _lock_path() -> Path:
    root = Path(os.environ.get("SSL_CACHE_ROOT", "/private/ofirlin-lab/suissad4/caches"))
    root.mkdir(parents=True, exist_ok=True)
    return root / ".warm_model_cache.lock"


def acquire_lock(timeout_s: float = 7200.0) -> int:
    import fcntl

    path = _lock_path()
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    start = time.time()
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.write(fd, f"pid={os.getpid()} t={time.time()}\n".encode())
            return fd
        except BlockingIOError:
            if time.time() - start > timeout_s:
                os.close(fd)
                raise TimeoutError(f"Could not acquire lock {path}")
            time.sleep(5)


def release_lock(fd: int) -> None:
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def sha256_file(path: Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        read = 0
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
            if max_bytes is not None and read >= max_bytes:
                break
    return h.hexdigest()


def warm_tabpfn() -> dict:
    if not os.environ.get("TABPFN_TOKEN"):
        raise RuntimeError("TABPFN_TOKEN is required for headless TabPFN-3 download")
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    import numpy as np
    import pandas as pd
    from tabpfn import TabPFNClassifier

    # Tiny synthetic fit forces checkpoint resolution/download.
    X = pd.DataFrame({"a": [0.0, 1.0, 0.5, 1.5], "b": [1, 0, 1, 0]})
    y = np.array([0, 1, 0, 1])
    clf = TabPFNClassifier(device="cpu")  # download can be CPU; GPU optional here
    clf.fit(X, y)
    _ = clf.predict(X)
    meta = {
        "family": "tabpfn3",
        "import": "tabpfn.TabPFNClassifier",
        "note": "checkpoint identifiers recorded by package cache under HF_HOME/TORCH_HOME",
    }
    try:
        meta["version"] = __import__("tabpfn").__version__
    except Exception:
        meta["version"] = None
    return meta


def warm_tabicl() -> dict:
    import numpy as np
    import pandas as pd
    from tabicl import TabICLClassifier

    X = pd.DataFrame({"a": [0.0, 1.0, 0.5, 1.5], "b": [1, 0, 1, 0]})
    y = np.array([0, 1, 0, 1])
    clf = TabICLClassifier(device="cpu", verbose=False)
    clf.fit(X, y)
    _ = clf.predict(X)
    meta = {
        "family": "tabicl_v2",
        "import": "tabicl.TabICLClassifier",
        "default_ckpt_hint": "tabicl-classifier-v2-20260212.ckpt",
    }
    try:
        meta["version"] = __import__("tabicl").__version__
    except Exception:
        meta["version"] = None
    return meta


def main() -> int:
    cache_root = Path(os.environ.get("SSL_CACHE_ROOT", "/private/ofirlin-lab/suissad4/caches"))
    manifest_path = cache_root / "model_cache_manifest.json"
    fd = acquire_lock()
    try:
        results = {"tabpfn": None, "tabicl": None, "errors": []}
        try:
            results["tabpfn"] = warm_tabpfn()
            print("tabpfn_warm_ok", results["tabpfn"])
        except Exception as exc:  # noqa: BLE001
            results["errors"].append(f"tabpfn: {type(exc).__name__}: {exc}")
            print("tabpfn_warm_FAILED", type(exc).__name__, str(exc)[:200], file=sys.stderr)

        try:
            results["tabicl"] = warm_tabicl()
            print("tabicl_warm_ok", results["tabicl"])
        except Exception as exc:  # noqa: BLE001
            results["errors"].append(f"tabicl: {type(exc).__name__}: {exc}")
            print("tabicl_warm_FAILED", type(exc).__name__, str(exc)[:200], file=sys.stderr)

        # Best-effort list of newly present large files under HF cache
        hf = Path(os.environ.get("HF_HOME", cache_root / "hf"))
        files = []
        if hf.exists():
            for p in hf.rglob("*"):
                if p.is_file() and p.stat().st_size > 1_000_000:
                    files.append(
                        {
                            "path": str(p),
                            "bytes": p.stat().st_size,
                            "sha256_prefix": sha256_file(p, max_bytes=8_000_000),
                        }
                    )
        results["large_cache_files"] = files[:50]
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
        os.replace(tmp, manifest_path)
        print("manifest_written", manifest_path)
        return 0 if not results["errors"] else 2
    finally:
        release_lock(fd)


if __name__ == "__main__":
    raise SystemExit(main())
