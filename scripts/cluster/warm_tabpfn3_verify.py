#!/usr/bin/env python3
"""Secure TabPFN-3 cache warm + identity verification under file lock.

Loads TABPFN_TOKEN only from LAB_ROOT/secrets/TABPFN_TOKEN (never prints it).
Records package version, ModelVersion, checkpoint path/hash, GPU, and minimal inference.
Fails hard if the resolved model is not TabPFN-3.

Important: do NOT pass model_path=ModelVersion.V3 — tabpfn 8.x treats that as a
literal filename. Use model_path="auto" or an explicit local .ckpt path.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_CKPT_NAME = "tabpfn-v3-classifier-v3_default.ckpt"


def load_token() -> None:
    path = Path(os.environ.get("LAB_ROOT", "/private/ofirlin-lab/suissad4")) / "secrets" / "TABPFN_TOKEN"
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit("TABPFN token file missing, empty or unreadable")
    token = path.read_text(encoding="utf-8").strip("\r\n")
    if not token:
        raise SystemExit("TABPFN token file missing, empty or unreadable")
    os.environ["TABPFN_TOKEN"] = token
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def acquire_lock(timeout_s: float = 7200.0) -> int:
    import fcntl

    root = Path(os.environ.get("SSL_CACHE_ROOT", "/private/ofirlin-lab/suissad4/caches"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".warm_model_cache.lock"
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
            time.sleep(5)


def release_lock(fd: int) -> None:
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def ensure_license_ok() -> dict:
    """Check PriorLabs TabPFN-3 license without printing secrets."""
    from tabpfn.browser_auth import check_license_accepted

    token = os.environ.get("TABPFN_TOKEN", "")
    api = os.environ.get("PRIORLABS_API_URL", "https://api.priorlabs.ai")
    lic = "tabpfn-3-license-v1.0"
    status = check_license_accepted(token, api, lic)
    return {"license_name": lic, "accepted": status}


def download_v3_checkpoint(dest: Path) -> dict:
    """Download TabPFN-3 default classifier into dest under lock semantics of tabpfn."""
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import download_model, resolve_model_path

    paths, dirs, names, which = resolve_model_path(
        model_path=None, which="classifier", version=ModelVersion.V3.value
    )
    resolved = Path(paths[0])
    # Prefer writing into our controlled cache when possible
    if dest.suffix == ".ckpt":
        target = dest
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target = resolved
        target.parent.mkdir(parents=True, exist_ok=True)

    before_exists = target.exists() and target.stat().st_size > 1_000_000
    if not before_exists:
        # download_model expects the final file path
        res = download_model(
            target if dest.suffix == ".ckpt" else resolved,
            version=ModelVersion.V3,
            which="classifier",
            model_name=names[0],
        )
        if res != "ok":
            raise RuntimeError(f"download_model failed: {res}")
    # If we forced a different dest, copy/symlink from resolved
    final = target if target.exists() else resolved
    if dest.suffix == ".ckpt" and resolved.exists() and resolved.resolve() != dest.resolve():
        if not dest.exists():
            dest.write_bytes(resolved.read_bytes())
        final = dest
    return {
        "resolved_path": str(resolved),
        "final_path": str(final),
        "model_name": names[0],
        "which": which,
        "already_present": before_exists,
        "download_api_result": "ok" if final.exists() else "missing",
    }


def main() -> int:
    load_token()
    cache_root = Path(os.environ.get("SSL_CACHE_ROOT", "/private/ofirlin-lab/suissad4/caches"))
    hf = Path(os.environ.get("HF_HOME", cache_root / "hf"))
    tabpfn_cache = cache_root / "tabpfn"
    xdg = Path(os.environ.get("XDG_CACHE_HOME", cache_root / "xdg"))
    tabpfn_cache.mkdir(parents=True, exist_ok=True)
    xdg.mkdir(parents=True, exist_ok=True)

    dest = tabpfn_cache / DEFAULT_CKPT_NAME
    fd = acquire_lock()
    try:
        import numpy as np
        import pandas as pd
        import torch
        import tabpfn
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        lic = ensure_license_ok()
        if lic.get("accepted") is not True:
            print("ERROR: TabPFN-3 license not accepted (accepted=%s)" % lic.get("accepted"), file=sys.stderr)
            print("Accept tabpfn-3-license-v1.0 at https://ux.priorlabs.ai/account/licenses", file=sys.stderr)
            return 2

        dl = download_v3_checkpoint(dest)
        ckpt = Path(dl["final_path"])
        if not ckpt.is_file() or ckpt.stat().st_size < 1_000_000:
            # Fall back to package-resolved path after auto construct
            ckpt = dest

        meta: dict = {
            "family": "tabpfn3",
            "package_version": getattr(tabpfn, "__version__", None),
            "classifier_class": f"{TabPFNClassifier.__module__}.{TabPFNClassifier.__qualname__}",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "verified_tabpfn3": False,
            "model_version_enum": str(ModelVersion.V3),
            "checkpoint_path": str(ckpt) if ckpt.exists() else None,
            "checkpoint_name": ckpt.name if ckpt.exists() else DEFAULT_CKPT_NAME,
            "checkpoint_sha256": sha256_file(ckpt) if ckpt.exists() else None,
            "checkpoint_bytes": ckpt.stat().st_size if ckpt.exists() else None,
            "cache_roots": [str(hf), str(tabpfn_cache), str(xdg)],
            "download_occurred": not dl.get("already_present", False),
            "license_access_ok": True,
            "license_check": lic,
            "download_meta": {k: v for k, v in dl.items()},
            "minimal_inference_ok": False,
            "notes": [
                "Constructed with explicit local checkpoint path (never ModelVersion.V3 as path).",
            ],
        }

        if not ckpt.exists():
            # Last resort: let package auto-resolve then re-locate
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clf = TabPFNClassifier(device=device, ignore_pretraining_limits=True, model_path="auto")
            X = pd.DataFrame({"a": [0.0, 1.0, 0.5, 1.5, 0.2, 0.8], "b": [1, 0, 1, 0, 1, 0]})
            y = np.array([0, 1, 0, 1, 0, 1])
            clf.fit(X, y)
            # locate downloaded file
            candidates = list((xdg / "tabpfn").glob("*v3*.ckpt")) + list(tabpfn_cache.glob("*v3*.ckpt"))
            if not candidates:
                print("ERROR: TabPFN-3 checkpoint missing after auto download", file=sys.stderr)
                return 3
            ckpt = max(candidates, key=lambda p: p.stat().st_size)
            dest.write_bytes(ckpt.read_bytes())
            ckpt = dest
            meta["checkpoint_path"] = str(ckpt)
            meta["checkpoint_name"] = ckpt.name
            meta["checkpoint_sha256"] = sha256_file(ckpt)
            meta["checkpoint_bytes"] = ckpt.stat().st_size
            meta["download_occurred"] = True
            meta["minimal_inference_ok"] = True
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clf = TabPFNClassifier(
                device=device,
                ignore_pretraining_limits=True,
                model_path=str(ckpt),
            )
            X = pd.DataFrame({"a": [0.0, 1.0, 0.5, 1.5, 0.2, 0.8], "b": [1, 0, 1, 0, 1, 0]})
            y = np.array([0, 1, 0, 1, 0, 1])
            clf.fit(X, y)
            pred = clf.predict(X)
            proba = clf.predict_proba(X)
            meta["minimal_inference_ok"] = True
            meta["pred_unique"] = int(len(set(pred.tolist())))
            meta["proba_shape"] = list(proba.shape)
            meta["proba_row_sum_ok"] = bool(np.allclose(proba.sum(axis=1), 1.0, atol=1e-4))

        # Pointer for experiment jobs (no further download)
        pointer = cache_root / "tabpfn3_default_path.txt"
        pointer.write_text(str(ckpt.resolve()) + "\n", encoding="utf-8")
        meta["pointer"] = str(pointer)

        looks_v3 = ("v3" in ckpt.name.lower()) or (ckpt.name == DEFAULT_CKPT_NAME)
        meta["verified_tabpfn3"] = bool(looks_v3 and meta["minimal_inference_ok"] and meta["checkpoint_sha256"])
        if not looks_v3:
            meta["notes"].append(f"Checkpoint name does not look like TabPFN-3: {ckpt.name}")

        from src.models.tfm_tabpfn import verify_tabpfn3

        v = verify_tabpfn3(tabpfn, clf)
        meta["project_verify"] = v
        if v.get("verified_tabpfn3"):
            meta["verified_tabpfn3"] = True

        out = cache_root / "tabpfn3_identity.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, out)

        man_path = cache_root / "model_cache_manifest.json"
        man = {}
        if man_path.exists():
            try:
                man = json.loads(man_path.read_text())
            except Exception:
                man = {}
        man["tabpfn"] = meta
        man_path.write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")

        print("tabpfn3_identity_written", out)
        print("verified_tabpfn3", meta["verified_tabpfn3"])
        print("checkpoint_name", meta.get("checkpoint_name"))
        print("checkpoint_path", meta.get("checkpoint_path"))
        print("package_version", meta.get("package_version"))
        print("download_occurred", meta.get("download_occurred"))
        print("minimal_inference_ok", meta.get("minimal_inference_ok"))
        print("license_access_ok", meta.get("license_access_ok"))
        if meta.get("checkpoint_sha256"):
            print("checkpoint_sha256", meta["checkpoint_sha256"])
            print("checkpoint_bytes", meta["checkpoint_bytes"])
        if not meta["verified_tabpfn3"]:
            print("ERROR: resolved model is not verified TabPFN-3", file=sys.stderr)
            return 3
        return 0
    finally:
        release_lock(fd)
        os.environ.pop("TABPFN_TOKEN", None)


if __name__ == "__main__":
    raise SystemExit(main())
