"""Run identity, config hashing, and extended result payload builders."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.results_io.shards import make_run_id

SPLIT_PROTOCOL_VERSION = "v1_absolute_budget"
_SRC_ROOT = Path(__file__).resolve().parents[1]


def config_hash(config: dict[str, Any] | None) -> str:
    """Stable short hash of a method/run configuration dict."""
    payload = json.dumps(config or {}, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _git_info(repo_root: Path | None = None) -> tuple[str | None, bool]:
    root = repo_root or _SRC_ROOT.parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None, False
    try:
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        dirty = bool(dirty_out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        dirty = False
    return commit, dirty


def _content_hash_src(src_root: Path | None = None) -> str:
    root = src_root or _SRC_ROOT
    hasher = hashlib.sha256()
    if not root.exists():
        return hasher.hexdigest()[:16]
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        try:
            hasher.update(path.read_bytes())
        except OSError:
            continue
        hasher.update(b"\0")
    return hasher.hexdigest()[:16]


def code_version(
    repo_root: Path | None = None,
    src_root: Path | None = None,
) -> str:
    """Return ``<git_sha>`` / ``<git_sha>-dirty`` or ``nogit:<src_hash>``."""
    commit, dirty = _git_info(repo_root)
    if commit:
        return f"{commit}-dirty" if dirty else commit
    return f"nogit:{_content_hash_src(src_root)}"


def git_commit_and_dirty(repo_root: Path | None = None) -> tuple[str | None, bool]:
    return _git_info(repo_root)


def _safe_pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _cuda_fingerprint() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_available": False,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "gpu_count": 0,
    }
    try:
        import torch
    except Exception:
        return info
    info["torch_available"] = True
    info["torch_version"] = getattr(torch, "__version__", None)
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    if info["cuda_available"]:
        info["gpu_count"] = int(torch.cuda.device_count())
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            info["gpu_name"] = None
    return info


def environment_fingerprint() -> dict[str, Any]:
    """Collect a JSON-serializable environment fingerprint for shard payloads."""
    packages = {
        "numpy": _safe_pkg_version("numpy"),
        "pandas": _safe_pkg_version("pandas"),
        "scikit-learn": _safe_pkg_version("scikit-learn"),
        "torch": _safe_pkg_version("torch"),
        "tabpfn": _safe_pkg_version("tabpfn"),
        "tabicl": _safe_pkg_version("tabicl"),
        "catboost": _safe_pkg_version("catboost"),
        "xgboost": _safe_pkg_version("xgboost"),
        "lightgbm": _safe_pkg_version("lightgbm"),
    }
    cuda = _cuda_fingerprint()
    return {
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "env_name": os.environ.get("SSL_ENV_NAME")
        or os.environ.get("CONDA_DEFAULT_ENV")
        or os.environ.get("MAMBA_DEFAULT_ENV"),
        "env_prefix": os.environ.get("CONDA_PREFIX") or os.environ.get("MAMBA_ROOT_PREFIX"),
        "packages": packages,
        **cuda,
    }


def environment_fingerprint_hash(fp: dict[str, Any] | None = None) -> str:
    return config_hash(fp if fp is not None else environment_fingerprint())


def build_result_payload(
    *,
    dataset: str,
    method: str,
    seed: int,
    n_labeled: int,
    status: str,
    method_config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    training_meta: dict[str, Any] | None = None,
    capabilities: Any | None = None,
    error_message: str | None = None,
    runtime_seconds: float | None = None,
    split_meta: dict[str, Any] | None = None,
    code_ver: str | None = None,
    split_protocol_version: str = SPLIT_PROTOCOL_VERSION,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an extended shard/result payload (new waves only; never rewrite history)."""
    method_config = method_config or {}
    training_meta = dict(training_meta or {})
    metrics = dict(metrics or {})
    split_meta = dict(split_meta or {})
    env_fp = environment_fingerprint()
    commit, dirty = git_commit_and_dirty()
    cv = code_ver or code_version()
    mch = config_hash(method_config)
    run_id = make_run_id(
        dataset=dataset,
        method=method,
        seed=int(seed),
        n_labeled=int(n_labeled),
        method_config_hash=mch,
        split_protocol_version=split_protocol_version,
        code_version=cv,
    )

    cap_fields: dict[str, Any] = {}
    if capabilities is not None:
        cap_fields = {
            "method_family": getattr(capabilities, "family", None),
            "input_view": getattr(capabilities, "input_view", None),
            "uses_unlabeled_data": getattr(capabilities, "uses_unlabeled_data", None),
            "protocol": getattr(capabilities, "protocol", None),
            "device": getattr(capabilities, "device", None),
            "method_fidelity": getattr(capabilities, "fidelity", None),
            "reference_paper": getattr(capabilities, "reference_paper", None),
            "upstream_commit": getattr(capabilities, "upstream_commit", None),
            "needs_external_validation": getattr(
                capabilities, "needs_external_validation", None
            ),
            "env_name_required": getattr(capabilities, "env", None),
            "supports_predict_proba": getattr(
                capabilities, "supports_predict_proba", None
            ),
        }

    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "n_labeled": int(n_labeled),
        "status": status,
        "error_message": error_message,
        "runtime_seconds": runtime_seconds,
        "method_config_hash": mch,
        "method_config": method_config,
        "split_protocol_version": split_protocol_version,
        "code_version": cv,
        "source_tree_hash": os.environ.get("SSL_SOURCE_TREE_HASH"),
        "git_commit": commit,
        "dirty_tree": dirty,
        "env_name": env_fp.get("env_name"),
        "env_lock_hash": environment_fingerprint_hash(env_fp),
        "environment": env_fp,
        "hostname": env_fp.get("hostname"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "gpu_name": env_fp.get("gpu_name"),
        "cuda_version": env_fp.get("cuda_version"),
        # Foundation-model / timing (filled from training_meta when present)
        "backbone": training_meta.get("backbone"),
        "package_version": training_meta.get("package_version"),
        "checkpoint": training_meta.get("checkpoint")
        or training_meta.get("checkpoint_ids"),
        "checkpoint_hash": training_meta.get("checkpoint_hash"),
        "checkpoint_ids": training_meta.get("checkpoint_ids"),
        "ensemble_count": training_meta.get("ensemble_count"),
        "native_preprocessing": training_meta.get("native_preprocessing"),
        "kv_cache": training_meta.get("kv_cache"),
        "offloading_mode": training_meta.get("offloading_mode"),
        "embedding_source": training_meta.get("embedding_source"),
        "cold_model_load_seconds": training_meta.get("cold_model_load_seconds"),
        "warm_inference_seconds": training_meta.get("warm_inference_seconds"),
        "peak_gpu_mem_mb": training_meta.get("peak_gpu_mem_mb"),
        **cap_fields,
        **split_meta,
        **metrics,
        "training_meta": training_meta,
    }
    if extra:
        payload.update(extra)
    return payload
