"""TabPFN-3 sklearn-like wrapper with checkpoint verification and timing."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.exceptions import OptionalDependencyError, TFMOOMError, UnsupportedMethodError
from src.models.tfm_utils import (
    GPUTimingMeta,
    is_oom_error,
    peak_memory_mb,
    reset_peak_memory,
    resolve_torch_device,
    timed_section,
)

LOGGER = logging.getLogger(__name__)

# Headline default: TabPFN-3 default classifier (not specialized binary/multiclass).
DEFAULT_TABPFN3_CHECKPOINT = "tabpfn-v3-classifier-v3_default.ckpt"
SPECIALIZED_CHECKPOINTS = {
    "binary": "tabpfn-v3-classifier-v3_20260417_binary.ckpt",
    "multiclass": "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt",
}


def _require_tabpfn():
    try:
        import tabpfn
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        raise OptionalDependencyError(
            "tabpfn",
            "Install tabpfn in ssl-tfm and set TABPFN_TOKEN after accepting the PriorLabs license.",
        ) from exc
    return tabpfn, TabPFNClassifier


def resolve_default_tabpfn3_path() -> str | None:
    """Return local TabPFN-3 checkpoint path if pre-warmed; else None.

    Looks at SSL_TABPFN3_CKPT env, then cache pointer file, then standard cache name.
    Never downloads.
    """
    env = os.environ.get("SSL_TABPFN3_CKPT")
    if env and Path(env).is_file():
        return env
    cache_root = Path(os.environ.get("SSL_CACHE_ROOT", "/private/ofirlin-lab/suissad4/caches"))
    pointer = cache_root / "tabpfn3_default_path.txt"
    if pointer.is_file():
        p = pointer.read_text(encoding="utf-8").strip()
        if p and Path(p).is_file():
            return p
    candidate = cache_root / "tabpfn" / DEFAULT_TABPFN3_CHECKPOINT
    if candidate.is_file():
        return str(candidate)
    return None


def verify_tabpfn3(tabpfn_module: Any, clf: Any) -> dict[str, Any]:
    """Verify installed package resolves to TabPFN-3; do not silently accept older defaults."""
    info: dict[str, Any] = {
        "package_version": getattr(tabpfn_module, "__version__", "unknown"),
        "checkpoint": None,
        "verified_tabpfn3": False,
        "verification_notes": [],
    }
    model_path = getattr(clf, "model_path", None)
    path_str = str(model_path) if model_path is not None else ""
    local = resolve_default_tabpfn3_path()
    if local and (not path_str or path_str == local):
        path_str = local
        info["verification_notes"].append(f"resolved_local_ckpt={Path(local).name}")

    try:
        from tabpfn.constants import ModelVersion

        info["verification_notes"].append(f"ModelVersion.V3={getattr(ModelVersion, 'V3', None)}")
    except Exception as exc:  # noqa: BLE001
        info["verification_notes"].append(f"ModelVersion import failed: {exc}")

    lower = path_str.lower()
    if path_str in {"", "auto", "None"} and local:
        path_str = local
        lower = path_str.lower()
        info["verification_notes"].append(f"resolved_auto_via_local={Path(local).name}")

    if "v2" in lower and "v3" not in lower:
        info["checkpoint"] = path_str
        info["verified_tabpfn3"] = False
        info["verification_notes"].append("Checkpoint path indicates TabPFN v2 family.")
        return info

    if "v3" in lower or Path(path_str).name == DEFAULT_TABPFN3_CHECKPOINT:
        info["checkpoint"] = path_str or DEFAULT_TABPFN3_CHECKPOINT
        info["verified_tabpfn3"] = True
        info["verification_notes"].append("Checkpoint name/path contains TabPFN-3 identifier.")
        return info

    # No explicit path: refuse to claim V3 without evidence (package may still
    # resolve to V2 artifacts if V3 license/download failed).
    info["checkpoint"] = path_str or None
    info["verified_tabpfn3"] = False
    info["verification_notes"].append(
        "No explicit TabPFN-3 checkpoint path; refusing silent default. "
        "Warm caches via scripts/cluster/warm_tabpfn3_verify.py after license acceptance."
    )
    return info


class TabPFN3Classifier:
    """Frozen TabPFN-3 baseline with native pandas / mixed-type input."""

    name = "tabpfn3"
    input_view = "raw"

    def __init__(
        self,
        random_state: int = 0,
        n_estimators: int | None = None,
        device: str | None = "auto",
        model_path: str | None = None,
        specialized: str | None = None,
        allow_auto_download: bool = False,
        predict_batch_size: int | None = None,
        n_classes: int = 2,
    ) -> None:
        if specialized is not None and specialized not in SPECIALIZED_CHECKPOINTS:
            raise ValueError(
                f"Unknown specialized checkpoint key '{specialized}'. "
                f"Allowed: {sorted(SPECIALIZED_CHECKPOINTS)} (ablations only)."
            )
        self.random_state = int(random_state)
        self.n_estimators = n_estimators
        self.device_request = device
        self.model_path = model_path
        self.specialized = specialized
        self.allow_auto_download = allow_auto_download
        self.predict_batch_size = predict_batch_size
        self.n_classes = int(n_classes)
        self.model_: Any | None = None
        self.classes_: np.ndarray | None = None
        self.training_meta: dict[str, Any] = {}
        self._timing = GPUTimingMeta()

    def _build_kwargs(self) -> dict[str, Any]:
        device = resolve_torch_device(self.device_request)
        self._timing.device = device
        kwargs: dict[str, Any] = {
            "device": device,
            "ignore_pretraining_limits": True,
        }
        # Prefer no silent downloads on compute nodes.
        if hasattr(self, "allow_auto_download"):
            # Parameter name varies across versions; try both later.
            pass
        if self.n_estimators is not None:
            kwargs["n_estimators"] = int(self.n_estimators)
        if self.specialized is not None:
            kwargs["model_path"] = SPECIALIZED_CHECKPOINTS[self.specialized]
        elif self.model_path is not None:
            kwargs["model_path"] = self.model_path
        else:
            local = resolve_default_tabpfn3_path()
            if local is not None:
                kwargs["model_path"] = local
            else:
                # tabpfn 8.x default model_path="auto" resolves to
                # tabpfn-v3-classifier-v3_default.ckpt. Do NOT pass
                # ModelVersion.V3 as model_path — it is treated as a filename.
                kwargs["model_path"] = "auto"
        # Randomness / reproducibility where supported
        kwargs["random_state"] = self.random_state
        return kwargs

    def fit(
        self,
        X_labeled: pd.DataFrame | np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: pd.DataFrame | np.ndarray | None = None,
    ) -> "TabPFN3Classifier":
        tabpfn, TabPFNClassifier = _require_tabpfn()
        y_labeled = np.asarray(y_labeled)
        self.classes_ = np.unique(y_labeled)
        if len(self.classes_) < 2:
            raise UnsupportedMethodError(
                self.name, "Need at least 2 classes in labeled train.", "failed"
            )

        kwargs = self._build_kwargs()
        # Construct with graceful kwargs filtering for version drift.
        clf = self._construct_classifier(TabPFNClassifier, kwargs)
        verify = verify_tabpfn3(tabpfn, clf)
        if not verify["verified_tabpfn3"] and self.specialized is None and self.model_path is None:
            raise UnsupportedMethodError(
                self.name,
                "Installed tabpfn does not verify as TabPFN-3 default. "
                f"notes={verify['verification_notes']}",
                "unsupported_wrong_tabpfn_version",
            )

        device = self._timing.device or "cpu"
        reset_peak_memory(device)
        try:
            with timed_section() as box:
                clf.fit(X_labeled, y_labeled)
            self._timing.cold_load_seconds = box[0] if box else None
        except Exception as exc:  # noqa: BLE001
            if is_oom_error(exc):
                raise TFMOOMError(self.name, str(exc)) from exc
            raise

        self.model_ = clf
        self._timing.peak_gpu_memory_mb = peak_memory_mb(device)
        self._timing.cuda_available = device.startswith("cuda")
        self.training_meta = {
            "backbone": "tabpfn3",
            "package_version": verify["package_version"],
            "checkpoint": verify.get("checkpoint") or kwargs.get("model_path") or DEFAULT_TABPFN3_CHECKPOINT,
            "ensemble_count": getattr(clf, "n_estimators", self.n_estimators),
            "native_preprocessing": True,
            "kv_cache": None,
            "offloading_mode": None,
            "embedding_source": None,
            "cold_load_seconds": self._timing.cold_load_seconds,
            "peak_gpu_memory_mb": self._timing.peak_gpu_memory_mb,
            "device": device,
            "method_fidelity": "official",
            "reference_family": "TabPFN-3 / PriorLabs",
            "protocol": "inductive",
            "uses_unlabeled_data": False,
            "specialized_ablation": self.specialized,
            "allow_auto_download": self.allow_auto_download,
            "tabpfn_token_set": bool(os.environ.get("TABPFN_TOKEN")),
        }
        return self

    def _construct_classifier(self, TabPFNClassifier: Any, kwargs: dict[str, Any]) -> Any:
        # Drop unsupported kwargs across versions.
        try:
            return TabPFNClassifier(**kwargs)
        except TypeError:
            filtered = dict(kwargs)
            for key in list(filtered):
                try:
                    return TabPFNClassifier(**filtered)
                except TypeError:
                    filtered.pop(key, None)
            return TabPFNClassifier()

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("TabPFN3Classifier is not fitted.")
        device = self._timing.device or "cpu"
        reset_peak_memory(device)
        batch = self.predict_batch_size
        try:
            with timed_section() as box:
                if batch is None or len(X) <= int(batch):
                    proba = np.asarray(self.model_.predict_proba(X))
                else:
                    proba = self._predict_proba_batched(X, int(batch))
            self._timing.warm_inference_seconds = box[0] if box else None
        except Exception as exc:  # noqa: BLE001
            if is_oom_error(exc) and (batch is None or batch > 8):
                new_batch = 8 if batch is None else max(1, batch // 2)
                LOGGER.warning("TabPFN OOM; retrying with batch_size=%s", new_batch)
                self.predict_batch_size = new_batch
                return self.predict_proba(X)
            if is_oom_error(exc):
                raise TFMOOMError(self.name, str(exc)) from exc
            raise
        peak = peak_memory_mb(device)
        if peak is not None:
            self._timing.peak_gpu_memory_mb = peak
            self.training_meta["peak_gpu_memory_mb"] = peak
        self.training_meta["warm_inference_seconds"] = self._timing.warm_inference_seconds
        return self._align_proba_columns(proba)

    def _predict_proba_batched(self, X: pd.DataFrame | np.ndarray, batch: int) -> np.ndarray:
        chunks = []
        n = len(X)
        for start in range(0, n, batch):
            end = min(n, start + batch)
            part = X.iloc[start:end] if isinstance(X, pd.DataFrame) else X[start:end]
            chunks.append(np.asarray(self.model_.predict_proba(part)))
        return np.vstack(chunks)

    def _align_proba_columns(self, proba: np.ndarray) -> np.ndarray:
        """Ensure probability columns follow sorted class ids 0..C-1 when possible."""
        proba = np.asarray(proba, dtype=np.float64)
        model_classes = getattr(self.model_, "classes_", None)
        if model_classes is None or self.classes_ is None:
            return proba
        model_classes = np.asarray(model_classes)
        # Map model column order into 0..n_classes-1 contiguous labels used by benchmark.
        n_out = int(max(int(self.classes_.max()) + 1, proba.shape[1]))
        out = np.zeros((proba.shape[0], n_out), dtype=np.float64)
        for j, cls in enumerate(model_classes):
            out[:, int(cls)] = proba[:, j]
        # Renormalize rows that received mass.
        row_sum = out.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum <= 0, 1.0, row_sum)
        return out / row_sum

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def get_embeddings(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Official embedding extraction via tabpfn-extensions when available."""
        if self.model_ is None:
            raise RuntimeError("Not fitted.")
        try:
            from tabpfn_extensions.embedding import TabPFNEmbedding  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedMethodError(
                self.name,
                f"Official TabPFN embedding extension unavailable: {exc}",
                "unsupported_embedding_api",
            ) from exc
        emb = TabPFNEmbedding(self.model_)
        out = emb.get_embeddings(X)
        self.training_meta["embedding_source"] = "tabpfn_extensions.TabPFNEmbedding"
        return np.asarray(out)

    def fit_from_context(self, ctx) -> "TabPFN3Classifier":
        views = ctx.views
        return self.fit(views.X_labeled_raw, views.y_labeled, X_unlabeled=views.X_unlabeled_raw)

    def predict_from_context(self, ctx, *, eval_split: str = "test"):
        from src.views import PredictionResult

        views = ctx.views
        X = {
            "test": views.X_test_raw,
            "val": views.X_validation_raw,
            "validation": views.X_validation_raw,
            "labeled": views.X_labeled_raw,
            "unlabeled": views.X_unlabeled_raw,
        }[eval_split]
        return PredictionResult(
            y_pred=self.predict(X),
            y_proba=self.predict_proba(X),
            training_meta=self.training_meta,
        )

    def fit_predict_from_context(self, ctx, *, eval_split: str = "test"):
        self.fit_from_context(ctx)
        return self.predict_from_context(ctx, eval_split=eval_split)


def build_tabpfn3_model(random_state: int = 0, n_classes: int = 2, **kwargs: Any) -> TabPFN3Classifier:
    return TabPFN3Classifier(random_state=random_state, n_classes=n_classes, **kwargs)
