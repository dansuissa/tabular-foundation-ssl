"""TabICLv2 sklearn-like wrapper with KV-cache logging and embedding probes."""
from __future__ import annotations

import inspect
import logging
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

# Pinned TabICLv2 classification checkpoint (package default as of tabicl>=2.0).
DEFAULT_TABICLV2_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
PINNED_TABICL_PACKAGE = "tabicl>=2.0.3"


def _require_tabicl():
    try:
        import tabicl
        from tabicl import TabICLClassifier
    except ImportError as exc:
        raise OptionalDependencyError(
            "tabicl",
            f"Install {PINNED_TABICL_PACKAGE} in ssl-tfm.",
        ) from exc
    return tabicl, TabICLClassifier


def probe_hidden_embedding_api(clf_cls: Any) -> dict[str, Any]:
    """Inspect pinned TabICL source/API for a reliable hidden-state extraction path.

    We never call logits or raw inputs 'embeddings'. Only register an embedding
    adapter if a tested extraction path exists without altering predictions.
    """
    info: dict[str, Any] = {
        "has_get_embeddings": hasattr(clf_cls, "get_embeddings"),
        "has_extract_embeddings": hasattr(clf_cls, "extract_embeddings"),
        "candidate_methods": [],
        "embedding_api_usable": False,
        "notes": [],
    }
    for name in ("get_embeddings", "extract_embeddings", "encode", "get_representations"):
        if hasattr(clf_cls, name):
            info["candidate_methods"].append(name)
    # Conservative: only mark usable if an explicit embedding method exists.
    if info["has_get_embeddings"] or info["has_extract_embeddings"]:
        info["embedding_api_usable"] = True
        info["notes"].append("Explicit embedding method present on classifier class.")
    else:
        info["notes"].append(
            "No public hidden-embedding API found; prediction-feature adapters must be separately named."
        )
    return info


class TabICLv2Classifier:
    """Frozen TabICLv2 baseline with native tabular input."""

    name = "tabiclv2"
    input_view = "raw"

    def __init__(
        self,
        random_state: int = 0,
        n_estimators: int | None = None,
        device: str | None = "auto",
        kv_cache: bool | str = True,
        checkpoint_version: str = DEFAULT_TABICLV2_CHECKPOINT,
        allow_auto_download: bool = False,
        predict_batch_size: int | None = None,
        n_classes: int = 2,
        batch_size: int | None = None,
    ) -> None:
        self.random_state = int(random_state)
        self.n_estimators = n_estimators
        self.device_request = device
        self.kv_cache = kv_cache
        self.checkpoint_version = checkpoint_version
        self.allow_auto_download = allow_auto_download
        self.predict_batch_size = predict_batch_size
        self.n_classes = int(n_classes)
        self.batch_size = batch_size
        self.model_: Any | None = None
        self.classes_: np.ndarray | None = None
        self.training_meta: dict[str, Any] = {}
        self._timing = GPUTimingMeta()
        self._embedding_probe: dict[str, Any] = {}

    def fit(
        self,
        X_labeled: pd.DataFrame | np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: pd.DataFrame | np.ndarray | None = None,
    ) -> "TabICLv2Classifier":
        tabicl, TabICLClassifier = _require_tabicl()
        y_labeled = np.asarray(y_labeled)
        self.classes_ = np.unique(y_labeled)
        if len(self.classes_) < 2:
            raise UnsupportedMethodError(self.name, "Need >=2 classes.", "failed")

        device = resolve_torch_device(self.device_request)
        self._timing.device = device
        self._embedding_probe = probe_hidden_embedding_api(TabICLClassifier)

        kwargs: dict[str, Any] = {
            "device": device,
            "checkpoint_version": self.checkpoint_version,
            "allow_auto_download": self.allow_auto_download,
            "kv_cache": self.kv_cache,
            "random_state": self.random_state,
        }
        if self.n_estimators is not None:
            kwargs["n_estimators"] = int(self.n_estimators)
        if self.batch_size is not None:
            kwargs["batch_size"] = int(self.batch_size)

        clf = self._construct(TabICLClassifier, kwargs)
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
        kv_active = bool(getattr(clf, "kv_cache", self.kv_cache))
        offload = getattr(clf, "offloading", None) or getattr(clf, "offload", None)
        self.training_meta = {
            "backbone": "tabiclv2",
            "package_version": getattr(tabicl, "__version__", "unknown"),
            "checkpoint": self.checkpoint_version,
            "ensemble_count": getattr(clf, "n_estimators", self.n_estimators),
            "native_preprocessing": True,
            "kv_cache": kv_active,
            "offloading_mode": offload,
            "embedding_source": None,
            "embedding_api_usable": self._embedding_probe.get("embedding_api_usable"),
            "embedding_probe": self._embedding_probe,
            "cold_load_seconds": self._timing.cold_load_seconds,
            "peak_gpu_memory_mb": self._timing.peak_gpu_memory_mb,
            "device": device,
            "method_fidelity": "official",
            "reference_family": "TabICLv2 / soda-inria/tabicl",
            "protocol": "inductive",
            "uses_unlabeled_data": False,
            "pinned_package_intent": PINNED_TABICL_PACKAGE,
            "allow_auto_download": self.allow_auto_download,
        }
        return self

    def _construct(self, cls: Any, kwargs: dict[str, Any]) -> Any:
        sig = None
        try:
            sig = inspect.signature(cls.__init__)
            allowed = set(sig.parameters) - {"self"}
            filtered = {k: v for k, v in kwargs.items() if k in allowed}
        except Exception:
            filtered = dict(kwargs)
        try:
            return cls(**filtered)
        except TypeError:
            # Progressive drop
            keys = list(filtered)
            for key in keys:
                filtered.pop(key, None)
                try:
                    return cls(**filtered)
                except TypeError:
                    continue
            return cls()

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("TabICLv2Classifier is not fitted.")
        device = self._timing.device or "cpu"
        reset_peak_memory(device)
        batch = self.predict_batch_size
        try:
            with timed_section() as box:
                if batch is None or len(X) <= int(batch):
                    proba = np.asarray(self.model_.predict_proba(X))
                else:
                    proba = self._batched(X, int(batch))
            self._timing.warm_inference_seconds = box[0] if box else None
        except Exception as exc:  # noqa: BLE001
            if is_oom_error(exc) and (batch is None or batch > 8):
                new_batch = 8 if batch is None else max(1, batch // 2)
                LOGGER.warning("TabICL OOM; retry batch_size=%s", new_batch)
                self.predict_batch_size = new_batch
                return self.predict_proba(X)
            if is_oom_error(exc):
                raise TFMOOMError(self.name, str(exc)) from exc
            raise
        peak = peak_memory_mb(device)
        if peak is not None:
            self.training_meta["peak_gpu_memory_mb"] = peak
        self.training_meta["warm_inference_seconds"] = self._timing.warm_inference_seconds
        return self._align_proba(proba)

    def _batched(self, X: pd.DataFrame | np.ndarray, batch: int) -> np.ndarray:
        chunks = []
        n = len(X)
        for start in range(0, n, batch):
            end = min(n, start + batch)
            part = X.iloc[start:end] if isinstance(X, pd.DataFrame) else X[start:end]
            chunks.append(np.asarray(self.model_.predict_proba(part)))
        return np.vstack(chunks)

    def _align_proba(self, proba: np.ndarray) -> np.ndarray:
        proba = np.asarray(proba, dtype=np.float64)
        model_classes = getattr(self.model_, "classes_", None)
        if model_classes is None:
            return proba
        model_classes = np.asarray(model_classes)
        n_out = int(max(int(model_classes.max()) + 1, proba.shape[1]))
        out = np.zeros((proba.shape[0], n_out), dtype=np.float64)
        for j, cls in enumerate(model_classes):
            out[:, int(cls)] = proba[:, j]
        row_sum = out.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum <= 0, 1.0, row_sum)
        return out / row_sum

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def get_embeddings(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Not fitted.")
        if not self._embedding_probe.get("embedding_api_usable"):
            raise UnsupportedMethodError(
                self.name,
                "TabICLv2 hidden embeddings are not available on the pinned API; "
                "use a separately named prediction-feature adapter instead.",
                "unsupported_embedding_api",
            )
        for name in ("get_embeddings", "extract_embeddings"):
            fn = getattr(self.model_, name, None)
            if callable(fn):
                out = np.asarray(fn(X))
                self.training_meta["embedding_source"] = f"tabicl.{name}"
                return out
        raise UnsupportedMethodError(
            self.name, "Embedding method disappeared after probe.", "unsupported_embedding_api"
        )

    def fit_from_context(self, ctx) -> "TabICLv2Classifier":
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


def build_tabiclv2_model(random_state: int = 0, n_classes: int = 2, **kwargs: Any) -> TabICLv2Classifier:
    return TabICLv2Classifier(random_state=random_state, n_classes=n_classes, **kwargs)
