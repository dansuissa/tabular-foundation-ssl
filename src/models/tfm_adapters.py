"""Frozen TFM + lightweight geometric heads (Laplacian / geometric attention).

Backbones stay frozen. Identity guard falls back to frozen TFM predictions when
validation accuracy does not improve.

TabICL without a verified embedding API uses separately named
``tabiclv2_predfeat_*`` adapters — never reported as embedding adapters.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

from src.exceptions import OptionalDependencyError, UnsupportedMethodError
from src.models.torch_utils import (
    normalize_probability_matrix,
    to_dense_float32,
    validate_probability_matrix,
)

LOGGER = logging.getLogger(__name__)

TFM_ADAPTER_METHODS = {
    "tabpfn3_laplacian_adapter",
    "tabiclv2_laplacian_adapter",
    "tabpfn3_geometric_attention",
    "tabiclv2_geometric_attention",
    "tabiclv2_predfeat_laplacian_adapter",
    "tabiclv2_predfeat_geometric",
}

FeatureMode = Literal["embedding", "predfeat", "auto"]


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError(
            "torch",
            "Install torch to train TFM geometric adapters.",
        ) from exc
    return torch


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    if classes.size == 0:
        return float("nan")
    scores = []
    for c in classes:
        mask = y_true == c
        if mask.any():
            scores.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(scores)) if scores else float("nan")


class _FrozenTFMAdapterBase:
    """Shared scaffolding: freeze TFM → build features → train head → identity guard."""

    name = "tfm_adapter"
    backbone_name: str = "tabpfn3"
    head_kind: Literal["laplacian", "geometric"] = "laplacian"
    feature_mode: FeatureMode = "auto"
    is_predfeat_named: bool = False

    def __init__(self, random_state: int = 0, n_classes: int = 2, **params: Any) -> None:
        self.random_state = int(random_state)
        self.n_classes = int(n_classes)
        self.cfg = {
            "device": "cpu",
            "max_epochs": 20,
            "patience": 5,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 128,
            "lambda_lap": 0.3,
            "graph_k": 10,
            "identity_guard": True,
            "include_processed_features": True,
            "include_tfm_proba": True,
            "distill_weight": 0.1,
            **params,
        }
        self.classes_: np.ndarray | None = None
        self.tfm_ = None
        self.head_ = None
        self.training_meta: dict[str, Any] = {}
        self._use_tfm_fallback = False
        self._feature_dim: int | None = None
        self._embedding_source: str | None = None
        self._feature_mode_used: str | None = None

    def _build_tfm(self):
        if self.backbone_name == "tabpfn3":
            from src.models.tfm_tabpfn import TabPFN3Classifier

            return TabPFN3Classifier(
                random_state=self.random_state,
                n_classes=self.n_classes,
                device=self.cfg.get("tfm_device", "auto"),
            )
        if self.backbone_name == "tabiclv2":
            from src.models.tfm_tabicl import TabICLv2Classifier

            return TabICLv2Classifier(
                random_state=self.random_state,
                n_classes=self.n_classes,
                device=self.cfg.get("tfm_device", "auto"),
            )
        raise ValueError(f"Unknown backbone {self.backbone_name}")

    def _try_embeddings(self, X) -> np.ndarray | None:
        if self.tfm_ is None:
            return None
        try:
            emb = self.tfm_.get_embeddings(X)
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim == 3:
                emb = emb.mean(axis=1)
            if emb.ndim != 2:
                raise ValueError(f"Unexpected embedding shape {emb.shape}")
            self._embedding_source = self.tfm_.training_meta.get("embedding_source", "tfm.get_embeddings")
            return emb
        except Exception as exc:  # noqa: BLE001
            LOGGER.info("%s embedding extraction failed: %s", self.name, exc)
            self._embedding_source = None
            return None

    def _build_features(
        self,
        X_raw,
        X_processed: np.ndarray | None,
        force_mode: FeatureMode | None = None,
    ) -> np.ndarray:
        """Build adapter input features from frozen TFM (+ optional processed)."""
        mode = force_mode or self.feature_mode
        parts: list[np.ndarray] = []
        used = "predfeat"

        if mode in ("embedding", "auto") and not self.is_predfeat_named:
            emb = self._try_embeddings(X_raw)
            if emb is not None:
                parts.append(emb)
                used = "embedding"
            elif mode == "embedding":
                raise UnsupportedMethodError(
                    self.name,
                    "Requested embedding features but TFM embedding API is unavailable.",
                    "unsupported_embedding_api",
                )

        proba = np.asarray(self.tfm_.predict_proba(X_raw), dtype=np.float32)
        if self.cfg.get("include_tfm_proba", True) or used == "predfeat":
            parts.append(proba)

        if self.cfg.get("include_processed_features", True) and X_processed is not None:
            Xp = to_dense_float32(X_processed)
            if Xp.shape[0] != proba.shape[0]:
                raise ValueError("Processed feature row count mismatch.")
            parts.append(Xp)

        if not parts:
            parts.append(proba)
        feats = np.hstack(parts).astype(np.float32)
        self._feature_mode_used = used
        return feats

    def fit_context(self, ctx) -> "_FrozenTFMAdapterBase":
        views = ctx.views
        self.cfg = {**self.cfg, **getattr(ctx, "method_config", {})}
        return self.fit(
            views.X_labeled_raw,
            views.y_labeled,
            X_unlabeled=views.X_unlabeled_raw,
            X_val=views.X_validation_raw if views.has_validation else None,
            y_val=views.y_validation if views.has_validation else None,
            X_labeled_processed=views.X_labeled_processed,
            X_unlabeled_processed=views.X_unlabeled_processed,
            X_val_processed=views.X_validation_processed if views.has_validation else None,
        )

    def fit(
        self,
        X_labeled,
        y_labeled,
        X_unlabeled=None,
        X_val=None,
        y_val=None,
        X_labeled_processed=None,
        X_unlabeled_processed=None,
        X_val_processed=None,
    ) -> "_FrozenTFMAdapterBase":
        torch = _require_torch()
        from src.models.torch_utils import set_global_determinism

        set_global_determinism(self.random_state)

        y_labeled = np.asarray(y_labeled)
        self.classes_ = np.unique(y_labeled)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_local = np.array([class_to_idx[c] for c in y_labeled], dtype=np.int64)

        # Fit frozen TFM on labeled only.
        self.tfm_ = self._build_tfm()
        self.tfm_.fit(X_labeled, y_labeled, X_unlabeled=None)
        tfm_meta = dict(getattr(self.tfm_, "training_meta", {}))

        # Capability gate for TabICL embedding-named adapters.
        if self.backbone_name == "tabiclv2" and not self.is_predfeat_named:
            if self.feature_mode in ("embedding", "auto"):
                emb = self._try_embeddings(X_labeled)
                if emb is None:
                    raise UnsupportedMethodError(
                        self.name,
                        "TabICLv2 hidden embeddings unavailable; use "
                        "tabiclv2_predfeat_laplacian_adapter / tabiclv2_predfeat_geometric "
                        "instead. This method is not an embedding adapter fallback.",
                        "unsupported_embedding_api",
                    )

        # Build train-pool features (labeled ∪ unlabeled) — never val/test.
        X_lab_f = self._build_features(X_labeled, X_labeled_processed)
        if X_unlabeled is not None and len(X_unlabeled) > 0:
            X_unlab_f = self._build_features(X_unlabeled, X_unlabeled_processed)
            X_pool_f = np.vstack([X_lab_f, X_unlab_f]).astype(np.float32)
            has_unlab = True
        else:
            X_unlab_f = np.empty((0, X_lab_f.shape[1]), dtype=np.float32)
            X_pool_f = X_lab_f
            has_unlab = False

        # Train head on features.
        X_val_f = None
        y_val_local = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_val_f = self._build_features(X_val, X_val_processed)
            y_val_arr = np.asarray(y_val)
            if set(np.unique(y_val_arr)).issubset(set(self.classes_)):
                y_val_local = np.array([class_to_idx[c] for c in y_val_arr], dtype=np.int64)
            else:
                y_val_local = y_val_arr.astype(np.int64)

        if self.head_kind == "laplacian":
            from src.models.laplacian_ssl import LaplacianMLPSSL

            head = LaplacianMLPSSL(
                random_state=self.random_state,
                n_classes=len(self.classes_),
                max_epochs=int(self.cfg["max_epochs"]),
                patience=int(self.cfg["patience"]),
                learning_rate=float(self.cfg["learning_rate"]),
                weight_decay=float(self.cfg["weight_decay"]),
                batch_size=int(self.cfg["batch_size"]),
                lambda_lap=float(self.cfg["lambda_lap"]),
                graph_k=int(self.cfg["graph_k"]),
                device=self.cfg["device"],
                hidden_dim=int(self.cfg.get("hidden_dim", 64)),
                embedding_dim=int(self.cfg.get("embedding_dim", 32)),
            )
            head.fit(
                X_lab_f,
                y_local,
                X_unlabeled=X_unlab_f if has_unlab else None,
                X_val=X_val_f,
                y_val=y_val_local,
            )
        else:
            from src.models.geometric_attention_ssl import GeometricAttentionSSL

            head = GeometricAttentionSSL(
                random_state=self.random_state,
                n_classes=len(self.classes_),
                method_name="geometric_attention_ssl",
                max_epochs=int(self.cfg["max_epochs"]),
                patience=int(self.cfg["patience"]),
                learning_rate=float(self.cfg["learning_rate"]),
                weight_decay=float(self.cfg["weight_decay"]),
                batch_size=int(self.cfg["batch_size"]),
                device=self.cfg["device"],
                hidden_dim=int(self.cfg.get("hidden_dim", 64)),
                embedding_dim=int(self.cfg.get("embedding_dim", 32)),
            )
            head.fit(
                X_lab_f,
                y_local,
                X_unlabeled=X_unlab_f if has_unlab else None,
                X_val=X_val_f,
                y_val=y_val_local,
            )

        self.head_ = head
        self._feature_dim = X_lab_f.shape[1]
        self._use_tfm_fallback = False

        # Identity guard on validation (never test).
        guard_meta: dict[str, Any] = {"identity_guard_triggered": False}
        if (
            self.cfg.get("identity_guard", True)
            and X_val is not None
            and y_val is not None
            and len(X_val) > 0
        ):
            y_val_arr = np.asarray(y_val)
            if set(np.unique(y_val_arr)).issubset(set(self.classes_)):
                y_val_eval = y_val_arr
            else:
                # Local ints → map back via classes_
                y_val_eval = self.classes_[y_val_arr.astype(int)]

            tfm_pred = self.tfm_.predict(X_val)
            # Align TFM preds to original class labels if needed
            tfm_pred = np.asarray(tfm_pred)
            adapter_proba = self._head_predict_proba(X_val, X_val_processed)
            adapter_pred = self.classes_[np.argmax(adapter_proba, axis=1)]

            tfm_score = _balanced_accuracy(y_val_eval, tfm_pred)
            ad_score = _balanced_accuracy(y_val_eval, adapter_pred)
            guard_meta.update(
                {
                    "tfm_val_balanced_accuracy": tfm_score,
                    "adapter_val_balanced_accuracy": ad_score,
                }
            )
            if not (ad_score > tfm_score + 1e-8):
                self._use_tfm_fallback = True
                guard_meta["identity_guard_triggered"] = True
                guard_meta["fallback_reason"] = (
                    f"adapter_val={ad_score:.4f} <= tfm_val={tfm_score:.4f}"
                )

        self.training_meta = {
            "method": self.name,
            "backbone": self.backbone_name,
            "method_fidelity": "reimplementation",
            "reference_family": "TFM geometric adapter",
            "protocol": "inductive",
            "uses_unlabeled_data": has_unlab,
            "tfm_frozen": True,
            "embedding_source": self._embedding_source,
            "feature_mode": self._feature_mode_used,
            "is_predfeat_adapter": bool(self.is_predfeat_named or self._feature_mode_used == "predfeat"),
            "claims_embedding_adapter": bool(
                not self.is_predfeat_named and self._feature_mode_used == "embedding"
            ),
            "feature_dim": self._feature_dim,
            "identity_guard": guard_meta,
            "use_tfm_fallback": self._use_tfm_fallback,
            "head_meta": getattr(head, "training_meta", {}),
            "tfm_meta": tfm_meta,
            "n_labeled": int(len(y_labeled)),
            "n_unlabeled": int(len(X_unlabeled) if X_unlabeled is not None else 0),
        }
        return self

    def _head_predict_proba(self, X_raw, X_processed=None) -> np.ndarray:
        feats = self._build_features(X_raw, X_processed)
        return self.head_.predict_proba(feats)

    def predict_proba(self, X, X_processed=None) -> np.ndarray:
        if self.tfm_ is None or self.classes_ is None:
            raise RuntimeError(f"{self.name} is not fitted.")
        if self._use_tfm_fallback or self.head_ is None:
            proba = np.asarray(self.tfm_.predict_proba(X), dtype=np.float64)
        else:
            # If caller only passes processed arrays (legacy), treat as raw for TFM.
            proba = self._head_predict_proba(X, X_processed)
        # Align to len(classes_)
        n_c = len(self.classes_)
        if proba.shape[1] != n_c:
            # Take columns matching local class indices if possible
            out = np.zeros((proba.shape[0], n_c), dtype=np.float64)
            for i, c in enumerate(self.classes_):
                if int(c) < proba.shape[1]:
                    out[:, i] = proba[:, int(c)]
                elif i < proba.shape[1]:
                    out[:, i] = proba[:, i]
            proba = out
        proba = normalize_probability_matrix(proba)
        return validate_probability_matrix(proba, n_classes=n_c)

    def predict(self, X, X_processed=None) -> np.ndarray:
        proba = self.predict_proba(X, X_processed=X_processed)
        return self.classes_[np.argmax(proba, axis=1)]


class TabPFN3LaplacianAdapter(_FrozenTFMAdapterBase):
    name = "tabpfn3_laplacian_adapter"
    backbone_name = "tabpfn3"
    head_kind = "laplacian"
    feature_mode = "auto"
    is_predfeat_named = False


class TabICLv2LaplacianAdapter(_FrozenTFMAdapterBase):
    """Embedding-only TabICL Laplacian adapter (fails closed if no embeddings)."""

    name = "tabiclv2_laplacian_adapter"
    backbone_name = "tabiclv2"
    head_kind = "laplacian"
    feature_mode = "embedding"
    is_predfeat_named = False


class TabICLv2PredFeatLaplacianAdapter(_FrozenTFMAdapterBase):
    """Prediction-feature Laplacian adapter — NOT an embedding adapter."""

    name = "tabiclv2_predfeat_laplacian_adapter"
    backbone_name = "tabiclv2"
    head_kind = "laplacian"
    feature_mode = "predfeat"
    is_predfeat_named = True


class TabPFN3GeometricAttentionAdapter(_FrozenTFMAdapterBase):
    name = "tabpfn3_geometric_attention"
    backbone_name = "tabpfn3"
    head_kind = "geometric"
    feature_mode = "auto"
    is_predfeat_named = False


class TabICLv2GeometricAttentionAdapter(_FrozenTFMAdapterBase):
    name = "tabiclv2_geometric_attention"
    backbone_name = "tabiclv2"
    head_kind = "geometric"
    feature_mode = "embedding"
    is_predfeat_named = False


class TabICLv2PredFeatGeometricAdapter(_FrozenTFMAdapterBase):
    """Prediction-feature geometric adapter — NOT an embedding adapter."""

    name = "tabiclv2_predfeat_geometric"
    backbone_name = "tabiclv2"
    head_kind = "geometric"
    feature_mode = "predfeat"
    is_predfeat_named = True


def build_tfm_adapter(name: str, random_state: int = 0, n_classes: int = 2, **params: Any):
    key = name.lower().strip()
    table = {
        "tabpfn3_laplacian_adapter": TabPFN3LaplacianAdapter,
        "tabiclv2_laplacian_adapter": TabICLv2LaplacianAdapter,
        "tabiclv2_predfeat_laplacian_adapter": TabICLv2PredFeatLaplacianAdapter,
        "tabpfn3_geometric_attention": TabPFN3GeometricAttentionAdapter,
        "tabiclv2_geometric_attention": TabICLv2GeometricAttentionAdapter,
        "tabiclv2_predfeat_geometric": TabICLv2PredFeatGeometricAdapter,
    }
    if key not in table:
        raise ValueError(f"Unknown TFM adapter '{name}'. Expected one of {sorted(table)}.")
    return table[key](random_state=random_state, n_classes=n_classes, **params)
