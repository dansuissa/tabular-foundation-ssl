"""Robust wrappers around scikit-learn label propagation and spreading.

The wrappers cap graph size, verify labeled-class coverage, retry a deterministic
neighbor schedule, and reject non-finite outputs while preserving failure
metadata for the benchmark record.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

from src.models.common import (
    GraphSSLMissingClassesError,
    GraphSSLNanError,
    build_graph_train_matrix,
    validate_predictions,
)

MAX_GRAPH_SSL_TRAIN_ROWS = 20000
DEFAULT_NEIGHBORS_SMALL = [7, 15, 30, 50, 5, 3]
DEFAULT_NEIGHBORS_MULTICLASS_LOWLABEL = [50, 30, 15, 7, 5, 3]


def _neighbor_schedule(n_classes: int, n_labeled: int, n_graph_rows: int) -> list[int]:
    if n_graph_rows <= 1:
        return []

    low_label = (n_labeled / max(n_classes, 1)) <= 3
    if n_classes >= 10 or low_label:
        base = DEFAULT_NEIGHBORS_MULTICLASS_LOWLABEL
    else:
        base = DEFAULT_NEIGHBORS_SMALL

    seen: set[int] = set()
    schedule: list[int] = []
    for k in base:
        k = int(min(k, n_graph_rows - 1))
        if k < 1:
            continue
        if k in seen:
            continue
        seen.add(k)
        schedule.append(k)
    return schedule


def _check_labeled_classes(y_labeled: np.ndarray, n_classes: int | None = None) -> None:
    labeled_classes = np.unique(y_labeled)
    if n_classes is not None and len(labeled_classes) < n_classes:
        raise GraphSSLMissingClassesError(
            f"Labeled set has {len(labeled_classes)} classes but {n_classes} expected."
        )
    if len(labeled_classes) < 2:
        raise GraphSSLMissingClassesError(
            f"Labeled set must contain at least 2 classes, got {len(labeled_classes)}."
        )


def _model_outputs_valid(model: Any, X_graph: np.ndarray) -> bool:
    if hasattr(model, "label_distributions_"):
        dist = np.asarray(model.label_distributions_)
        if dist.size == 0 or np.any(~np.isfinite(dist)):
            return False
    try:
        proba = model.predict_proba(X_graph)
        pred = model.predict(X_graph)
        if np.any(~np.isfinite(proba)) or np.any(~np.isfinite(pred.astype(float))):
            return False
    except Exception:
        return False
    return True


def fit_robust_graph_ssl(
    method: Literal["label_spreading", "label_propagation"],
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray | None,
    random_state: int,
    n_classes: int,
) -> tuple[Any, dict[str, Any]]:
    y_labeled = np.asarray(y_labeled)
    _check_labeled_classes(y_labeled, n_classes=n_classes)

    X_graph, y_graph, n_unlabeled_used = build_graph_train_matrix(
        X_labeled,
        y_labeled,
        X_unlabeled,
        max_graph_rows=MAX_GRAPH_SSL_TRAIN_ROWS,
        random_state=random_state,
    )
    n_graph_rows = len(X_graph)
    n_labeled = len(y_labeled)
    min_per_class = int(pd_series_min_count(y_labeled))
    tried_neighbors: list[int] = []
    last_error = "unknown"

    schedule = _neighbor_schedule(n_classes=n_classes, n_labeled=n_labeled, n_graph_rows=n_graph_rows)
    for capped_neighbors in schedule:
        tried_neighbors.append(capped_neighbors)
        try:
            if method == "label_spreading":
                model = LabelSpreading(
                    kernel="knn",
                    n_neighbors=capped_neighbors,
                    alpha=0.2,
                    max_iter=100,
                    tol=1e-3,
                )
            else:
                model = LabelPropagation(
                    kernel="knn",
                    n_neighbors=capped_neighbors,
                    max_iter=1000,
                    tol=1e-3,
                )
            model.fit(X_graph, y_graph)
            if not _model_outputs_valid(model, X_graph):
                last_error = f"invalid outputs for n_neighbors={capped_neighbors}"
                continue
            meta = {
                "graph_n_neighbors_used": capped_neighbors,
                "graph_retry_count": len(tried_neighbors) - 1,
                "graph_n_rows": n_graph_rows,
                "graph_n_unlabeled_used": n_unlabeled_used,
            }
            return model, meta
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            continue

    raise GraphSSLNanError(
        "Graph SSL failed after retries. "
        f"tried_neighbors={tried_neighbors}, n_graph_rows={n_graph_rows}, "
        f"n_labeled={n_labeled}, n_unlabeled_used={n_unlabeled_used}, "
        f"n_classes={n_classes}, min_labeled_per_class={min_per_class}, "
        f"last_error={last_error}"
    )


def pd_series_min_count(y: np.ndarray) -> int:
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min()) if len(counts) else 0


class RobustGraphSSLModel:
    def __init__(
        self,
        method: Literal["label_spreading", "label_propagation"],
        random_state: int = 0,
        n_classes: int = 2,
    ) -> None:
        self.method = method
        self.random_state = random_state
        self.n_classes = n_classes
        self.model = None
        self.training_meta: dict[str, Any] = {}

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
    ) -> "RobustGraphSSLModel":
        self.model, self.training_meta = fit_robust_graph_ssl(
            method=self.method,
            X_labeled=X_labeled,
            y_labeled=y_labeled,
            X_unlabeled=X_unlabeled,
            random_state=self.random_state,
            n_classes=self.n_classes,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        validate_predictions(y_pred, y_proba)
        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        y_proba = self.model.predict_proba(X)
        validate_predictions(self.model.predict(X), y_proba)
        return y_proba
