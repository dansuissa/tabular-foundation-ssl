"""Memory-safe feature and prediction utilities shared by classical models."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse


MAX_DENSE_FLOAT32_CELLS = 50_000_000


class GraphSSLMissingClassesError(Exception):
    """Labeled set does not contain all required classes for graph SSL."""


class GraphSSLNanError(Exception):
    """Graph SSL produced NaN/inf outputs after retries."""


def as_array(X: np.ndarray | sparse.spmatrix) -> np.ndarray | sparse.spmatrix:
    if sparse.issparse(X):
        return X
    return np.asarray(X, dtype=np.float32)


def to_dense_float32(X: np.ndarray | sparse.spmatrix) -> np.ndarray:
    X = as_array(X)
    if sparse.issparse(X):
        n_rows, n_cols = X.shape
        if n_rows * n_cols > MAX_DENSE_FLOAT32_CELLS:
            raise MemoryError(
                f"Graph SSL dense conversion too large: {n_rows} x {n_cols} cells."
            )
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if np.any(~np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def stack_features(
    X_labeled: np.ndarray | sparse.spmatrix,
    X_unlabeled: np.ndarray | sparse.spmatrix,
) -> np.ndarray | sparse.spmatrix:
    X_labeled = as_array(X_labeled)
    X_unlabeled = as_array(X_unlabeled)
    if sparse.issparse(X_labeled) or sparse.issparse(X_unlabeled):
        if not sparse.issparse(X_labeled):
            X_labeled = sparse.csr_matrix(X_labeled)
        if not sparse.issparse(X_unlabeled):
            X_unlabeled = sparse.csr_matrix(X_unlabeled)
        return sparse.vstack([X_labeled, X_unlabeled], format="csr")
    return np.vstack([X_labeled, X_unlabeled])


def subsample_unlabeled_rows(
    X_unlabeled: np.ndarray | sparse.spmatrix,
    max_rows: int,
    random_state: int,
) -> np.ndarray | sparse.spmatrix:
    X_unlabeled = as_array(X_unlabeled)
    if len(X_unlabeled) <= max_rows:
        return X_unlabeled
    rng = np.random.RandomState(random_state)
    keep_idx = rng.choice(len(X_unlabeled), size=max_rows, replace=False)
    if sparse.issparse(X_unlabeled):
        return X_unlabeled[keep_idx]
    return X_unlabeled[keep_idx]


def build_graph_train_matrix(
    X_labeled: np.ndarray | sparse.spmatrix,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray | sparse.spmatrix | None,
    max_graph_rows: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    y_labeled = np.asarray(y_labeled)
    labeled_classes = set(np.unique(y_labeled).tolist())
    if len(labeled_classes) < 2 and (-1 not in labeled_classes):
        pass

    X_labeled = as_array(X_labeled)
    if X_unlabeled is None or len(X_unlabeled) == 0:
        X_graph = to_dense_float32(X_labeled)
        return X_graph, y_labeled.copy(), 0

    X_unlabeled = as_array(X_unlabeled)
    max_unlabeled = max(max_graph_rows - len(X_labeled), 0)
    n_unlabeled_used = len(X_unlabeled)
    if max_unlabeled < len(X_unlabeled):
        X_unlabeled = subsample_unlabeled_rows(
            X_unlabeled,
            max_rows=max_unlabeled,
            random_state=random_state,
        )
        n_unlabeled_used = len(X_unlabeled)

    X_graph = stack_features(X_labeled, X_unlabeled)
    X_graph = to_dense_float32(X_graph)
    y_graph = np.concatenate(
        [y_labeled, np.full(n_unlabeled_used, -1, dtype=int)]
    )
    return X_graph, y_graph, n_unlabeled_used


def validate_predictions(y_pred: np.ndarray, y_proba: np.ndarray | None) -> None:
    if y_proba is not None and np.any(~np.isfinite(y_proba)):
        raise ValueError("Predicted probabilities contain NaN or infinite values.")
    if np.any(pd_isnan(np.asarray(y_pred, dtype=float))):
        raise ValueError("Predictions contain NaN values.")


def pd_isnan(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == object:
        return np.array([value is None for value in arr], dtype=bool)
    return np.isnan(arr.astype(float))


def empty_training_meta() -> dict[str, Any]:
    return {
        "n_pseudo_added_total": np.nan,
        "n_pseudo_added_last_iter": np.nan,
        "self_training_iterations": np.nan,
        "pseudo_label_class_distribution": np.nan,
        "pseudo_label_fraction": np.nan,
        "mean_selected_confidence": np.nan,
        "min_selected_confidence": np.nan,
        "max_selected_confidence": np.nan,
        "mean_selected_density": np.nan,
        "stopped_reason": np.nan,
    }
