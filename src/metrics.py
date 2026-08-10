"""Core classification metrics with safe probability-dependent fallbacks."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


def _sanitize_proba(y_proba: np.ndarray | None) -> np.ndarray | None:
    if y_proba is None:
        return None
    if not np.all(np.isfinite(y_proba)):
        return None
    return y_proba


def compute_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    y_proba: np.ndarray | None,
    metric_names: list[str],
    task_type: str,
) -> dict[str, float | None]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = _sanitize_proba(y_proba)
    results: dict[str, float | None] = {}
    probability_warning = y_proba is None and any(
        metric in metric_names
        for metric in {"log_loss", "roc_auc", "roc_auc_ovr", "average_precision"}
    )

    for metric in metric_names:
        if metric == "accuracy":
            results[metric] = float(accuracy_score(y_true, y_pred))
        elif metric == "balanced_accuracy":
            results[metric] = float(balanced_accuracy_score(y_true, y_pred))
        elif metric in {"f1_macro", "macro_f1"}:
            results[metric] = float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            )
        elif metric == "log_loss":
            results[metric] = _safe_log_loss(y_true, y_proba)
        elif metric in {"roc_auc", "roc_auc_ovr"}:
            results[metric] = _safe_roc_auc(y_true, y_proba, task_type)
        elif metric == "average_precision":
            results[metric] = _safe_average_precision(y_true, y_proba, task_type)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    if probability_warning and not any(
        results.get(metric) is not None
        for metric in {"log_loss", "roc_auc", "roc_auc_ovr", "average_precision"}
        if metric in metric_names
    ):
        results["_probability_metrics_skipped"] = True

    return results


def _safe_log_loss(y_true: np.ndarray, y_proba: np.ndarray | None) -> float | None:
    if y_proba is None:
        return None
    try:
        labels = np.unique(y_true)
        return float(log_loss(y_true, y_proba, labels=labels))
    except ValueError:
        return None


def _safe_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray | None,
    task_type: str,
) -> float | None:
    if y_proba is None:
        return None

    try:
        if task_type == "binary":
            if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                scores = y_proba[:, 1]
            else:
                scores = y_proba.ravel()
            return float(roc_auc_score(y_true, scores))
        return float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return None


def _safe_average_precision(
    y_true: np.ndarray,
    y_proba: np.ndarray | None,
    task_type: str,
) -> float | None:
    if y_proba is None or task_type != "binary":
        return None
    try:
        if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
            scores = y_proba[:, 1]
        else:
            scores = y_proba.ravel()
        return float(average_precision_score(y_true, scores))
    except ValueError:
        return None


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"metric_{name}": value
        for name, value in metrics.items()
        if not name.startswith("_")
    }
