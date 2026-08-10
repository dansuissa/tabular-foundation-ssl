"""Extended metrics: Brier, ECE, NLL aliases."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 15,
) -> float | None:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if y_proba.ndim != 2 or y_proba.shape[0] != len(y_true):
        return None
    conf = y_proba.max(axis=1)
    pred = y_proba.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1]) if i > 0 else (conf >= bins[i]) & (conf <= bins[i + 1])
        if not np.any(mask):
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.sum() / total) * abs(acc - avg_conf)
    return float(ece)


def brier_multiclass(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if y_proba.ndim != 2:
        return None
    n_classes = y_proba.shape[1]
    try:
        # One-hot
        eye = np.eye(n_classes)[y_true.astype(int)]
        return float(np.mean(np.sum((y_proba - eye) ** 2, axis=1)))
    except Exception:
        return None


def safe_nll(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    try:
        labels = np.arange(y_proba.shape[1])
        return float(log_loss(y_true, y_proba, labels=labels))
    except Exception:
        return None


def compute_extended_prob_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
) -> dict[str, float | None]:
    if y_proba is None or not np.all(np.isfinite(y_proba)):
        return {
            "brier": None,
            "ece": None,
            "nll": None,
        }
    y_proba = np.asarray(y_proba)
    out: dict[str, float | None] = {
        "brier": brier_multiclass(y_true, y_proba),
        "ece": expected_calibration_error(y_true, y_proba),
        "nll": safe_nll(y_true, y_proba),
    }
    # Binary Brier via sklearn when applicable
    if y_proba.shape[1] == 2:
        try:
            out["brier_binary_sklearn"] = float(
                brier_score_loss(y_true, y_proba[:, 1])
            )
        except Exception:
            out["brier_binary_sklearn"] = None
    return out
