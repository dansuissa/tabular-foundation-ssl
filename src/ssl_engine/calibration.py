"""Scalar temperature scaling for SSL pseudo-label probabilities.

Scientific intent
-----------------
Temperature scaling (Guo et al., 2017) is a single-parameter post-hoc
calibrator that softens or sharpens predictive distributions without
changing the argmax ranking when T > 0. It is preferred over isotonic
regression on the small validation sets typical of low-label budgets.

Leakage rules
-------------
* Fit temperature **only** on labeled validation probabilities, or on
  labeled-only out-of-fold (OOF) probabilities from StratifiedKFold.
* Never use unlabeled true labels or any test labels/features.
* Calibration must not peek at the evaluation test split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold

TEMPERATURE_BOUNDS: tuple[float, float] = (0.05, 5.0)
_EPS = 1e-12


@dataclass
class CalibrationResult:
    """Result of fitting a scalar temperature.

    Attributes
    ----------
    temperature:
        Fitted T in ``TEMPERATURE_BOUNDS``, or 1.0 if uncalibrated.
    calibration_mode:
        ``\"validation\"`` | ``\"labeled_cv\"`` | ``\"uncalibrated\"``.
    n_samples_used:
        Number of labeled rows used to fit T.
    nll:
        Held-out / OOF negative log-likelihood at the fitted T (NaN if N/A).
    meta:
        Extra diagnostics (bounds, n_splits, reason, …).
    """

    temperature: float
    calibration_mode: str
    n_samples_used: int = 0
    nll: float = float("nan")
    meta: dict[str, Any] = field(default_factory=dict)


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling via softmax in log-space.

    Computes ``softmax(log(proba) / T)`` row-wise. Equivalent to raising
    each probability to ``1/T`` then renormalizing, but numerically more
    stable for small probabilities.

    Parameters
    ----------
    proba:
        Array of shape ``(n_samples, n_classes)`` with non-negative rows
        that (approximately) sum to 1.
    temperature:
        Scalar T > 0. Values in ``[0.05, 5.0]`` are typical.

    Returns
    -------
    Calibrated probability array of the same shape.
    """
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2:
        raise ValueError(f"proba must be 2-D, got shape {proba.shape}")
    t = float(temperature)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError(f"temperature must be a positive finite scalar, got {temperature!r}")
    if abs(t - 1.0) < 1e-15:
        out = proba.copy()
        out = np.clip(out, _EPS, None)
        out /= out.sum(axis=1, keepdims=True)
        return out.astype(np.float64)

    # log-space softmax: softmax(log p / T)
    log_p = np.log(np.clip(proba, _EPS, None))
    scaled = log_p / t
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    exp_s /= exp_s.sum(axis=1, keepdims=True)
    return exp_s.astype(np.float64)


def _nll(y: np.ndarray, proba: np.ndarray) -> float:
    """Mean negative log-likelihood for integer class labels."""
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    proba = np.asarray(proba, dtype=np.float64)
    n = y.shape[0]
    if n == 0:
        return float("nan")
    rows = np.arange(n)
    p_true = np.clip(proba[rows, y], _EPS, None)
    return float(-np.mean(np.log(p_true)))


def _encode_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map arbitrary labels to 0..C-1 column indices matching proba columns.

    Assumes ``proba`` columns are ordered by ``np.unique(y)`` ascending when
    the caller constructed them consistently. Returns encoded y and the
    unique label values in column order.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    mapping = {c: i for i, c in enumerate(classes.tolist())}
    encoded = np.asarray([mapping[v] for v in y.tolist()], dtype=np.int64)
    return encoded, classes


def _align_proba_to_labels(
    y: np.ndarray,
    proba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Validate shapes and return (encoded_y, proba) or None if unusable."""
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=np.float64)
    if y.size == 0 or proba.size == 0:
        return None
    if proba.ndim != 2 or proba.shape[0] != y.shape[0]:
        return None
    if not np.all(np.isfinite(proba)):
        return None
    if proba.shape[1] < 2:
        return None
    encoded, classes = _encode_labels(y)
    if classes.size > proba.shape[1]:
        return None
    # If fewer unique labels than columns (possible under CV folds), keep
    # only columns that appear; temperature still uses full softmax width.
    if encoded.max() >= proba.shape[1]:
        return None
    # At least two classes required for meaningful calibration.
    if np.unique(encoded).size < 2:
        return None
    return encoded, proba


def fit_temperature_scaling(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    temperature_bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
) -> CalibrationResult:
    """Fit scalar temperature by minimizing NLL on ``(y_true, proba)``.

    Does **not** use unlabeled or test labels. Caller must ensure ``proba``
    was produced without those labels.
    """
    aligned = _align_proba_to_labels(y_true, proba)
    if aligned is None:
        return CalibrationResult(
            temperature=1.0,
            calibration_mode="uncalibrated",
            n_samples_used=0,
            nll=float("nan"),
            meta={"reason": "insufficient_or_invalid_inputs"},
        )

    y_enc, proba_arr = aligned
    lo, hi = float(temperature_bounds[0]), float(temperature_bounds[1])
    if not (0.0 < lo < hi):
        raise ValueError(f"Invalid temperature_bounds={temperature_bounds!r}")

    def objective(t: float) -> float:
        return _nll(y_enc, apply_temperature(proba_arr, t))

    result = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
    if not result.success or not np.isfinite(result.x):
        return CalibrationResult(
            temperature=1.0,
            calibration_mode="uncalibrated",
            n_samples_used=int(y_enc.shape[0]),
            nll=float("nan"),
            meta={"reason": "optimizer_failed", "message": str(result.message)},
        )

    t_hat = float(np.clip(result.x, lo, hi))
    nll = objective(t_hat)
    return CalibrationResult(
        temperature=t_hat,
        calibration_mode="validation",
        n_samples_used=int(y_enc.shape[0]),
        nll=float(nll),
        meta={
            "temperature_bounds": (lo, hi),
            "optimizer_success": True,
            "n_classes": int(proba_arr.shape[1]),
        },
    )


def _max_stratified_folds(y: np.ndarray, requested: int) -> int:
    """Largest feasible StratifiedKFold split count (at least 2 if possible)."""
    y = np.asarray(y)
    if y.size < 2:
        return 0
    _, counts = np.unique(y, return_counts=True)
    max_by_class = int(counts.min())
    # Need ≥2 samples per class for ≥2 folds with stratification.
    n_splits = min(int(requested), max_by_class, int(y.size))
    if n_splits < 2:
        return 0
    return n_splits


def fit_temperature(
    *,
    y_val: np.ndarray | None = None,
    proba_val: np.ndarray | None = None,
    y_labeled: np.ndarray | None = None,
    proba_oof: np.ndarray | None = None,
    X_labeled: np.ndarray | None = None,
    fold_proba_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
    n_splits: int = 5,
    random_state: int = 0,
    temperature_bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
) -> CalibrationResult:
    """Fit temperature with validation-first, then labeled-only CV fallback.

    Priority
    --------
    1. ``(y_val, proba_val)`` — preferred; labeled validation only.
    2. Precomputed ``(y_labeled, proba_oof)`` — labeled OOF probabilities.
    3. StratifiedKFold over ``(X_labeled, y_labeled)`` with ``fold_proba_fn``
       that maps ``(X_tr, y_tr, X_va) → proba_va`` (must not use test /
       unlabeled labels).
    4. Otherwise ``temperature=1.0``, ``calibration_mode=\"uncalibrated\"``.

    Leakage: never pass unlabeled or test labels into this function.
    """
    # 1) Explicit validation probabilities
    if y_val is not None and proba_val is not None:
        result = fit_temperature_scaling(
            y_val, proba_val, temperature_bounds=temperature_bounds
        )
        if result.calibration_mode != "uncalibrated":
            result.calibration_mode = "validation"
            return result
        # fall through if validation was unusable

    # 2) Precomputed OOF probabilities on labeled data
    if y_labeled is not None and proba_oof is not None:
        result = fit_temperature_scaling(
            y_labeled, proba_oof, temperature_bounds=temperature_bounds
        )
        if result.calibration_mode != "uncalibrated":
            result.calibration_mode = "labeled_cv"
            result.meta = {**result.meta, "source": "precomputed_oof"}
            return result

    # 3) Deterministic labeled-only StratifiedKFold with caller-provided fold fn
    if (
        fold_proba_fn is not None
        and X_labeled is not None
        and y_labeled is not None
    ):
        X = np.asarray(X_labeled)
        y = np.asarray(y_labeled)
        if X.shape[0] != y.shape[0]:
            return CalibrationResult(
                temperature=1.0,
                calibration_mode="uncalibrated",
                n_samples_used=0,
                meta={"reason": "X_labeled_y_labeled_length_mismatch"},
            )

        splits = _max_stratified_folds(y, n_splits)
        if splits < 2:
            return CalibrationResult(
                temperature=1.0,
                calibration_mode="uncalibrated",
                n_samples_used=int(y.shape[0]),
                meta={
                    "reason": "insufficient_for_stratified_cv",
                    "requested_n_splits": int(n_splits),
                    "feasible_n_splits": int(splits),
                },
            )

        skf = StratifiedKFold(
            n_splits=splits, shuffle=True, random_state=int(random_state)
        )
        oof = None
        for train_idx, val_idx in skf.split(X, y):
            fold_proba = np.asarray(
                fold_proba_fn(X[train_idx], y[train_idx], X[val_idx]),
                dtype=np.float64,
            )
            if fold_proba.ndim != 2 or fold_proba.shape[0] != val_idx.shape[0]:
                return CalibrationResult(
                    temperature=1.0,
                    calibration_mode="uncalibrated",
                    n_samples_used=int(y.shape[0]),
                    meta={"reason": "fold_proba_fn_bad_shape"},
                )
            if oof is None:
                oof = np.zeros((y.shape[0], fold_proba.shape[1]), dtype=np.float64)
            elif fold_proba.shape[1] != oof.shape[1]:
                return CalibrationResult(
                    temperature=1.0,
                    calibration_mode="uncalibrated",
                    n_samples_used=int(y.shape[0]),
                    meta={"reason": "fold_proba_n_classes_mismatch"},
                )
            oof[val_idx] = fold_proba

        assert oof is not None
        result = fit_temperature_scaling(
            y, oof, temperature_bounds=temperature_bounds
        )
        if result.calibration_mode != "uncalibrated":
            result.calibration_mode = "labeled_cv"
            result.meta = {
                **result.meta,
                "source": "stratified_kfold",
                "n_splits": int(splits),
                "random_state": int(random_state),
            }
            return result
        result.meta = {**result.meta, "reason": "cv_fit_failed"}
        return result

    return CalibrationResult(
        temperature=1.0,
        calibration_mode="uncalibrated",
        n_samples_used=0,
        nll=float("nan"),
        meta={"reason": "no_validation_and_no_feasible_cv"},
    )
