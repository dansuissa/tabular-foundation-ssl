"""CPU-friendly helpers for the neural SSL baselines.

All torch imports are performed lazily inside functions so that this module
(and ``neural_ssl``) remains importable when PyTorch is not installed. A clean
``ImportError`` is raised only when a neural method is actually trained, which
the benchmark runner records as a per-run failure row rather than crashing the
whole sweep.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split

# Guard against accidental dense blow-ups on large/sparse inputs.
MAX_DENSE_FLOAT32_CELLS = 50_000_000


def require_torch():
    """Import torch or raise a clean, actionable ImportError."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyTorch is not installed. Neural SSL baselines (sslae, vime_lite, "
            "scarf) require torch. Install it with: pip install torch"
        ) from exc
    return torch


def set_global_determinism(seed: int) -> None:
    """Seed python, numpy and torch for reproducible CPU runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch = require_torch()
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - older torch versions
        pass


def to_dense_float32(X: np.ndarray | sparse.spmatrix) -> np.ndarray:
    """Convert input to a dense float32 array, defending against NaN/inf.

    Raises MemoryError if a sparse matrix is too large to densify safely.
    """
    if sparse.issparse(X):
        n_rows, n_cols = X.shape
        if n_rows * n_cols > MAX_DENSE_FLOAT32_CELLS:
            raise MemoryError(
                f"Neural SSL dense conversion too large: {n_rows} x {n_cols} cells "
                f"(> {MAX_DENSE_FLOAT32_CELLS}). Refusing to densify."
            )
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if not np.all(np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def make_mlp(
    torch,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    depth: int,
    dropout: float,
    final_activation: bool = False,
):
    """Build a simple MLP as an ``nn.Sequential`` (no subclassing at import)."""
    nn = torch.nn
    layers: list[Any] = []
    prev = in_dim
    for _ in range(max(depth, 1)):
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(nn.ReLU())
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = hidden_dim
    layers.append(nn.Linear(prev, out_dim))
    if final_activation:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def internal_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float,
    seed: int,
    min_val_per_class: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, str]:
    """Carve a stratified validation holdout out of the labeled set.

    The shared ``run_model`` interface only passes labeled/unlabeled train data
    to neural ``fit``; there is no external validation set. Neural baselines
    therefore create an internal stratified holdout for early stopping. When a
    valid holdout cannot be formed (too few labeled rows / per-class counts),
    we fall back to ``no_val`` and the caller trains with training-loss
    patience.

    Returns ``(X_tr, y_tr, X_val, y_val, strategy)`` where strategy is one of
    ``"internal_stratified_val"`` or ``"no_val"`` (X_val/y_val are None then).
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_val = int(round(val_fraction * n_samples))

    can_split = (
        val_fraction > 0
        and n_val >= len(classes)
        and n_samples - n_val >= len(classes)
        and counts.min() >= 2 * min_val_per_class
    )
    if not can_split:
        return X, y, None, None, "no_val"

    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y,
            test_size=val_fraction,
            random_state=seed,
            stratify=y,
        )
    except ValueError:
        return X, y, None, None, "no_val"

    if len(np.unique(y_tr)) < len(classes) or len(X_val) == 0:
        return X, y, None, None, "no_val"
    return X_tr, y_tr, X_val, y_val, "internal_stratified_val"


def iterate_minibatches(
    n_rows: int,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
):
    """Yield index arrays for minibatches."""
    idx = np.arange(n_rows)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n_rows, batch_size):
        yield idx[start : start + batch_size]


def feature_shuffle_corrupt(
    torch,
    x,
    corruption_mask,
):
    """Replace masked entries with values resampled from the same feature column.

    ``x`` is a (B, D) tensor; ``corruption_mask`` is a (B, D) {0,1} tensor where
    1 marks an entry to corrupt. Replacement values are drawn from the empirical
    marginal of each feature within the batch (column-wise shuffle), matching the
    VIME/SCARF style of corruption.
    """
    batch_size = x.shape[0]
    if batch_size <= 1:
        return x.clone()
    perm = torch.stack(
        [torch.randperm(batch_size, device=x.device) for _ in range(x.shape[1])],
        dim=1,
    )
    shuffled = torch.gather(x, 0, perm)
    return x * (1.0 - corruption_mask) + shuffled * corruption_mask


def check_proba_finite(proba: np.ndarray) -> np.ndarray:
    """Raise a clean error if predicted probabilities are not finite."""
    if proba is None or not np.all(np.isfinite(proba)):
        raise ValueError("Neural model produced non-finite predicted probabilities.")
    return proba


def normalize_probability_matrix(proba: np.ndarray) -> np.ndarray:
    """Clip negatives and renormalize each row to sum to 1 in float64.

    Softmax outputs computed in float32 can sum to slightly more/less than 1
    after casting to float64, which trips sklearn's ``log_loss`` "values do not
    sum to one" check. Renormalizing here removes that warning while preserving
    the predicted class ordering.
    """
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    proba = np.clip(proba, 0.0, None)
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
    return proba / row_sums


def validate_probability_matrix(
    proba: np.ndarray,
    n_classes: int,
    sum_atol: float = 1e-6,
) -> np.ndarray:
    """Validate a probability matrix or raise a clean ValueError.

    Checks shape ``(n_samples, n_classes)``, finiteness, non-negativity, and
    that every row sums to 1 within ``sum_atol``. Returns the validated matrix.
    """
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2:
        raise ValueError(
            f"Probability matrix must be 2D, got shape {proba.shape}."
        )
    if proba.shape[1] != n_classes:
        raise ValueError(
            f"Probability matrix has {proba.shape[1]} columns, expected {n_classes}."
        )
    if not np.all(np.isfinite(proba)):
        raise ValueError("Probability matrix contains NaN or infinite values.")
    if np.any(proba < -sum_atol):
        raise ValueError("Probability matrix contains negative values.")
    row_sums = proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=sum_atol):
        raise ValueError(
            "Probability rows do not sum to 1 within tolerance "
            f"(min={row_sums.min():.6f}, max={row_sums.max():.6f})."
        )
    return proba
