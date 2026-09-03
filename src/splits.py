"""Deterministic, disjoint SSL train/validation/test split construction.

The absolute label budget is sampled from the training pool before its labeled
validation holdout is created.  All remaining training examples become the
unlabeled pool; their labels are retained only for audit diagnostics and are
never exposed through the model fit context.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


class SplitError(ValueError):
    """Raised when SSL splits cannot be constructed."""


class InvalidBudgetError(SplitError):
    """Raised when n_labeled is too small to cover all classes."""


@dataclass(frozen=True)
class DataSplits:
    X_labeled_train: pd.DataFrame
    y_labeled_train: pd.Series
    X_unlabeled_train: pd.DataFrame
    y_unlabeled_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    n_labeled: int
    n_unlabeled: int
    train_labeled_size: int
    val_size: int
    test_size: int
    validation_strategy: str
    labeled_classes_present: int
    all_classes_present_in_labeled: bool
    min_labeled_per_class: int
    max_labeled_per_class: int
    train_pool_class_counts: str
    labeled_class_counts: str
    val_class_counts: str
    test_class_counts: str
    unlabeled_class_counts: str


def _assert_disjoint(*index_sets: set[int], context: str) -> None:
    seen: set[int] = set()
    for index_set in index_sets:
        overlap = seen.intersection(index_set)
        if overlap:
            raise SplitError(f"Overlapping indices detected during {context}.")
        seen.update(index_set)


def _labeled_class_stats(y_labeled: pd.Series, n_classes_total: int) -> dict:
    counts = y_labeled.value_counts()
    return {
        "labeled_classes_present": int(counts.shape[0]),
        "all_classes_present_in_labeled": bool(counts.shape[0] == n_classes_total),
        "min_labeled_per_class": int(counts.min()) if not counts.empty else 0,
        "max_labeled_per_class": int(counts.max()) if not counts.empty else 0,
    }


def _counts_json(y: pd.Series) -> str:
    """Compact JSON-ish string for class counts, stable ordering by label."""
    counts = y.value_counts().sort_index()
    # Keep it short and CSV-friendly.
    items = ",".join([f"{k}:{int(v)}" for k, v in counts.items()])
    return "{" + items + "}"


def _sample_labeled_indices(y: pd.Series, n_labeled: int, seed: int) -> np.ndarray:
    """Sample labeled indices under strict per-class minimums and proportional allocation.

    Rules (paper wave requirement):
    - The labeled subset must contain every class.
    - Minimum labeled examples per class:
        * n_labeled=50  -> min_per_class=3
        * n_labeled>=100 -> min_per_class=5  (applies to 100/250/500 here)
    - Remaining budget is allocated proportionally to class prevalence using a
      largest-remainder scheme, subject to class capacity.
    """
    y_arr = y.to_numpy()
    classes, class_counts = np.unique(y_arr, return_counts=True)
    n_classes = int(len(classes))
    if n_classes <= 1:
        raise InvalidBudgetError(f"Need at least 2 classes, got {n_classes}.")

    if n_labeled == 50:
        min_per_class = 3
    else:
        min_per_class = 5

    required_min = n_classes * min_per_class
    if n_labeled < required_min:
        raise InvalidBudgetError(
            f"n_labeled={n_labeled} is too small for n_classes={n_classes} "
            f"under min_per_class={min_per_class} (need >= {required_min})."
        )
    # Every class must have enough examples to supply its minimum.
    if int(class_counts.min()) < min_per_class:
        smallest = int(class_counts.min())
        raise InvalidBudgetError(
            f"Some class has only {smallest} examples in the train pool, "
            f"cannot satisfy min_per_class={min_per_class}."
        )

    # Start with the mandatory minimum per class.
    alloc = np.full(n_classes, min_per_class, dtype=int)
    remaining = int(n_labeled - alloc.sum())
    capacity = class_counts.astype(int) - alloc
    if np.any(capacity < 0):
        raise InvalidBudgetError("Per-class capacity is negative after minimum allocation.")

    if remaining > 0:
        # Proportional allocation by prevalence in the train pool.
        weights = class_counts.astype(float)
        weights = weights / max(weights.sum(), 1.0)
        ideal_extra = remaining * weights
        floor_extra = np.floor(ideal_extra).astype(int)
        # Apply floors, but cap by capacity.
        floor_extra = np.minimum(floor_extra, capacity)
        alloc += floor_extra
        remaining = int(n_labeled - alloc.sum())
        capacity = class_counts.astype(int) - alloc

        if remaining > 0:
            # Largest remainder for the remaining slots, respecting capacity.
            remainders = ideal_extra - np.floor(ideal_extra)
            order = np.argsort(-remainders)  # descending
            for idx in order:
                if remaining <= 0:
                    break
                if capacity[idx] <= 0:
                    continue
                alloc[idx] += 1
                capacity[idx] -= 1
                remaining -= 1

        if remaining > 0:
            # If still remaining (due to capacity caps), fill by largest remaining capacity.
            order = np.argsort(-capacity)
            for idx in order:
                if remaining <= 0:
                    break
                if capacity[idx] <= 0:
                    continue
                take = int(min(capacity[idx], remaining))
                alloc[idx] += take
                capacity[idx] -= take
                remaining -= take

        if remaining != 0:
            raise InvalidBudgetError(
                f"Could not allocate labeled budget n_labeled={n_labeled} "
                f"under per-class capacities (remaining={remaining})."
            )

    # Sample per class without replacement.
    rng = np.random.RandomState(seed)
    chosen: list[int] = []
    for cls, n_take in zip(classes, alloc):
        cls_indices = np.where(y_arr == cls)[0]
        chosen.extend(rng.choice(cls_indices, size=int(n_take), replace=False).tolist())

    if len(chosen) != n_labeled:
        raise SplitError(
            f"Internal split error: expected {n_labeled} labeled indices, got {len(chosen)}."
        )
    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=int)


def _split_labeled_validation(
    X_labeled: pd.DataFrame,
    y_labeled: pd.Series,
    val_size_from_labeled: float,
    seed: int,
    n_classes_total: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    n_labeled = len(X_labeled)
    val_count = max(1, int(round(n_labeled * val_size_from_labeled)))
    if n_labeled - val_count < n_classes_total:
        all_idx = np.arange(n_labeled, dtype=int)
        return all_idx, np.array([], dtype=int), "no_val_low_label_multiclass"

    try:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_size_from_labeled,
            random_state=seed + 2,
        )
        train_idx, val_idx = next(splitter.split(X_labeled, y_labeled))
        train_idx = np.asarray(train_idx, dtype=int)
        val_idx = np.asarray(val_idx, dtype=int)
        if len(np.unique(y_labeled.iloc[train_idx])) < n_classes_total:
            all_idx = np.arange(n_labeled, dtype=int)
            return all_idx, np.array([], dtype=int), "no_val_low_label_multiclass"
        return train_idx, val_idx, "stratified_labeled_val"
    except ValueError:
        all_idx = np.arange(n_labeled, dtype=int)
        return all_idx, np.array([], dtype=int), "no_val_low_label_multiclass"


def make_ssl_split(
    X: pd.DataFrame,
    y: pd.Series,
    n_labeled: int,
    test_size: float,
    val_size_from_labeled: float,
    seed: int,
) -> DataSplits:
    """Create SSL splits with an absolute labeled budget."""
    if n_labeled <= 0:
        raise SplitError(f"n_labeled must be positive, got {n_labeled}.")
    if not 0 < test_size < 1:
        raise SplitError(f"test_size must be in (0, 1), got {test_size}.")
    if not 0 < val_size_from_labeled < 1:
        raise SplitError(
            f"val_size_from_labeled must be in (0, 1), got {val_size_from_labeled}."
        )

    n_classes_total = int(y.nunique(dropna=True))

    train_test_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    train_pool_idx, test_idx = next(train_test_splitter.split(X, y))
    train_pool_idx = np.asarray(train_pool_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    _assert_disjoint(set(train_pool_idx), set(test_idx), context="train/test split")

    if n_labeled > len(train_pool_idx):
        raise SplitError(
            f"n_labeled={n_labeled} exceeds train pool size={len(train_pool_idx)}."
        )

    X_train_pool = X.iloc[train_pool_idx].reset_index(drop=True)
    y_train_pool = y.iloc[train_pool_idx].reset_index(drop=True)

    labeled_local_idx = _sample_labeled_indices(y_train_pool, n_labeled, seed + 1)
    labeled_set = set(labeled_local_idx.tolist())
    unlabeled_local_idx = np.array(
        [idx for idx in range(len(X_train_pool)) if idx not in labeled_set],
        dtype=int,
    )
    _assert_disjoint(
        set(labeled_local_idx),
        set(unlabeled_local_idx),
        context="labeled/unlabeled split",
    )

    X_labeled = X_train_pool.iloc[labeled_local_idx].reset_index(drop=True)
    y_labeled = y_train_pool.iloc[labeled_local_idx].reset_index(drop=True)
    X_unlabeled_train = X_train_pool.iloc[unlabeled_local_idx].reset_index(drop=True)
    y_unlabeled_train = y_train_pool.iloc[unlabeled_local_idx].reset_index(drop=True)

    labeled_train_local_idx, val_local_idx, validation_strategy = _split_labeled_validation(
        X_labeled,
        y_labeled,
        val_size_from_labeled,
        seed,
        n_classes_total,
    )
    _assert_disjoint(
        set(labeled_train_local_idx),
        set(val_local_idx),
        context="labeled train/validation split",
    )

    X_labeled_train = X_labeled.iloc[labeled_train_local_idx].reset_index(drop=True)
    y_labeled_train = y_labeled.iloc[labeled_train_local_idx].reset_index(drop=True)
    if len(val_local_idx) > 0:
        X_val = X_labeled.iloc[val_local_idx].reset_index(drop=True)
        y_val = y_labeled.iloc[val_local_idx].reset_index(drop=True)
    else:
        X_val = X_labeled.iloc[:0].copy()
        y_val = y_labeled.iloc[:0].copy()

    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    class_stats = _labeled_class_stats(y_labeled, n_classes_total)

    return DataSplits(
        X_labeled_train=X_labeled_train,
        y_labeled_train=y_labeled_train,
        X_unlabeled_train=X_unlabeled_train,
        y_unlabeled_train=y_unlabeled_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        n_labeled=n_labeled,
        n_unlabeled=len(X_unlabeled_train),
        train_labeled_size=len(X_labeled_train),
        val_size=len(X_val),
        test_size=len(X_test),
        validation_strategy=validation_strategy,
        **class_stats,
        train_pool_class_counts=_counts_json(y_train_pool),
        labeled_class_counts=_counts_json(y_labeled),
        val_class_counts=_counts_json(y_val),
        test_class_counts=_counts_json(y_test),
        unlabeled_class_counts=_counts_json(y_unlabeled_train),
    )
