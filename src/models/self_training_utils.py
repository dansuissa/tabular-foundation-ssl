"""Shared bounded pseudo-label loop for the four classical self-trainers.

Configuration limits confidence, per-class additions, total additions, and the
number of iterations so large unlabeled pools cannot silently dominate fitting.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.models.common import as_array, empty_training_meta, stack_features

SELF_TRAINING_CONFIGS: dict[str, dict[str, Any]] = {
    "self_training_lr": {
        "threshold": 0.95,
        "max_iter": 5,
        "max_pseudo_per_iter": 2000,
        "max_pseudo_per_class_per_iter": 500,
        "min_added_fraction": 0.001,
        "max_remaining_fraction": 0.2,
        "max_total_pseudo_fraction": 0.50,
    },
    "self_training_xgboost": {
        "threshold": 0.95,
        "max_iter": 3,
        "max_pseudo_per_iter": 2000,
        "max_pseudo_per_class_per_iter": 500,
        "min_added_fraction": 0.001,
        "max_remaining_fraction": 0.2,
        "max_total_pseudo_fraction": 0.50,
    },
    "self_training_lightgbm": {
        "threshold": 0.95,
        "max_iter": 3,
        "max_pseudo_per_iter": 2000,
        "max_pseudo_per_class_per_iter": 500,
        "min_added_fraction": 0.001,
        "max_remaining_fraction": 0.2,
        "max_total_pseudo_fraction": 0.50,
    },
    "self_training_catboost": {
        "threshold": 0.97,
        "max_iter": 3,
        "max_pseudo_per_iter": 1000,
        "max_pseudo_per_class_per_iter": 250,
        "min_added_fraction": 0.001,
        "max_remaining_fraction": 0.2,
        "max_total_pseudo_fraction": 0.40,
    },
}


def fit_self_training(
    name: str,
    base_estimator_factory: Callable[[], Any],
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray | None,
    random_state: int,
) -> tuple[Any, dict[str, Any]]:
    config = SELF_TRAINING_CONFIGS[name]
    meta = empty_training_meta()

    if X_unlabeled is None or len(X_unlabeled) == 0:
        model = base_estimator_factory()
        model.fit(as_array(X_labeled), np.asarray(y_labeled))
        meta.update(
            {
                "self_training_iterations": 0.0,
                "n_pseudo_added_total": 0.0,
                "n_pseudo_added_last_iter": 0.0,
                "pseudo_label_fraction": 0.0,
                "stopped_reason": "no_unlabeled",
            }
        )
        return model, meta

    return _capped_self_training(
        base_estimator_factory=base_estimator_factory,
        X_labeled=as_array(X_labeled),
        y_labeled=np.asarray(y_labeled),
        X_unlabeled=as_array(X_unlabeled),
        config=config,
    )


def _select_pseudo_labels(
    confidence: np.ndarray,
    pseudo_labels: np.ndarray,
    threshold: float,
    max_pseudo_per_iter: int,
    max_pseudo_per_class_per_iter: int,
    max_from_remaining: int,
) -> np.ndarray:
    accept = confidence >= threshold
    if not np.any(accept):
        return np.array([], dtype=int)

    candidate_idx = np.where(accept)[0]
    candidate_conf = confidence[accept]
    candidate_labels = pseudo_labels[accept]
    order = np.argsort(-candidate_conf)
    candidate_idx = candidate_idx[order]
    candidate_labels = candidate_labels[order]

    selected: list[int] = []
    per_class: dict[Any, int] = {}
    for idx, label in zip(candidate_idx, candidate_labels):
        count = per_class.get(label, 0)
        if count >= max_pseudo_per_class_per_iter:
            continue
        selected.append(int(idx))
        per_class[label] = count + 1
        if len(selected) >= max_pseudo_per_iter:
            break
        if len(selected) >= max_from_remaining:
            break
    return np.asarray(selected, dtype=int)


def _capped_self_training(
    base_estimator_factory: Callable[[], Any],
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    X_pool = as_array(X_unlabeled)
    original_unlabeled_count = len(X_pool)
    remaining = np.ones(len(X_pool), dtype=bool)
    X_current = as_array(X_labeled)
    y_current = y_labeled.copy()
    n_pseudo_total = 0
    n_pseudo_last = 0
    iterations = 0
    stopped_reason = "max_iter"
    all_confidences: list[float] = []

    model = base_estimator_factory()
    tol = 1e-9

    for iteration in range(config["max_iter"]):
        if original_unlabeled_count > 0:
            current_fraction = n_pseudo_total / original_unlabeled_count
            if current_fraction >= config["max_total_pseudo_fraction"] - tol:
                stopped_reason = "max_total_pseudo_fraction"
                break

        model.fit(X_current, y_current)
        if not remaining.any():
            stopped_reason = "no_remaining_unlabeled"
            break

        pool_idx = np.where(remaining)[0]
        proba = model.predict_proba(X_pool[pool_idx])
        if np.any(~np.isfinite(proba)):
            raise ValueError("Self-training probabilities contain NaN or infinite values.")

        confidence = np.max(proba, axis=1)
        pseudo_labels = model.classes_[np.argmax(proba, axis=1)]
        max_from_remaining = max(
            1,
            int(np.ceil(config["max_remaining_fraction"] * remaining.sum())),
        )
        remaining_total_cap = int(
            np.floor(config["max_total_pseudo_fraction"] * original_unlabeled_count - n_pseudo_total)
        )
        if remaining_total_cap <= 0:
            stopped_reason = "max_total_pseudo_fraction"
            break
        max_from_remaining = min(max_from_remaining, remaining_total_cap)
        selected_local = _select_pseudo_labels(
            confidence=confidence,
            pseudo_labels=pseudo_labels,
            threshold=config["threshold"],
            max_pseudo_per_iter=config["max_pseudo_per_iter"],
            max_pseudo_per_class_per_iter=config["max_pseudo_per_class_per_iter"],
            max_from_remaining=max_from_remaining,
        )
        if selected_local.size == 0:
            stopped_reason = "no_candidates"
            iterations = iteration
            break

        added_fraction = selected_local.size / original_unlabeled_count
        if added_fraction < config["min_added_fraction"]:
            stopped_reason = "min_added_fraction"
            iterations = iteration
            break

        global_idx = pool_idx[selected_local]
        selected_conf = confidence[selected_local]
        all_confidences.extend(selected_conf.tolist())

        X_current = stack_features(X_current, X_pool[global_idx])
        y_current = np.concatenate([y_current, pseudo_labels[selected_local]])
        remaining[global_idx] = False
        n_pseudo_last = int(selected_local.size)
        n_pseudo_total += n_pseudo_last
        iterations = iteration + 1

        if original_unlabeled_count > 0:
            current_fraction = n_pseudo_total / original_unlabeled_count
            if current_fraction >= config["max_total_pseudo_fraction"] - tol:
                stopped_reason = "max_total_pseudo_fraction"
                break

    model.fit(X_current, y_current)
    class_counts = {
        int(label): int(count)
        for label, count in zip(*np.unique(y_current, return_counts=True))
    }
    pseudo_fraction = float(n_pseudo_total / original_unlabeled_count) if original_unlabeled_count > 0 else 0.0
    pseudo_fraction = min(pseudo_fraction, float(config["max_total_pseudo_fraction"]))

    meta = {
        "n_pseudo_added_total": float(n_pseudo_total),
        "n_pseudo_added_last_iter": float(n_pseudo_last),
        "self_training_iterations": float(iterations),
        "pseudo_label_class_distribution": str(class_counts),
        "pseudo_label_fraction": pseudo_fraction,
        "mean_selected_confidence": float(np.mean(all_confidences))
        if all_confidences
        else np.nan,
        "min_selected_confidence": float(np.min(all_confidences))
        if all_confidences
        else np.nan,
        "max_selected_confidence": float(np.max(all_confidences))
        if all_confidences
        else np.nan,
        "stopped_reason": stopped_reason,
    }
    return model, meta
