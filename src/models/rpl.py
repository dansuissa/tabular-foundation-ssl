"""Reliability- and plausibility-gated pseudo-labeling baselines."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from src.models.common import as_array, stack_features, subsample_unlabeled_rows
from src.models.self_training_utils import _select_pseudo_labels
from src.models.supervised import build_xgboost_estimator


class RPLLogisticRegression:
    name = "rpl_lr"

    def __init__(
        self,
        random_state: int = 0,
        reliability_threshold: float = 0.90,
        plausibility_threshold: float = 0.20,
        max_rounds: int = 5,
    ) -> None:
        self.random_state = random_state
        self.reliability_threshold = reliability_threshold
        self.plausibility_threshold = plausibility_threshold
        self.max_rounds = max_rounds
        self.model = LogisticRegression(max_iter=2000, random_state=random_state)
        self.training_meta: dict = {}

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
    ) -> "RPLLogisticRegression":
        X_train = np.asarray(X_labeled)
        y_train = np.asarray(y_labeled)

        if X_unlabeled is None or len(X_unlabeled) == 0:
            self.model.fit(X_train, y_train)
            return self

        X_pool = np.asarray(X_unlabeled)
        remaining_mask = np.ones(len(X_pool), dtype=bool)

        for _ in range(self.max_rounds):
            self.model.fit(X_train, y_train)
            if not remaining_mask.any():
                break

            proba = self.model.predict_proba(X_pool[remaining_mask])
            pseudo_labels = self.model.classes_[np.argmax(proba, axis=1)]
            sorted_proba = np.sort(proba, axis=1)
            reliability = sorted_proba[:, -1]
            plausibility = sorted_proba[:, -1] - sorted_proba[:, -2]

            accept = (reliability >= self.reliability_threshold) & (
                plausibility >= self.plausibility_threshold
            )
            if not np.any(accept):
                break

            accepted_idx = np.where(remaining_mask)[0][accept]
            X_train = np.vstack([X_train, X_pool[accepted_idx]])
            y_train = np.concatenate([y_train, pseudo_labels[accept]])
            remaining_mask[accepted_idx] = False

        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class RPLLiteXGBoost:
    name = "rpl_lite_xgboost"

    def __init__(
        self,
        random_state: int = 0,
        n_classes: int = 2,
        confidence_threshold: float = 0.90,
        max_iter: int = 3,
        max_pseudo_per_iter: int = 2000,
        max_pseudo_per_class_per_iter: int = 500,
        max_density_rows: int = 30000,
        k_neighbors: int = 10,
        max_remaining_fraction: float = 0.2,
        max_total_pseudo_fraction: float = 0.50,
    ) -> None:
        self.random_state = random_state
        self.n_classes = n_classes
        self.confidence_threshold = confidence_threshold
        self.max_iter = max_iter
        self.max_pseudo_per_iter = max_pseudo_per_iter
        self.max_pseudo_per_class_per_iter = max_pseudo_per_class_per_iter
        self.max_density_rows = max_density_rows
        self.k_neighbors = k_neighbors
        self.max_remaining_fraction = max_remaining_fraction
        self.max_total_pseudo_fraction = max_total_pseudo_fraction
        self.model = None
        self.training_meta: dict = {}

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
    ) -> "RPLLiteXGBoost":
        X_train = as_array(X_labeled)
        y_train = np.asarray(y_labeled)
        n_pseudo_total = 0
        n_pseudo_last = 0
        pseudo_class_counts: Counter[int] = Counter()
        all_confidences: list[float] = []
        all_densities: list[float] = []
        stopped_reason = "max_iter"

        if X_unlabeled is None or len(X_unlabeled) == 0:
            self.model = build_xgboost_estimator(self.random_state, self.n_classes)
            self.model.fit(X_train, y_train)
            self.training_meta = {
                "n_pseudo_added_total": 0.0,
                "n_pseudo_added_last_iter": 0.0,
                "self_training_iterations": 0.0,
                "pseudo_label_class_distribution": str(dict(Counter(y_train.tolist()))),
                "pseudo_label_fraction": 0.0,
                "stopped_reason": "no_unlabeled",
            }
            return self

        X_pool = as_array(X_unlabeled)
        original_unlabeled_count = len(X_pool)
        remaining = np.ones(len(X_pool), dtype=bool)
        self.model = build_xgboost_estimator(self.random_state, self.n_classes)
        iterations = 0

        for iteration in range(self.max_iter):
            if original_unlabeled_count > 0:
                current_fraction = n_pseudo_total / original_unlabeled_count
                if current_fraction >= self.max_total_pseudo_fraction - 1e-9:
                    stopped_reason = "max_total_pseudo_fraction"
                    iterations = iteration
                    break

            self.model.fit(X_train, y_train)
            if not remaining.any():
                stopped_reason = "no_remaining_unlabeled"
                break

            pool_idx = np.where(remaining)[0]
            proba = self.model.predict_proba(X_pool[pool_idx])
            if np.any(~np.isfinite(proba)):
                raise ValueError("RPL-lite probabilities contain NaN or infinite values.")

            confidence = np.max(proba, axis=1)
            pseudo_labels = self.model.classes_[np.argmax(proba, axis=1)]
            density = self._density_scores(X_train, X_pool[pool_idx])
            selection_score = confidence * density
            order = np.argsort(-selection_score)

            max_from_remaining = max(
                1,
                int(np.ceil(self.max_remaining_fraction * remaining.sum())),
            )
            remaining_total_cap = int(
                np.floor(self.max_total_pseudo_fraction * original_unlabeled_count - n_pseudo_total)
            )
            if remaining_total_cap <= 0:
                stopped_reason = "max_total_pseudo_fraction"
                iterations = iteration
                break
            max_from_remaining = min(max_from_remaining, remaining_total_cap)
            selected_local = _select_pseudo_labels(
                confidence=confidence,
                pseudo_labels=pseudo_labels,
                threshold=self.confidence_threshold,
                max_pseudo_per_iter=self.max_pseudo_per_iter,
                max_pseudo_per_class_per_iter=self.max_pseudo_per_class_per_iter,
                max_from_remaining=max_from_remaining,
            )
            if selected_local.size == 0:
                stopped_reason = "no_candidates"
                iterations = iteration
                break

            # Re-rank accepted candidates by selection score for logging.
            selected_local = selected_local[
                np.argsort(-selection_score[selected_local])
            ]
            global_idx = pool_idx[selected_local]
            added_labels = pseudo_labels[selected_local]
            all_confidences.extend(confidence[selected_local].tolist())
            all_densities.extend(density[selected_local].tolist())

            X_train = stack_features(X_train, X_pool[global_idx])
            y_train = np.concatenate([y_train, added_labels])
            remaining[global_idx] = False
            n_pseudo_last = len(global_idx)
            n_pseudo_total += n_pseudo_last
            pseudo_class_counts.update(added_labels.tolist())
            iterations = iteration + 1

            if original_unlabeled_count > 0:
                current_fraction = n_pseudo_total / original_unlabeled_count
                if current_fraction >= self.max_total_pseudo_fraction - 1e-9:
                    stopped_reason = "max_total_pseudo_fraction"
                    break

        self.model.fit(X_train, y_train)
        self.training_meta = {
            "n_pseudo_added_total": float(n_pseudo_total),
            "n_pseudo_added_last_iter": float(n_pseudo_last),
            "self_training_iterations": float(iterations),
            "pseudo_label_class_distribution": str(dict(pseudo_class_counts)),
            "pseudo_label_fraction": float(
                min(
                    n_pseudo_total / original_unlabeled_count if original_unlabeled_count > 0 else 0.0,
                    self.max_total_pseudo_fraction,
                )
            ),
            "mean_selected_confidence": float(np.mean(all_confidences))
            if all_confidences
            else np.nan,
            "mean_selected_density": float(np.mean(all_densities))
            if all_densities
            else np.nan,
            "stopped_reason": stopped_reason,
        }
        return self

    def _density_scores(
        self,
        X_labeled: np.ndarray,
        X_unlabeled_subset: np.ndarray,
    ) -> np.ndarray:
        X_labeled = as_array(X_labeled)
        X_unlabeled_subset = as_array(X_unlabeled_subset)

        X_density_unlabeled = X_unlabeled_subset
        if len(X_labeled) + len(X_unlabeled_subset) > self.max_density_rows:
            max_unlabeled = max(self.max_density_rows - len(X_labeled), 1)
            X_density_unlabeled = subsample_unlabeled_rows(
                X_unlabeled_subset,
                max_rows=max_unlabeled,
                random_state=self.random_state,
            )

        X_density = stack_features(X_labeled, X_density_unlabeled)
        if not isinstance(X_density, np.ndarray):
            X_density = np.asarray(X_density, dtype=np.float32)
        n_neighbors = min(self.k_neighbors, len(X_density))
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
        nn.fit(X_density)

        query = np.asarray(X_unlabeled_subset, dtype=np.float32)
        distances, _ = nn.kneighbors(query)
        mean_distance = distances.mean(axis=1)
        density = 1.0 / (mean_distance + 1e-8)
        if density.max() > density.min():
            density = (density - density.min()) / (density.max() - density.min())
        else:
            density = np.ones_like(density)
        return density

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def build_rpl_model(method: str, random_state: int = 0, n_classes: int = 2):
    if method == "rpl_lr":
        return RPLLogisticRegression(random_state=random_state)
    if method == "rpl_lite_xgboost":
        return RPLLiteXGBoost(random_state=random_state, n_classes=n_classes)
    raise ValueError(f"Unknown RPL method: {method}")
