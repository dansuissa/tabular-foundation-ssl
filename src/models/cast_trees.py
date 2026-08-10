"""CAST self-training with CatBoost and LightGBM base learners.

Scientific intent
-----------------
Reuse the shared CAST core (density + PseudoLabelEngine) for GBDT teachers.
This matches CAST's model-agnostic design (Kim et al.) rather than
maintaining a separate CAST copy per tree library.

Leakage rules
-------------
* Density fit on labeled training features only.
* Pseudo-label selection never uses unlabeled/test labels.
* Validation labels only for optional temperature calibration / early stop.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import numpy as np

from src.exceptions import OptionalDependencyError
from src.models.cast_core import CASTModifier
from src.models.common import as_array, empty_training_meta
from src.ssl_engine.pseudo_label_engine import SelectionConfig


def _require_catboost():
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise OptionalDependencyError("catboost", "Install catboost for cast_catboost.") from exc
    return CatBoostClassifier


def _require_lightgbm():
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise OptionalDependencyError("lightgbm", "Install lightgbm for cast_lightgbm.") from exc
    return LGBMClassifier


def _build_catboost(random_state: int, n_classes: int) -> Any:
    CatBoostClassifier = _require_catboost()
    loss = "Logloss" if n_classes <= 2 else "MultiClass"
    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=int(random_state),
        verbose=False,
        allow_writing_files=False,
        loss_function=loss,
    )


def _build_lightgbm(random_state: int) -> Any:
    LGBMClassifier = _require_lightgbm()
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=int(random_state),
        n_jobs=-1,
        verbose=-1,
    )


class CASTTreeSSL:
    """Iterative CAST self-training with a GBDT base learner."""

    def __init__(
        self,
        name: str,
        base_factory: Callable[[], Any],
        random_state: int = 0,
        n_classes: int = 2,
        max_iter: int = 3,
        budget: int | None = None,
        cast_alpha: float = 1.0,
        confidence_threshold: float = 0.90,
        max_total_pseudo_fraction: float = 0.5,
    ) -> None:
        self.name = name
        self.base_factory = base_factory
        self.random_state = int(random_state)
        self.n_classes = int(n_classes)
        self.max_iter = int(max_iter)
        self.budget = budget
        self.cast_alpha = float(cast_alpha)
        self.confidence_threshold = float(confidence_threshold)
        self.max_total_pseudo_fraction = float(max_total_pseudo_fraction)
        self.model: Any | None = None
        self.training_meta: dict[str, Any] = empty_training_meta()

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "CASTTreeSSL":
        X_train = as_array(X_labeled)
        y_train = np.asarray(y_labeled)
        X_lab_proc = np.asarray(X_train, dtype=np.float64)
        X_lab_proc = np.nan_to_num(X_lab_proc, nan=0.0, posinf=0.0, neginf=0.0)

        meta = empty_training_meta()
        meta.update(
            {
                "method": self.name,
                "backbone": "catboost" if "catboost" in self.name else "lightgbm",
                "protocol": "inductive",
                "uses_unlabeled_data": True,
                "method_fidelity": "paper_faithful_core",
                "reference_family": "CAST (Kim et al.) + GBDT",
                "cast_alpha": self.cast_alpha,
                "rounds": self.max_iter,
                "fallback_reason": None,
            }
        )

        sel_cfg = SelectionConfig(
            confidence_threshold=self.confidence_threshold,
            use_class_balanced=True,
            prevent_majority_domination=True,
            stricter_at_budget_50=True,
            multiplier_of_n_labeled=1.0,
            per_round_cap=2000,
            per_class_cap=500,
            mode="accumulate",
            random_state=self.random_state,
        )
        cast = CASTModifier(
            density_kwargs={"alpha": self.cast_alpha, "random_state": self.random_state},
            selection_config=sel_cfg,
        )
        cast.fit_density(X_lab_proc, y_train)

        if X_unlabeled is None or len(X_unlabeled) == 0:
            self.model = self.base_factory()
            self.model.fit(X_train, y_train)
            meta["stopped_reason"] = "no_unlabeled"
            meta["n_pseudo_added_total"] = 0.0
            self.training_meta = meta
            return self

        X_pool = as_array(X_unlabeled)
        X_pool_proc = np.nan_to_num(
            np.asarray(X_pool, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        remaining = np.ones(len(X_pool), dtype=bool)
        n_pseudo_total = 0
        all_conf: list[float] = []
        all_dens: list[float] = []
        pseudo_counts: Counter[int] = Counter()
        stopped_reason = "max_iter"
        last_n_accepted = 0
        it = -1
        budget = self.budget if self.budget is not None else len(y_train)

        self.model = self.base_factory()
        self.model.fit(X_train, y_train)

        # Optional temperature from val
        if X_val is not None and y_val is not None and len(X_val) > 0:
            proba_val = np.asarray(self.model.predict_proba(as_array(X_val)), dtype=np.float64)
            cast.calibrate(y_val=np.asarray(y_val), proba_val=proba_val, random_state=self.random_state)

        for it in range(self.max_iter):
            if n_pseudo_total / max(len(X_pool), 1) >= self.max_total_pseudo_fraction - 1e-9:
                stopped_reason = "max_total_pseudo_fraction"
                break
            if not remaining.any():
                stopped_reason = "pool_exhausted"
                break

            proba = np.asarray(self.model.predict_proba(X_pool), dtype=np.float64)
            # Align columns to 0..C-1 when model.classes_ differs
            model_classes = getattr(self.model, "classes_", None)
            if model_classes is not None:
                model_classes = np.asarray(model_classes)
                n_out = int(max(int(model_classes.max()) + 1, proba.shape[1], self.n_classes))
                aligned = np.zeros((proba.shape[0], n_out), dtype=np.float64)
                for j, cls in enumerate(model_classes):
                    aligned[:, int(cls)] = proba[:, j]
                row_sum = aligned.sum(axis=1, keepdims=True)
                row_sum = np.where(row_sum <= 0, 1.0, row_sum)
                proba = aligned / row_sum

            # Only select among remaining
            existing = ~remaining
            selection = cast.select_pseudo_labels(
                proba,
                X_pool_proc,
                n_labeled=len(y_train),
                budget=budget,
                round_idx=it,
                existing_mask=existing,
                labeled_classes=np.unique(y_train),
                classes=np.arange(proba.shape[1]),
                n_already_accepted=n_pseudo_total,
            )
            last_n_accepted = int(selection.indices.size)
            if selection.indices.size == 0:
                stopped_reason = selection.stopping_reason
                break

            idx = selection.indices
            X_train = np.vstack([X_train, X_pool[idx]])
            y_train = np.concatenate([y_train, selection.pseudo_labels])
            remaining[idx] = False
            n_pseudo_total += int(idx.size)
            all_conf.extend(selection.confidences.tolist())
            all_dens.extend(selection.densities.tolist())
            pseudo_counts.update(int(c) for c in selection.pseudo_labels.tolist())

            self.model = self.base_factory()
            self.model.fit(X_train, y_train)

        meta.update(
            {
                "n_pseudo_added_total": float(n_pseudo_total),
                "n_pseudo_added_last_iter": float(last_n_accepted),
                "self_training_iterations": float(it + 1 if n_pseudo_total else 0),
                "pseudo_label_class_distribution": str(dict(pseudo_counts)),
                "pseudo_label_fraction": float(n_pseudo_total / max(len(X_pool), 1)),
                "mean_selected_confidence": float(np.mean(all_conf)) if all_conf else np.nan,
                "min_selected_confidence": float(np.min(all_conf)) if all_conf else np.nan,
                "max_selected_confidence": float(np.max(all_conf)) if all_conf else np.nan,
                "mean_selected_density": float(np.mean(all_dens)) if all_dens else np.nan,
                "stopped_reason": stopped_reason,
                "calibration_temperature": cast.temperature,
                "calibration_mode": (
                    cast.calibration_result.calibration_mode
                    if cast.calibration_result
                    else "uncalibrated"
                ),
                "cast_density_fit_meta": dict(cast.density_adjuster.fit_meta_),  # type: ignore[union-attr]
                "true_pseudo_label_accuracy": np.nan,
                "pl_diagnostics_deferred": True,
            }
        )
        self.training_meta = meta
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.name} is not fitted.")
        return np.asarray(self.model.predict(as_array(X))).reshape(-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.name} is not fitted.")
        return np.asarray(self.model.predict_proba(as_array(X)), dtype=np.float64)


class CASTCatBoost(CASTTreeSSL):
    name = "cast_catboost"

    def __init__(self, random_state: int = 0, n_classes: int = 2, **kwargs: Any) -> None:
        super().__init__(
            name="cast_catboost",
            base_factory=lambda: _build_catboost(random_state, n_classes),
            random_state=random_state,
            n_classes=n_classes,
            **kwargs,
        )


class CASTLightGBM(CASTTreeSSL):
    name = "cast_lightgbm"

    def __init__(self, random_state: int = 0, n_classes: int = 2, **kwargs: Any) -> None:
        super().__init__(
            name="cast_lightgbm",
            base_factory=lambda: _build_lightgbm(random_state),
            random_state=random_state,
            n_classes=n_classes,
            **kwargs,
        )


def build_cast_tree_model(
    method: str,
    random_state: int = 0,
    n_classes: int = 2,
    **kwargs: Any,
) -> CASTTreeSSL:
    if method == "cast_catboost":
        return CASTCatBoost(random_state=random_state, n_classes=n_classes, **kwargs)
    if method == "cast_lightgbm":
        return CASTLightGBM(random_state=random_state, n_classes=n_classes, **kwargs)
    raise KeyError(f"Unknown CAST tree method: {method}")
