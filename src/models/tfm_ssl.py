"""Validation-guarded self-training for TabPFN-3 and TabICL v2."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss

from src.models.tfm_tabicl import TabICLv2Classifier
from src.models.tfm_tabpfn import TabPFN3Classifier
from src.ssl_engine.calibration import apply_temperature, fit_temperature
from src.ssl_engine.pseudo_label_engine import PseudoLabelEngine, SelectionConfig

TFM_SSL_METHODS = {"tabpfn3_self_training", "tabiclv2_self_training"}


def _concat_raw(X_a, X_b):
    if isinstance(X_a, pd.DataFrame) and isinstance(X_b, pd.DataFrame):
        return pd.concat([X_a, X_b], axis=0, ignore_index=True)
    return np.vstack([np.asarray(X_a), np.asarray(X_b)])


def _slice_raw(X, idx: np.ndarray):
    if isinstance(X, pd.DataFrame):
        return X.iloc[np.asarray(idx)].reset_index(drop=True)
    return np.asarray(X)[np.asarray(idx)]


def _make_tfm(backbone: str, random_state: int, n_classes: int, **kwargs):
    common = dict(random_state=random_state, n_classes=n_classes, allow_auto_download=False)
    common.update(
        {
            key: value
            for key, value in kwargs.items()
            if key in {"n_estimators", "device", "predict_batch_size", "kv_cache", "model_path", "specialized"}
        }
    )
    if backbone == "tabpfn3":
        return TabPFN3Classifier(**common)
    if backbone == "tabiclv2":
        return TabICLv2Classifier(**common)
    raise ValueError(backbone)


def _val_metrics(y_true, proba) -> dict[str, float]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = proba.argmax(axis=1)
    out = {"balanced_accuracy": float(balanced_accuracy_score(y_true, pred))}
    try:
        out["log_loss"] = float(log_loss(y_true, proba, labels=list(range(proba.shape[1]))))
    except Exception:
        out["log_loss"] = float("nan")
    return out


def _better_round(
    candidate: dict[str, float],
    best: dict[str, float],
    min_ba_gain: float,
    max_logloss_degradation: float,
) -> bool:
    ba_candidate, ba_best = candidate["balanced_accuracy"], best["balanced_accuracy"]
    ll_candidate, ll_best = candidate["log_loss"], best["log_loss"]
    if not np.isfinite(ba_candidate):
        return False
    if ba_candidate + 1e-12 < ba_best + min_ba_gain:
        return False
    if np.isfinite(ll_candidate) and np.isfinite(ll_best):
        if ll_candidate > ll_best + max_logloss_degradation and ba_candidate < ba_best + 2 * min_ba_gain:
            return False
    return True


def _selection_config(random_state: int) -> SelectionConfig:
    return SelectionConfig(
        random_state=random_state,
        confidence_threshold=0.90,
        margin_threshold=0.10,
        use_class_balanced=True,
        prevent_majority_domination=True,
        stricter_at_budget_50=True,
        multiplier_of_n_labeled=1.0,
        per_round_cap=2000,
        require_all_classes=False,
        mode="accumulate",
    )


def _calibrate(model, X_labeled, y_labeled, X_val, y_val, random_state: int):
    if X_val is not None and y_val is not None and len(X_val) > 0:
        return fit_temperature(
            y_val=np.asarray(y_val),
            proba_val=np.asarray(model.predict_proba(X_val)),
            random_state=random_state,
        )
    return fit_temperature(
        y_labeled=np.asarray(y_labeled),
        proba_oof=np.asarray(model.predict_proba(X_labeled)),
        random_state=random_state,
    )


class TFMSelfTraining:
    """Iterative hard pseudo-label self-training with validation fallback."""

    def __init__(
        self,
        backbone: str,
        name: str,
        random_state: int = 0,
        n_classes: int = 2,
        **kwargs: Any,
    ) -> None:
        self.backbone = backbone
        self.name = name
        self.random_state = int(random_state)
        self.n_classes = int(n_classes)
        self.kwargs = kwargs
        self.model = None
        self.training_meta: dict[str, Any] = {}

    def fit_from_context(self, ctx):
        views = ctx.views
        self.kwargs = {**self.kwargs, **(ctx.method_config or {})}
        return self.fit(
            views.X_labeled_raw,
            views.y_labeled,
            X_unlabeled=views.X_unlabeled_raw,
            X_val=views.X_validation_raw if views.has_validation else None,
            y_val=views.y_validation if views.has_validation else None,
            n_labeled_budget=views.n_labeled,
        )

    def predict_from_context(self, ctx, *, eval_split: str = "test"):
        from src.views import PredictionResult

        views = ctx.views
        X = {
            "test": views.X_test_raw,
            "val": views.X_validation_raw,
            "validation": views.X_validation_raw,
            "labeled": views.X_labeled_raw,
            "unlabeled": views.X_unlabeled_raw,
        }[eval_split]
        return PredictionResult(
            y_pred=self.predict(X),
            y_proba=self.predict_proba(X),
            training_meta=self.training_meta,
        )

    def fit_predict_from_context(self, ctx, *, eval_split: str = "test"):
        self.fit_from_context(ctx)
        return self.predict_from_context(ctx, eval_split=eval_split)

    def fit(
        self,
        X_labeled,
        y_labeled,
        X_unlabeled=None,
        X_val=None,
        y_val=None,
        n_labeled_budget=None,
    ):
        y_labeled = np.asarray(y_labeled)
        n_budget = int(n_labeled_budget or len(y_labeled))
        max_rounds = int(self.kwargs.get("max_loops", 3))
        min_ba_gain = float(self.kwargs.get("min_improvement_ba", 0.002))
        max_ll_degradation = float(self.kwargs.get("max_logloss_degradation", 0.02))
        engine = PseudoLabelEngine(_selection_config(self.random_state))

        initial = _make_tfm(self.backbone, self.random_state, self.n_classes, **self.kwargs)
        initial.fit(X_labeled, y_labeled)
        best_model = initial
        best_round = 0
        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        best_metrics = _val_metrics(y_val, initial.predict_proba(X_val)) if has_val else {
            "balanced_accuracy": -1.0,
            "log_loss": float("inf"),
        }
        round_logs = [{"round": 0, "n_pseudo_new": 0, "val": best_metrics}]

        if X_unlabeled is None or len(X_unlabeled) == 0:
            self.model = initial
            self.training_meta = {
                "backbone": self.backbone,
                "method_fidelity": "novel_experimental",
                "reference_family": "validation-guarded hard-label TFM self-training",
                "protocol": "inductive",
                "uses_unlabeled_data": False,
                "selected_round": 0,
                "rounds_attempted": 0,
                "n_pseudo_added_total": 0,
                "stopped_reason": "no_unlabeled",
                "fallback_reason": "round0",
                "round_logs": round_logs,
            }
            return self

        X_context = X_labeled
        y_context = y_labeled.copy()
        remaining = np.ones(len(X_unlabeled), dtype=bool)
        accepted_total = 0
        current = initial
        stopped_reason = "max_rounds"

        for round_idx in range(1, max_rounds + 1):
            if not remaining.any():
                stopped_reason = "no_remaining_unlabeled"
                break

            pool_idx = np.where(remaining)[0]
            proba = np.asarray(current.predict_proba(_slice_raw(X_unlabeled, pool_idx)))
            calibration = _calibrate(
                current,
                X_context,
                y_context,
                X_val,
                y_val,
                self.random_state + round_idx,
            )
            proba = apply_temperature(proba, calibration.temperature)
            selection = engine.select(
                proba,
                n_labeled=len(y_labeled),
                budget=n_budget,
                labeled_classes=np.unique(y_labeled),
                n_already_accepted=accepted_total,
                round_idx=round_idx,
            )

            if len(selection.indices) == 0:
                stopped_reason = selection.stopping_reason or "zero_pseudo_labels"
                round_logs.append(
                    {"round": round_idx, "n_pseudo_new": 0, "val": None, "stopped": stopped_reason}
                )
                break

            global_idx = pool_idx[selection.indices]
            X_context = _concat_raw(X_context, _slice_raw(X_unlabeled, global_idx))
            y_context = np.concatenate(
                [y_context, selection.pseudo_labels.astype(y_labeled.dtype)]
            )
            remaining[global_idx] = False
            accepted_total += len(global_idx)

            candidate = _make_tfm(
                self.backbone,
                self.random_state + round_idx,
                self.n_classes,
                **self.kwargs,
            )
            candidate.fit(X_context, y_context)
            metrics = _val_metrics(y_val, candidate.predict_proba(X_val)) if has_val else best_metrics

            if not has_val or _better_round(metrics, best_metrics, min_ba_gain, max_ll_degradation):
                best_model = candidate
                best_round = round_idx
                best_metrics = metrics

            round_logs.append(
                {
                    "round": round_idx,
                    "n_pseudo_new": int(len(global_idx)),
                    "val": metrics,
                    "calibration_temperature": calibration.temperature,
                    "accepted_by_class": selection.accepted_by_class,
                }
            )
            current = candidate

        self.model = best_model
        self.training_meta = {
            "backbone": self.backbone,
            "method_fidelity": "novel_experimental",
            "reference_family": "validation-guarded hard-label TFM self-training",
            "protocol": "inductive",
            "uses_unlabeled_data": True,
            "rounds_attempted": len(round_logs) - 1,
            "selected_round": best_round,
            "round_logs": round_logs,
            "n_pseudo_added_total": accepted_total if best_round > 0 else 0,
            "stopped_reason": stopped_reason,
            "fallback_reason": "round0" if best_round == 0 else None,
            "min_improvement_ba": min_ba_gain,
            "max_logloss_degradation": max_ll_degradation,
        }
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError(f"{self.name} is not fitted.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError(f"{self.name} is not fitted.")
        return self.model.predict_proba(X)


def build_tfm_ssl_model(
    method: str,
    random_state: int = 0,
    n_classes: int = 2,
    **kwargs: Any,
):
    if method == "tabpfn3_self_training":
        return TFMSelfTraining("tabpfn3", method, random_state, n_classes, **kwargs)
    if method == "tabiclv2_self_training":
        return TFMSelfTraining("tabiclv2", method, random_state, n_classes, **kwargs)
    raise ValueError(f"Unknown TFM self-training method: {method}")
