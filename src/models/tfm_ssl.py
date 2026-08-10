"""TFM pseudo-label / self-training / consensus / teacher-student methods.

Hard leakage rules enforced by construction:
- unlabeled true labels never enter training;
- test features/labels never enter adaptation (except named transductive method);
- validation labels only for calibration / risk control / guarded fallback.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss

from src.exceptions import UnsupportedMethodError
from src.models.cast_core import CASTModifier
from src.models.tfm_tabicl import TabICLv2Classifier
from src.models.tfm_tabpfn import TabPFN3Classifier
from src.ssl_engine.calibration import apply_temperature, fit_temperature
from src.ssl_engine.diagnostics import post_hoc_pseudo_label_quality
from src.ssl_engine.pseudo_label_engine import PseudoLabelEngine, SelectionConfig

TFM_SSL_METHODS = {
    "tabpfn3_self_training",
    "tabiclv2_self_training",
    "tabpfn3_pl_one_shot",
    "tabiclv2_pl_one_shot",
    "tabpfn3_loop_risk",
    "tabiclv2_loop_risk",
    "tabpfn3_cast",
    "tabiclv2_cast",
    "tfm_consensus_context_tabiclv2",
    "tabpfn3_teacher_catboost",
    "tabiclv2_teacher_catboost",
    "tfm_consensus_catboost",
    "tabpfn3_unlabeled_prior_adjustment",
    "tabpfn3_distpfn_transductive",
}


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
    common.update({k: v for k, v in kwargs.items() if k in {
        "n_estimators", "device", "predict_batch_size", "kv_cache", "model_path", "specialized"
    }})
    if backbone == "tabpfn3":
        return TabPFN3Classifier(**common)
    if backbone == "tabiclv2":
        return TabICLv2Classifier(**common)
    raise ValueError(backbone)


def _val_metrics(y_true, proba, classes) -> dict[str, float]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = proba.argmax(axis=1)
    out = {"balanced_accuracy": float(balanced_accuracy_score(y_true, pred))}
    try:
        out["log_loss"] = float(log_loss(y_true, proba, labels=list(range(proba.shape[1]))))
    except Exception:
        out["log_loss"] = float("nan")
    return out


def _better_loop(cand: dict[str, float], best: dict[str, float], eps_ba: float, eps_ll: float) -> bool:
    ba_c, ba_b = cand["balanced_accuracy"], best["balanced_accuracy"]
    ll_c, ll_b = cand["log_loss"], best["log_loss"]
    if not np.isfinite(ba_c):
        return False
    if ba_c + 1e-12 < ba_b + eps_ba:
        return False
    if np.isfinite(ll_c) and np.isfinite(ll_b):
        if ll_c > ll_b + eps_ll and ba_c < ba_b + 2 * eps_ba:
            return False
    return True


def _default_selection_config(random_state: int, **overrides) -> SelectionConfig:
    cfg = SelectionConfig(
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
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _calibrate_teacher(model, X_lab, y_lab, X_val, y_val, random_state: int):
    if X_val is not None and y_val is not None and len(X_val) > 0 and len(y_val) > 0:
        return fit_temperature(
            y_val=np.asarray(y_val),
            proba_val=np.asarray(model.predict_proba(X_val)),
            random_state=random_state,
        )
    return fit_temperature(
        y_labeled=np.asarray(y_lab),
        proba_oof=np.asarray(model.predict_proba(X_lab)),
        random_state=random_state,
    )


class _TFMContextSSL:
    name = "tfm_ssl"
    backbone = "tabpfn3"

    def __init__(self, random_state: int = 0, n_classes: int = 2, **kwargs: Any) -> None:
        self.random_state = int(random_state)
        self.n_classes = int(n_classes)
        self.kwargs = kwargs
        self.model = None
        self.training_meta: dict[str, Any] = {}

    def fit_from_context(self, ctx):
        views = ctx.views
        cfg = {**self.kwargs, **(ctx.method_config or {})}
        self.kwargs = cfg
        return self.fit(
            views.X_labeled_raw,
            views.y_labeled,
            X_unlabeled=views.X_unlabeled_raw,
            X_val=views.X_validation_raw if views.has_validation else None,
            y_val=views.y_validation if views.has_validation else None,
            n_labeled_budget=views.n_labeled,
            y_unlabeled_hidden=None,
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

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class TFMOneShotPL(_TFMContextSSL):
    def __init__(self, backbone: str, name: str, random_state: int = 0, n_classes: int = 2, **kwargs):
        super().__init__(random_state=random_state, n_classes=n_classes, **kwargs)
        self.backbone = backbone
        self.name = name

    def fit(self, X_labeled, y_labeled, X_unlabeled=None, X_val=None, y_val=None, n_labeled_budget=None, y_unlabeled_hidden=None):
        y_labeled = np.asarray(y_labeled)
        n_budget = int(n_labeled_budget or len(y_labeled))
        teacher = _make_tfm(self.backbone, self.random_state, self.n_classes, **self.kwargs)
        teacher.fit(X_labeled, y_labeled)
        meta = {
            "backbone": self.backbone,
            "method_fidelity": "novel_experimental",
            "reference_family": "TFM one-shot pseudo-label context",
            "protocol": "inductive",
            "uses_unlabeled_data": True,
            "rounds_attempted": 1,
            "selected_round": 1,
        }
        for k in ("package_version", "checkpoint", "ensemble_count", "cold_load_seconds", "peak_gpu_memory_mb", "device", "kv_cache"):
            if teacher.training_meta and k in teacher.training_meta:
                meta[k] = teacher.training_meta[k]
        if X_unlabeled is None or len(X_unlabeled) == 0:
            self.model = teacher
            meta.update({"n_pseudo_added_total": 0, "stopped_reason": "no_unlabeled", "selected_round": 0})
            self.training_meta = meta
            return self

        proba_u = np.asarray(teacher.predict_proba(X_unlabeled))
        cal = _calibrate_teacher(teacher, X_labeled, y_labeled, X_val, y_val, self.random_state)
        proba_u = apply_temperature(proba_u, cal.temperature)
        engine = PseudoLabelEngine(_default_selection_config(self.random_state))
        sel = engine.select(
            proba_u,
            X=None,
            n_labeled=len(y_labeled),
            budget=n_budget,
            labeled_classes=np.unique(y_labeled),
        )
        meta.update({
            "calibration_temperature": cal.temperature,
            "calibration_mode": cal.calibration_mode,
            "n_pseudo_added_total": int(len(sel.indices)),
            "stopped_reason": sel.stopping_reason,
            "accepted_by_class": sel.accepted_by_class,
            "thresholds": sel.thresholds_used,
            "mean_selected_confidence": float(sel.confidences.mean()) if len(sel.indices) else np.nan,
        })
        if len(sel.indices) == 0:
            self.model = teacher
            meta["selected_round"] = 0
            meta["fallback_reason"] = "zero_pseudo_labels"
            self.training_meta = meta
            return self

        X_ctx = _concat_raw(X_labeled, _slice_raw(X_unlabeled, sel.indices))
        y_ctx = np.concatenate([y_labeled, sel.pseudo_labels.astype(y_labeled.dtype)])
        final = _make_tfm(self.backbone, self.random_state, self.n_classes, **self.kwargs)
        final.fit(X_ctx, y_ctx)
        self.model = final
        meta["fallback_reason"] = None
        meta["final_context_size"] = int(len(y_ctx))
        if y_unlabeled_hidden is not None:
            meta["post_hoc_pl_quality"] = post_hoc_pseudo_label_quality(
                np.asarray(y_unlabeled_hidden), sel.indices, sel.pseudo_labels, sel.confidences
            )
        self.training_meta = meta
        return self


class TFMLoopRisk(_TFMContextSSL):
    """Hard-label risk-controlled looping (NOT a soft-label LoopTabFM reproduction)."""

    def __init__(self, backbone: str, name: str, random_state: int = 0, n_classes: int = 2, **kwargs):
        super().__init__(random_state=random_state, n_classes=n_classes, **kwargs)
        self.backbone = backbone
        self.name = name

    def fit(self, X_labeled, y_labeled, X_unlabeled=None, X_val=None, y_val=None, n_labeled_budget=None, y_unlabeled_hidden=None):
        y_labeled = np.asarray(y_labeled)
        n_budget = int(n_labeled_budget or len(y_labeled))
        max_loops = int(self.kwargs.get("max_loops", 3))
        eps_ba = float(self.kwargs.get("min_improvement_ba", 0.002))
        eps_ll = float(self.kwargs.get("max_logloss_degradation", 0.02))
        engine = PseudoLabelEngine(_default_selection_config(self.random_state, mode="accumulate"))

        model0 = _make_tfm(self.backbone, self.random_state, self.n_classes, **self.kwargs)
        model0.fit(X_labeled, y_labeled)
        loop_logs = []
        best_model = model0
        best_round = 0
        best_metrics = {"balanced_accuracy": -1.0, "log_loss": float("inf")}
        has_val = X_val is not None and y_val is not None and len(X_val) > 0 and len(y_val) > 0
        if has_val:
            best_metrics = _val_metrics(y_val, model0.predict_proba(X_val), np.unique(y_labeled))
        loop_logs.append({"round": 0, "n_pseudo_new": 0, "val": best_metrics})

        X_ctx = X_labeled
        y_ctx = y_labeled.copy()
        remaining = np.ones(len(X_unlabeled), dtype=bool) if X_unlabeled is not None else np.zeros(0, dtype=bool)
        accepted_total = 0
        stopped = "max_loops"

        if X_unlabeled is None or len(X_unlabeled) == 0:
            self.model = model0
            self.training_meta = {
                "backbone": self.backbone,
                "method_fidelity": "novel_experimental",
                "reference_family": "hard-label risk-controlled TFM self-training (not soft LoopTabFM)",
                "protocol": "inductive",
                "loop_logs": loop_logs,
                "selected_round": 0,
                "stopped_reason": "no_unlabeled",
                "uses_unlabeled_data": False,
            }
            return self

        current = model0
        for r in range(1, max_loops + 1):
            if not remaining.any():
                stopped = "no_remaining_unlabeled"
                break
            pool_idx = np.where(remaining)[0]
            proba_pool = np.asarray(current.predict_proba(_slice_raw(X_unlabeled, pool_idx)))
            cal = _calibrate_teacher(current, X_ctx, y_ctx, X_val, y_val, self.random_state + r)
            proba_pool = apply_temperature(proba_pool, cal.temperature)
            sel = engine.select(
                proba_pool,
                n_labeled=len(y_labeled),
                budget=n_budget,
                labeled_classes=np.unique(y_labeled),
                n_already_accepted=accepted_total,
                round_idx=r,
            )
            if len(sel.indices) == 0:
                stopped = sel.stopping_reason or "zero_pseudo_labels"
                loop_logs.append({"round": r, "n_pseudo_new": 0, "val": None, "stopped": stopped})
                break
            global_idx = pool_idx[sel.indices]
            X_ctx = _concat_raw(X_ctx, _slice_raw(X_unlabeled, global_idx))
            y_ctx = np.concatenate([y_ctx, sel.pseudo_labels.astype(y_labeled.dtype)])
            remaining[global_idx] = False
            accepted_total += len(global_idx)
            nxt = _make_tfm(self.backbone, self.random_state + r, self.n_classes, **self.kwargs)
            nxt.fit(X_ctx, y_ctx)
            metrics = best_metrics
            if has_val:
                metrics = _val_metrics(y_val, nxt.predict_proba(X_val), np.unique(y_labeled))
                if _better_loop(metrics, best_metrics, eps_ba, eps_ll):
                    best_metrics = metrics
                    best_model = nxt
                    best_round = r
            else:
                best_model = nxt
                best_round = r
            loop_logs.append({
                "round": r,
                "n_pseudo_new": int(len(global_idx)),
                "val": metrics,
                "calibration_temperature": cal.temperature,
                "accepted_by_class": sel.accepted_by_class,
            })
            current = nxt

        self.model = best_model
        self.training_meta = {
            "backbone": self.backbone,
            "method_fidelity": "novel_experimental",
            "reference_family": "hard-label risk-controlled TFM self-training (not soft LoopTabFM)",
            "protocol": "inductive",
            "uses_unlabeled_data": True,
            "rounds_attempted": len(loop_logs) - 1,
            "selected_round": best_round,
            "loop_logs": loop_logs,
            "n_pseudo_added_total": accepted_total if best_round > 0 else 0,
            "stopped_reason": stopped,
            "fallback_reason": "loop0" if best_round == 0 else None,
            "min_improvement_ba": eps_ba,
            "max_logloss_degradation": eps_ll,
        }
        return self


class TFMCAST(_TFMContextSSL):
    def __init__(self, backbone: str, name: str, random_state: int = 0, n_classes: int = 2, **kwargs):
        super().__init__(random_state=random_state, n_classes=n_classes, **kwargs)
        self.backbone = backbone
        self.name = name

    def fit(self, X_labeled, y_labeled, X_unlabeled=None, X_val=None, y_val=None, n_labeled_budget=None, y_unlabeled_hidden=None):
        y_labeled = np.asarray(y_labeled)
        n_budget = int(n_labeled_budget or len(y_labeled))
        X_lab_proc = self.kwargs.get("X_labeled_processed")
        X_u_proc = self.kwargs.get("X_unlabeled_processed")
        if X_lab_proc is None:
            if isinstance(X_labeled, pd.DataFrame):
                X_lab_proc = X_labeled.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
            else:
                X_lab_proc = np.asarray(X_labeled, dtype=np.float32)
        if X_u_proc is None and X_unlabeled is not None:
            if isinstance(X_unlabeled, pd.DataFrame):
                X_u_proc = X_unlabeled.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
            else:
                X_u_proc = np.asarray(X_unlabeled, dtype=np.float32)

        tfm_kwargs = {k: v for k, v in self.kwargs.items() if not k.startswith("X_")}
        teacher = _make_tfm(self.backbone, self.random_state, self.n_classes, **tfm_kwargs)
        teacher.fit(X_labeled, y_labeled)
        meta = {
            "backbone": self.backbone,
            "method_fidelity": "novel_experimental",
            "reference_family": "CAST adaptation to TFM teachers",
            "protocol": "inductive",
            "uses_unlabeled_data": True,
        }
        if X_unlabeled is None or len(X_unlabeled) == 0:
            self.model = teacher
            meta["stopped_reason"] = "no_unlabeled"
            self.training_meta = meta
            return self

        cast = CASTModifier(selection_config=_default_selection_config(self.random_state))
        cast.fit_density(np.asarray(X_lab_proc, dtype=np.float32), y_labeled)
        cal = _calibrate_teacher(teacher, X_labeled, y_labeled, X_val, y_val, self.random_state)
        cast.temperature = cal.temperature
        cast.calibration_result = cal
        proba_u = np.asarray(teacher.predict_proba(X_unlabeled))
        sel = cast.select_pseudo_labels(
            proba_u,
            np.asarray(X_u_proc, dtype=np.float32),
            n_labeled=len(y_labeled),
            budget=n_budget,
            labeled_classes=np.unique(y_labeled),
        )
        meta.update({
            "calibration_temperature": cal.temperature,
            "calibration_mode": cal.calibration_mode,
            "n_pseudo_added_total": int(len(sel.indices)),
            "stopped_reason": sel.stopping_reason,
            "accepted_by_class": sel.accepted_by_class,
            "mean_selected_confidence": float(sel.confidences.mean()) if len(sel.indices) else np.nan,
            "mean_selected_density": float(sel.densities.mean()) if len(sel.indices) else np.nan,
            "density_adjustment": True,
        })
        if len(sel.indices) == 0:
            self.model = teacher
            meta["fallback_reason"] = "zero_pseudo_labels"
            self.training_meta = meta
            return self
        X_ctx = _concat_raw(X_labeled, _slice_raw(X_unlabeled, sel.indices))
        y_ctx = np.concatenate([y_labeled, sel.pseudo_labels.astype(y_labeled.dtype)])
        final = _make_tfm(self.backbone, self.random_state, self.n_classes, **tfm_kwargs)
        final.fit(X_ctx, y_ctx)
        self.model = final
        self.training_meta = meta
        return self

    def fit_from_context(self, ctx):
        views = ctx.views
        self.kwargs = {
            **self.kwargs,
            **(ctx.method_config or {}),
            "X_labeled_processed": views.X_labeled_processed,
            "X_unlabeled_processed": views.X_unlabeled_processed,
        }
        return self.fit(
            views.X_labeled_raw,
            views.y_labeled,
            X_unlabeled=views.X_unlabeled_raw,
            X_val=views.X_validation_raw if views.has_validation else None,
            y_val=views.y_validation if views.has_validation else None,
            n_labeled_budget=views.n_labeled,
        )


class TFMConsensusTabICL(_TFMContextSSL):
    name = "tfm_consensus_context_tabiclv2"
    backbone = "tabiclv2"

    def fit(self, X_labeled, y_labeled, X_unlabeled=None, X_val=None, y_val=None, n_labeled_budget=None, y_unlabeled_hidden=None):
        y_labeled = np.asarray(y_labeled)
        n_budget = int(n_labeled_budget or len(y_labeled))
        tabpfn = _make_tfm("tabpfn3", self.random_state, self.n_classes, **self.kwargs)
        tabicl = _make_tfm("tabiclv2", self.random_state, self.n_classes, **self.kwargs)
        tabpfn.fit(X_labeled, y_labeled)
        tabicl.fit(X_labeled, y_labeled)
        meta = {
            "backbone": "tabiclv2",
            "method_fidelity": "novel_experimental",
            "reference_family": "TabPFN-3 + TabICLv2 consensus context",
            "protocol": "inductive",
            "uses_unlabeled_data": True,
            "selected_round": 0,
        }
        if X_unlabeled is None or len(X_unlabeled) == 0:
            self.model = tabicl
            meta["stopped_reason"] = "no_unlabeled"
            self.training_meta = meta
            return self
        p1 = np.asarray(tabpfn.predict_proba(X_unlabeled))
        p2 = np.asarray(tabicl.predict_proba(X_unlabeled))
        cal1 = _calibrate_teacher(tabpfn, X_labeled, y_labeled, X_val, y_val, self.random_state)
        cal2 = _calibrate_teacher(tabicl, X_labeled, y_labeled, X_val, y_val, self.random_state + 1)
        p1 = apply_temperature(p1, cal1.temperature)
        p2 = apply_temperature(p2, cal2.temperature)
        cfg = _default_selection_config(
            self.random_state,
            agreement_required=True,
            confidence_threshold=0.85,
            agreement_min_confidence=0.80,
        )
        engine = PseudoLabelEngine(cfg)
        sel = engine.select(
            p2,
            teacher2_proba=p1,
            n_labeled=len(y_labeled),
            budget=n_budget,
            labeled_classes=np.unique(y_labeled),
        )
        meta.update({
            "n_pseudo_added_total": int(len(sel.indices)),
            "stopped_reason": sel.stopping_reason,
            "accepted_by_class": sel.accepted_by_class,
            "calibration_temperature_tabpfn": cal1.temperature,
            "calibration_temperature_tabicl": cal2.temperature,
        })
        if len(sel.indices) == 0:
            self.model = tabicl
            meta["fallback_reason"] = "loop0_no_consensus"
            self.training_meta = meta
            return self
        X_ctx = _concat_raw(X_labeled, _slice_raw(X_unlabeled, sel.indices))
        y_ctx = np.concatenate([y_labeled, sel.pseudo_labels.astype(y_labeled.dtype)])
        final = _make_tfm("tabiclv2", self.random_state, self.n_classes, **self.kwargs)
        final.fit(X_ctx, y_ctx)
        if X_val is not None and y_val is not None and len(X_val) > 0:
            m0 = _val_metrics(y_val, tabicl.predict_proba(X_val), np.unique(y_labeled))
            m1 = _val_metrics(y_val, final.predict_proba(X_val), np.unique(y_labeled))
            if m1["balanced_accuracy"] + 1e-6 < m0["balanced_accuracy"]:
                self.model = tabicl
                meta["fallback_reason"] = "consensus_hurt_validation"
                meta["selected_round"] = 0
            else:
                self.model = final
                meta["selected_round"] = 1
                meta["fallback_reason"] = None
        else:
            self.model = final
            meta["selected_round"] = 1
        self.training_meta = meta
        return self


def _fit_catboost_weighted(X, y, w, random_state, n_classes, cat_features=None):
    from src.models.supervised import _require_catboost
    CatBoostClassifier = _require_catboost()
    loss = "Logloss" if n_classes <= 2 else "MultiClass"
    model = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05, loss_function=loss,
        random_seed=random_state, verbose=False, allow_writing_files=False,
    )
    fit_kwargs = {}
    if cat_features is not None:
        fit_kwargs["cat_features"] = cat_features
    if w is not None:
        try:
            model.fit(X, y, sample_weight=w, **fit_kwargs)
            return model
        except TypeError:
            pass
    model.fit(X, y, **fit_kwargs)
    return model


def _cat_feature_indices(X: pd.DataFrame) -> list[int]:
    return [i for i, c in enumerate(X.columns) if not pd.api.types.is_numeric_dtype(X[c])]


class TFMTeacherCatBoost(_TFMContextSSL):
    def __init__(self, backbone: str, name: str, random_state: int = 0, n_classes: int = 2, **kwargs):
        super().__init__(random_state=random_state, n_classes=n_classes, **kwargs)
        self.backbone = backbone
        self.name = name

    def fit(self, X_labeled, y_labeled, X_unlabeled=None, X_val=None, y_val=None, n_labeled_budget=None, y_unlabeled_hidden=None):
        y_labeled = np.asarray(y_labeled)
        n_budget = int(n_labeled_budget or len(y_labeled))
        teacher = _make_tfm(self.backbone, self.random_state, self.n_classes, **self.kwargs)
        teacher.fit(X_labeled, y_labeled)
        cat_idx = _cat_feature_indices(X_labeled) if isinstance(X_labeled, pd.DataFrame) else None
        base = _fit_catboost_weighted(X_labeled, y_labeled, None, self.random_state, self.n_classes, cat_idx)
        meta = {
            "backbone": self.backbone, "student": "catboost",
            "method_fidelity": "novel_experimental",
            "reference_family": "TFM teacher → CatBoost student",
            "protocol": "inductive", "uses_unlabeled_data": True,
        }
        if X_unlabeled is None or len(X_unlabeled) == 0:
            self.model = base
            meta["stopped_reason"] = "no_unlabeled"
            self.training_meta = meta
            return self
        proba_u = np.asarray(teacher.predict_proba(X_unlabeled))
        cal = _calibrate_teacher(teacher, X_labeled, y_labeled, X_val, y_val, self.random_state)
        proba_u = apply_temperature(proba_u, cal.temperature)
        engine = PseudoLabelEngine(_default_selection_config(self.random_state))
        sel = engine.select(proba_u, n_labeled=len(y_labeled), budget=n_budget, labeled_classes=np.unique(y_labeled))
        meta.update({
            "n_pseudo_added_total": int(len(sel.indices)),
            "calibration_temperature": cal.temperature,
            "stopped_reason": sel.stopping_reason,
            "accepted_by_class": sel.accepted_by_class,
        })
        if len(sel.indices) == 0:
            self.model = base
            meta["fallback_reason"] = "zero_pseudo_labels"
            self.training_meta = meta
            return self
        X_aug = _concat_raw(X_labeled, _slice_raw(X_unlabeled, sel.indices))
        y_aug = np.concatenate([y_labeled, sel.pseudo_labels.astype(y_labeled.dtype)])
        w = np.concatenate([np.ones(len(y_labeled)), np.asarray(sel.weights, dtype=np.float64)])
        pl_mass = float(w[len(y_labeled):].sum())
        max_ratio = float(self.kwargs.get("max_pseudo_weight_ratio", 1.0))
        if pl_mass > max_ratio * len(y_labeled) and pl_mass > 0:
            w[len(y_labeled):] *= (max_ratio * len(y_labeled)) / pl_mass
        student = _fit_catboost_weighted(X_aug, y_aug, w, self.random_state, self.n_classes, cat_idx)
        if X_val is not None and y_val is not None and len(X_val) > 0:
            ba_base = balanced_accuracy_score(y_val, base.predict(X_val))
            ba_stu = balanced_accuracy_score(y_val, student.predict(X_val))
            meta["val_balanced_accuracy_base"] = float(ba_base)
            meta["val_balanced_accuracy_student"] = float(ba_stu)
            if ba_stu + 1e-6 < ba_base:
                self.model = base
                meta["fallback_reason"] = "student_hurt_validation"
            else:
                self.model = student
                meta["fallback_reason"] = None
        else:
            self.model = student
            meta["fallback_reason"] = None
        meta["pseudo_label_sample_weight_sum"] = float(w[len(y_labeled):].sum())
        self.training_meta = meta
        return self


class TFMConsensusCatBoost(TFMTeacherCatBoost):
    name = "tfm_consensus_catboost"
    backbone = "consensus"

    def fit(self, X_labeled, y_labeled, X_unlabeled=None, X_val=None, y_val=None, n_labeled_budget=None, y_unlabeled_hidden=None):
        y_labeled = np.asarray(y_labeled)
        n_budget = int(n_labeled_budget or len(y_labeled))
        t1 = _make_tfm("tabpfn3", self.random_state, self.n_classes, **self.kwargs)
        t2 = _make_tfm("tabiclv2", self.random_state, self.n_classes, **self.kwargs)
        t1.fit(X_labeled, y_labeled)
        t2.fit(X_labeled, y_labeled)
        cat_idx = _cat_feature_indices(X_labeled) if isinstance(X_labeled, pd.DataFrame) else None
        base = _fit_catboost_weighted(X_labeled, y_labeled, None, self.random_state, self.n_classes, cat_idx)
        meta = {
            "backbone": "tabpfn3+tabiclv2", "student": "catboost",
            "method_fidelity": "novel_experimental",
            "reference_family": "TFM consensus → CatBoost",
            "protocol": "inductive", "uses_unlabeled_data": True,
        }
        if X_unlabeled is None or len(X_unlabeled) == 0:
            self.model = base
            meta["stopped_reason"] = "no_unlabeled"
            self.training_meta = meta
            return self
        cal1 = _calibrate_teacher(t1, X_labeled, y_labeled, X_val, y_val, self.random_state)
        cal2 = _calibrate_teacher(t2, X_labeled, y_labeled, X_val, y_val, self.random_state + 1)
        p1 = apply_temperature(np.asarray(t1.predict_proba(X_unlabeled)), cal1.temperature)
        p2 = apply_temperature(np.asarray(t2.predict_proba(X_unlabeled)), cal2.temperature)
        engine = PseudoLabelEngine(_default_selection_config(self.random_state, agreement_required=True, confidence_threshold=0.85))
        sel = engine.select(p2, teacher2_proba=p1, n_labeled=len(y_labeled), budget=n_budget, labeled_classes=np.unique(y_labeled))
        meta["n_pseudo_added_total"] = int(len(sel.indices))
        meta["stopped_reason"] = sel.stopping_reason
        if len(sel.indices) == 0:
            self.model = base
            meta["fallback_reason"] = "zero_consensus"
            self.training_meta = meta
            return self
        X_aug = _concat_raw(X_labeled, _slice_raw(X_unlabeled, sel.indices))
        y_aug = np.concatenate([y_labeled, sel.pseudo_labels.astype(y_labeled.dtype)])
        conf = 0.5 * (p1[sel.indices].max(1) + p2[sel.indices].max(1))
        w = np.concatenate([np.ones(len(y_labeled)), conf])
        pl_mass = float(w[len(y_labeled):].sum())
        if pl_mass > len(y_labeled) and pl_mass > 0:
            w[len(y_labeled):] *= len(y_labeled) / pl_mass
        student = _fit_catboost_weighted(X_aug, y_aug, w, self.random_state, self.n_classes, cat_idx)
        if X_val is not None and y_val is not None and len(X_val) > 0:
            if balanced_accuracy_score(y_val, student.predict(X_val)) + 1e-6 < balanced_accuracy_score(y_val, base.predict(X_val)):
                self.model = base
                meta["fallback_reason"] = "student_hurt_validation"
            else:
                self.model = student
                meta["fallback_reason"] = None
        else:
            self.model = student
        self.training_meta = meta
        return self


class TabPFNUnlabeledPriorAdjustment(_TFMContextSSL):
    name = "tabpfn3_unlabeled_prior_adjustment"
    backbone = "tabpfn3"

    def fit(self, X_labeled, y_labeled, X_unlabeled=None, X_val=None, y_val=None, n_labeled_budget=None, y_unlabeled_hidden=None):
        y_labeled = np.asarray(y_labeled)
        model = _make_tfm("tabpfn3", self.random_state, self.n_classes, **self.kwargs)
        model.fit(X_labeled, y_labeled)
        self.model = model
        n_classes = int(max(int(y_labeled.max()) + 1, self.n_classes))
        labeled_prior = np.bincount(y_labeled, minlength=n_classes).astype(np.float64)
        labeled_prior = labeled_prior / max(labeled_prior.sum(), 1.0)
        if X_unlabeled is None or len(X_unlabeled) == 0:
            self._w = np.ones(n_classes)
            self.training_meta = {
                "protocol": "inductive", "method_fidelity": "novel_experimental",
                "reference_family": "unlabeled-pool prior adjustment (inductive)",
                "stopped_reason": "no_unlabeled", "uses_unlabeled_data": False, "backbone": "tabpfn3",
            }
            return self
        proba_u = np.asarray(model.predict_proba(X_unlabeled))
        u_prior = proba_u.mean(axis=0)
        u_prior = u_prior / max(u_prior.sum(), 1e-12)
        self._w = u_prior / np.clip(labeled_prior, 1e-6, None)
        self._w = self._w / self._w.mean()
        self.training_meta = {
            "protocol": "inductive", "method_fidelity": "novel_experimental",
            "reference_family": "unlabeled-pool prior adjustment (inductive)",
            "uses_unlabeled_data": True, "backbone": "tabpfn3",
            "labeled_prior": labeled_prior.tolist(),
            "unlabeled_prior_estimate": u_prior.tolist(),
            "prior_weights": self._w.tolist(),
            "adjustment_source": "unlabeled_train_pool_only",
        }
        return self

    def predict_proba(self, X):
        proba = np.asarray(self.model.predict_proba(X))
        adj = proba * self._w.reshape(1, -1)
        return adj / np.clip(adj.sum(axis=1, keepdims=True), 1e-12, None)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class TabPFNDistPFNTransductive(_TFMContextSSL):
    name = "tabpfn3_distpfn_transductive"
    backbone = "tabpfn3"

    def fit(self, *args, **kwargs):
        raise UnsupportedMethodError(
            self.name,
            "Faithful DistPFN reproduction is not available in this codebase yet; "
            "refusing to silently substitute a heuristic. Use tabpfn3_unlabeled_prior_adjustment "
            "for the inductive exploratory prior-adjustment baseline.",
            "unsupported_faithful_distpfn_unavailable",
        )


def build_tfm_ssl_model(method: str, random_state: int = 0, n_classes: int = 2, **kwargs):
    mapping = {
        "tabpfn3_self_training": lambda: TFMLoopRisk("tabpfn3", method, random_state, n_classes, **kwargs),
        "tabiclv2_self_training": lambda: TFMLoopRisk("tabiclv2", method, random_state, n_classes, **kwargs),
        "tabpfn3_pl_one_shot": lambda: TFMOneShotPL("tabpfn3", method, random_state, n_classes, **kwargs),
        "tabiclv2_pl_one_shot": lambda: TFMOneShotPL("tabiclv2", method, random_state, n_classes, **kwargs),
        "tabpfn3_loop_risk": lambda: TFMLoopRisk("tabpfn3", method, random_state, n_classes, **kwargs),
        "tabiclv2_loop_risk": lambda: TFMLoopRisk("tabiclv2", method, random_state, n_classes, **kwargs),
        "tabpfn3_cast": lambda: TFMCAST("tabpfn3", method, random_state, n_classes, **kwargs),
        "tabiclv2_cast": lambda: TFMCAST("tabiclv2", method, random_state, n_classes, **kwargs),
        "tfm_consensus_context_tabiclv2": lambda: TFMConsensusTabICL(random_state=random_state, n_classes=n_classes, **kwargs),
        "tabpfn3_teacher_catboost": lambda: TFMTeacherCatBoost("tabpfn3", method, random_state, n_classes, **kwargs),
        "tabiclv2_teacher_catboost": lambda: TFMTeacherCatBoost("tabiclv2", method, random_state, n_classes, **kwargs),
        "tfm_consensus_catboost": lambda: TFMConsensusCatBoost(random_state=random_state, n_classes=n_classes, **kwargs),
        "tabpfn3_unlabeled_prior_adjustment": lambda: TabPFNUnlabeledPriorAdjustment(random_state=random_state, n_classes=n_classes, **kwargs),
        "tabpfn3_distpfn_transductive": lambda: TabPFNDistPFNTransductive(random_state=random_state, n_classes=n_classes, **kwargs),
    }
    if method not in mapping:
        raise ValueError(method)
    return mapping[method]()
