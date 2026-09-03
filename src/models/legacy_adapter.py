"""Adapter wrapping legacy ndarray BaseModel APIs onto FitContext.

Processed views only — bit-compatible with the historical ``run_model`` path
when the underlying ``fit`` ignores validation. If ``fit`` accepts ``X_val`` /
``y_val``, external validation is forwarded for risk-aware methods without
changing classical callers.
"""
from __future__ import annotations

import inspect
from typing import Any, Protocol

import numpy as np

from src.views import FitContext, PredictionResult


class _LegacyModel(Protocol):
    name: str

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
    ) -> Any:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        ...


def _fit_with_optional_val(model: Any, views) -> None:
    kwargs: dict[str, Any] = {"X_unlabeled": views.X_unlabeled_processed}
    try:
        params = inspect.signature(model.fit).parameters
    except (TypeError, ValueError):
        params = {}
    if "X_val" in params and views.has_validation:
        kwargs["X_val"] = views.X_validation_processed
    if "y_val" in params and views.has_validation:
        kwargs["y_val"] = views.y_validation
    if "n_labeled_budget" in params:
        kwargs["n_labeled_budget"] = views.n_labeled
    model.fit(views.X_labeled_processed, views.y_labeled, **kwargs)


class LegacyAdapter:
    """Wrap an existing ndarray model so it consumes ``FitContext``."""

    def __init__(self, model: _LegacyModel) -> None:
        self.model = model
        self.name = getattr(model, "name", type(model).__name__)
        self.training_meta: dict[str, Any] = {}

    def fit_from_context(self, ctx: FitContext) -> "LegacyAdapter":
        _fit_with_optional_val(self.model, ctx.views)
        self.training_meta = dict(getattr(self.model, "training_meta", {}) or {})
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        try:
            return self.model.predict_proba(X)
        except (AttributeError, NotImplementedError):
            return None

    def fit_predict_from_context(
        self,
        ctx: FitContext,
        *,
        eval_split: str = "test",
    ) -> PredictionResult:
        views = ctx.views
        X_eval = _select_processed(views, eval_split)
        if callable(getattr(self.model, "fit_from_context", None)):
            self.model.fit_from_context(ctx)
            if callable(getattr(self.model, "predict_from_context", None)):
                return self.model.predict_from_context(ctx, eval_split=eval_split)
        else:
            _fit_with_optional_val(self.model, views)
        input_view = getattr(self.model, "input_view", None)
        if input_view in {"raw", "both"} and eval_split == "test":
            X_eval_raw = views.X_test_raw
            y_pred = self.model.predict(X_eval_raw)
            try:
                y_proba = self.model.predict_proba(X_eval_raw)
            except (AttributeError, NotImplementedError):
                y_proba = None
        else:
            y_pred = self.model.predict(X_eval)
            try:
                y_proba = self.model.predict_proba(X_eval)
            except (AttributeError, NotImplementedError):
                y_proba = None
        self.training_meta = dict(getattr(self.model, "training_meta", {}) or {})
        return PredictionResult(
            y_pred=y_pred,
            y_proba=y_proba,
            training_meta=self.training_meta,
        )


def _select_processed(views, eval_split: str) -> np.ndarray:
    key = eval_split.lower()
    if key == "test":
        return views.X_test_processed
    if key in {"val", "validation"}:
        return views.X_validation_processed
    if key == "labeled":
        return views.X_labeled_processed
    if key == "unlabeled":
        return views.X_unlabeled_processed
    raise ValueError(
        f"Unknown eval_split={eval_split!r}; "
        "expected one of: test, val/validation, labeled, unlabeled."
    )
