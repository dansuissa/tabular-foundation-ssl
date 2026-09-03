"""Shared model protocols and adapters for array- and context-aware estimators.

Legacy estimators consume processed NumPy arrays.  Newer methods consume a
``FitContext`` so the dispatcher can provide native and processed views without
letting either interface bypass the benchmark split protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from src.views import FitContext, PredictionResult


@dataclass
class ModelResult:
    y_pred: np.ndarray
    y_proba: np.ndarray | None = None
    training_meta: dict[str, Any] = field(default_factory=dict)


class BaseModel(Protocol):
    name: str

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
    ) -> "BaseModel":
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        ...


class ContextAwareModel(Protocol):
    """Models that consume the dual-view ``FitContext`` directly."""

    name: str

    def fit_from_context(self, ctx: FitContext) -> Any:
        ...

    def predict_from_context(
        self,
        ctx: FitContext,
        *,
        eval_split: str = "test",
    ) -> PredictionResult:
        ...


def run_model(
    model: BaseModel,
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray | None,
    X_eval: np.ndarray,
) -> ModelResult:
    model.fit(X_labeled, y_labeled, X_unlabeled=X_unlabeled)
    y_pred = model.predict(X_eval)
    try:
        y_proba = model.predict_proba(X_eval)
    except (AttributeError, NotImplementedError):
        y_proba = None
    training_meta = getattr(model, "training_meta", {})
    return ModelResult(y_pred=y_pred, y_proba=y_proba, training_meta=training_meta)


def _is_context_aware(model: Any) -> bool:
    return callable(getattr(model, "fit_from_context", None)) and (
        callable(getattr(model, "predict_from_context", None))
        or callable(getattr(model, "fit_predict_from_context", None))
    )


def run_model_from_context(
    model: Any,
    ctx: FitContext,
    *,
    eval_split: str = "test",
) -> PredictionResult:
    """Dispatch FitContext-aware methods vs legacy ndarray models.

    Context-aware models expose ``fit_from_context`` plus either
    ``predict_from_context`` or ``fit_predict_from_context``.

    Legacy ndarray models are wrapped with :class:`LegacyAdapter` and run only
    on processed views (bit-compatible with :func:`run_model`).
    """
    if _is_context_aware(model):
        if callable(getattr(model, "fit_predict_from_context", None)):
            result = model.fit_predict_from_context(ctx, eval_split=eval_split)
        else:
            model.fit_from_context(ctx)
            result = model.predict_from_context(ctx, eval_split=eval_split)
        if isinstance(result, PredictionResult):
            return result
        # Tolerate ModelResult-shaped returns from early adapters.
        return PredictionResult(
            y_pred=result.y_pred,
            y_proba=getattr(result, "y_proba", None),
            training_meta=dict(getattr(result, "training_meta", {}) or {}),
        )

    from src.models.legacy_adapter import LegacyAdapter

    adapter = LegacyAdapter(model)
    return adapter.fit_predict_from_context(ctx, eval_split=eval_split)
