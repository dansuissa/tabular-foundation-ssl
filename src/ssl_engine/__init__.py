"""Shared SSL engine: calibration, CAST density, and pseudo-label selection.

Public exports used by TFM and classical method wrappers. Historical
``results/`` artifacts are never written by this package.
"""

from __future__ import annotations

from src.ssl_engine.calibration import CalibrationResult, apply_temperature, fit_temperature
from src.ssl_engine.density import AdjustedScores, CASTDensityAdjuster
from src.ssl_engine.diagnostics import post_hoc_pseudo_label_quality
from src.ssl_engine.pseudo_label_engine import (
    PseudoLabelEngine,
    SelectionConfig,
    SelectionResult,
    scores_from_proba,
)

__all__ = [
    "AdjustedScores",
    "CASTDensityAdjuster",
    "CalibrationResult",
    "PseudoLabelEngine",
    "SelectionConfig",
    "SelectionResult",
    "apply_temperature",
    "fit_temperature",
    "post_hoc_pseudo_label_quality",
    "scores_from_proba",
]
