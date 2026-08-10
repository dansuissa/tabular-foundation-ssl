"""Model-agnostic CAST confidence modifier.

Scientific intent
-----------------
CAST (Cluster-Aware Self-Training) is a plug-in that regularizes predictive
confidence with class-conditional density estimated from labeled training
data. This module wraps ``CASTDensityAdjuster`` and the shared
``PseudoLabelEngine`` so CatBoost, LightGBM, TabPFN-3, and TabICLv2 share
one implementation rather than four divergent copies.

Leakage rules
-------------
* Density is fit on labeled training features (+ labels) only.
* Selection uses unlabeled features and model probabilities only — never
  unlabeled or test labels.
* Temperature calibration (optional) uses labeled validation or labeled
  OOF probabilities only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from src.ssl_engine.calibration import (
    CalibrationResult,
    apply_temperature,
    fit_temperature,
)
from src.ssl_engine.density import AdjustedScores, CASTDensityAdjuster
from src.ssl_engine.pseudo_label_engine import (
    PseudoLabelEngine,
    SelectionConfig,
    SelectionResult,
)


@dataclass
class CASTModifier:
    """Reusable CAST density + selection pipeline for any probabilistic model.

    Typical use
    -----------
    1. ``fit_density(X_labeled, y_labeled)``
    2. Optionally ``calibrate(...)`` on validation / OOF probabilities
    3. ``adjust_proba(proba, X)`` or ``select_pseudo_labels(proba, X, ...)``

    Parameters
    ----------
    density_adjuster:
        Pre-built adjuster, or constructed from ``density_kwargs``.
    selection_config:
        Base selection config; a copy with ``density_adjuster`` attached is
        used at select time.
    density_kwargs:
        Forwarded to ``CASTDensityAdjuster`` when ``density_adjuster`` is None.
    """

    density_adjuster: CASTDensityAdjuster | None = None
    selection_config: SelectionConfig | None = None
    density_kwargs: dict[str, Any] = field(default_factory=dict)
    temperature: float = 1.0
    calibration_result: CalibrationResult | None = None
    _engine: PseudoLabelEngine | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.density_adjuster is None:
            self.density_adjuster = CASTDensityAdjuster(**self.density_kwargs)
        if self.selection_config is None:
            self.selection_config = SelectionConfig(
                density_adjuster=self.density_adjuster,
                confidence_threshold=0.90,
                use_class_balanced=True,
                prevent_majority_domination=True,
                stricter_at_budget_50=True,
                multiplier_of_n_labeled=1.0,
            )
        else:
            # Ensure the shared adjuster is wired into selection.
            self.selection_config = replace(
                self.selection_config,
                density_adjuster=self.density_adjuster,
            )
        self._engine = PseudoLabelEngine(self.selection_config)

    # ------------------------------------------------------------------
    # Fit / calibrate
    # ------------------------------------------------------------------

    def fit_density(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
    ) -> "CASTModifier":
        """Fit class-conditional densities on labeled training data only."""
        assert self.density_adjuster is not None
        self.density_adjuster.fit(X_labeled, y_labeled)
        return self

    def calibrate(
        self,
        *,
        y_val: np.ndarray | None = None,
        proba_val: np.ndarray | None = None,
        y_labeled: np.ndarray | None = None,
        proba_oof: np.ndarray | None = None,
        random_state: int = 0,
    ) -> CalibrationResult:
        """Fit scalar temperature; never uses unlabeled/test labels."""
        result = fit_temperature(
            y_val=y_val,
            proba_val=proba_val,
            y_labeled=y_labeled,
            proba_oof=proba_oof,
            random_state=random_state,
        )
        self.calibration_result = result
        self.temperature = float(result.temperature)
        return result

    # ------------------------------------------------------------------
    # Transform / select
    # ------------------------------------------------------------------

    def apply_calibration(self, proba: np.ndarray) -> np.ndarray:
        """Apply the stored temperature to ``proba``."""
        return apply_temperature(proba, self.temperature)

    def adjust_proba(
        self,
        proba: np.ndarray,
        X: np.ndarray,
        *,
        apply_temp: bool = True,
    ) -> AdjustedScores:
        """Return CAST-adjusted scores for ``(proba, X)``.

        Optionally applies temperature scaling first. Density fit must
        already have been called.
        """
        assert self.density_adjuster is not None
        p = self.apply_calibration(proba) if apply_temp else np.asarray(proba, dtype=np.float64)
        return self.density_adjuster.adjust(p, X)

    def select_pseudo_labels(
        self,
        proba: np.ndarray,
        X: np.ndarray,
        *,
        n_labeled: int | None = None,
        budget: int | None = None,
        round_idx: int = 0,
        existing_mask: np.ndarray | None = None,
        teacher2_proba: np.ndarray | None = None,
        apply_temp: bool = True,
        classes: np.ndarray | None = None,
        labeled_classes: np.ndarray | None = None,
        n_already_accepted: int | None = None,
        sample_weights_fn: Any = None,
        config: SelectionConfig | None = None,
    ) -> SelectionResult:
        """Calibrate (optional) → CAST density adjust → shared PL selection."""
        assert self._engine is not None
        assert self.density_adjuster is not None
        p = self.apply_calibration(proba) if apply_temp else np.asarray(proba, dtype=np.float64)
        t2 = None
        if teacher2_proba is not None:
            t2 = (
                self.apply_calibration(teacher2_proba)
                if apply_temp
                else np.asarray(teacher2_proba, dtype=np.float64)
            )

        sel_cfg = config
        if sel_cfg is None:
            sel_cfg = replace(self.selection_config, density_adjuster=self.density_adjuster)  # type: ignore[arg-type]
        else:
            sel_cfg = replace(sel_cfg, density_adjuster=self.density_adjuster)

        result = self._engine.select(
            p,
            X,
            n_labeled=n_labeled,
            budget=budget,
            round_idx=round_idx,
            existing_mask=existing_mask,
            teacher2_proba=t2,
            sample_weights_fn=sample_weights_fn,
            classes=classes,
            labeled_classes=labeled_classes,
            n_already_accepted=n_already_accepted,
            config=sel_cfg,
        )
        result.meta = {
            **result.meta,
            "cast_temperature": float(self.temperature),
            "cast_calibration_mode": (
                self.calibration_result.calibration_mode
                if self.calibration_result is not None
                else "uncalibrated"
            ),
            "cast_density_fit_meta": dict(self.density_adjuster.fit_meta_),
        }
        return result

    def modify_confidence(
        self,
        proba: np.ndarray,
        X: np.ndarray,
        *,
        apply_temp: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convenience: return ``(raw_confidence, density_factor, adjusted)``."""
        scores = self.adjust_proba(proba, X, apply_temp=apply_temp)
        return scores.confidence, scores.density_factor, scores.adjusted_confidence
