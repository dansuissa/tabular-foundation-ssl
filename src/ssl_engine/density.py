"""CAST-style class-conditional density adjustment for pseudo-label confidence.

Scientific intent
-----------------
CAST (Cluster-Aware Self-Training; Kim et al.) regularizes classifier
confidence by class-conditional local density estimated from **labeled**
training features only. Pseudo-labels in low-density regions receive lower
effective confidence, aligning selection with the cluster assumption.

This module implements a leakage-safe, model-agnostic density factor
``γ ∈ (0, 1]`` and multiplicative adjusted confidence
``c_adj = c_raw * γ`` (equivalent to CAST with ``α = 1`` after min-max
scaling of γ; callers may blend via ``alpha``).

Leakage rules
-------------
* Fit density models **only** on labeled processed/raw training features
  and their labels.
* Never use unlabeled true labels, validation labels for density fit, or
  any test features/labels.
* No dense O(n²) pairwise matrices; Gaussian / KDE scoring is O(n · d²)
  or O(n · m · d) respectively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.neighbors import KernelDensity

_EPS = 1e-12
DensityBackend = Literal["gaussian", "kde"]


@dataclass
class AdjustedScores:
    """Per-sample CAST density-adjusted scores.

    All arrays are aligned to the rows of the input ``proba`` / ``X``.
    Raw confidence, density factor, and adjusted confidence are logged
    separately so ablations can isolate density effects.
    """

    confidence: np.ndarray
    density_factor: np.ndarray
    adjusted_confidence: np.ndarray
    pseudo_labels: np.ndarray
    margins: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ClassGaussian:
    mean: np.ndarray
    precision: np.ndarray  # inverse covariance (d, d)
    log_det: float
    n_samples: int
    used_pooled: bool


class CASTDensityAdjuster:
    """Class-conditional density estimator for CAST confidence modulation.

    Parameters
    ----------
    backend:
        ``\"gaussian\"`` (default) — class-conditional Gaussian with
        shrinkage / pooled covariance for tiny classes.
        ``\"kde\"`` — sklearn ``KernelDensity(kernel='gaussian')`` per class.
    min_samples_full_cov:
        Minimum labeled rows in a class to estimate a full class covariance.
        Below this, fall back to pooled covariance; if that is also
        impossible, use a uniform density factor of 1.0 for that class.
    alpha:
        CAST blend in ``c_adj = c * (alpha * γ + (1 - alpha))``.
        ``alpha=1`` is pure multiplicative density scaling.
    reg:
        Diagonal ridge added to covariances for numerical stability.
    kde_bandwidth:
        Bandwidth for the KDE backend.
    random_state:
        Reserved for deterministic tie-breaks / future stochastic estimators.
    """

    def __init__(
        self,
        *,
        backend: DensityBackend = "gaussian",
        min_samples_full_cov: int = 5,
        alpha: float = 1.0,
        reg: float = 1e-4,
        kde_bandwidth: float = 1.0,
        random_state: int = 0,
    ) -> None:
        if backend not in ("gaussian", "kde"):
            raise ValueError(f"Unknown backend={backend!r}")
        if not (0.0 <= float(alpha) <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
        self.backend: DensityBackend = backend
        self.min_samples_full_cov = int(min_samples_full_cov)
        self.alpha = float(alpha)
        self.reg = float(reg)
        self.kde_bandwidth = float(kde_bandwidth)
        self.random_state = int(random_state)

        self.classes_: np.ndarray | None = None
        self.n_features_: int | None = None
        self._gaussians: dict[Any, _ClassGaussian] = {}
        self._kdes: dict[Any, KernelDensity] = {}
        self._pooled: _ClassGaussian | None = None
        self._fitted = False
        self.fit_meta_: dict[str, Any] = {}

    def fit(self, X_labeled: np.ndarray, y_labeled: np.ndarray) -> "CASTDensityAdjuster":
        """Fit class-conditional densities on labeled features only.

        Leakage: ``X_labeled`` / ``y_labeled`` must be the labeled training
        split (or a labeled-only subset). Do not pass test or unlabeled labels.
        """
        X = np.asarray(X_labeled, dtype=np.float64)
        y = np.asarray(y_labeled)
        if X.ndim != 2:
            raise ValueError(f"X_labeled must be 2-D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X_labeled and y_labeled length mismatch")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit CASTDensityAdjuster on empty labeled set")

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        classes = np.unique(y)
        self.classes_ = classes
        self.n_features_ = int(X.shape[1])
        self._gaussians = {}
        self._kdes = {}
        self._pooled = self._fit_gaussian(X, used_pooled=False)

        class_meta: dict[str, Any] = {}
        for cls in classes.tolist():
            mask = y == cls
            n_c = int(mask.sum())
            X_c = X[mask]
            if self.backend == "kde":
                if n_c >= 1:
                    kde = KernelDensity(
                        kernel="gaussian",
                        bandwidth=self.kde_bandwidth,
                    )
                    kde.fit(X_c)
                    self._kdes[cls] = kde
                    class_meta[str(cls)] = {
                        "n_samples": n_c,
                        "backend": "kde",
                        "fallback": False,
                    }
                else:
                    class_meta[str(cls)] = {
                        "n_samples": 0,
                        "backend": "uniform",
                        "fallback": True,
                    }
            else:
                if n_c >= self.min_samples_full_cov and n_c >= 2:
                    g = self._fit_gaussian(X_c, used_pooled=False)
                    self._gaussians[cls] = g
                    class_meta[str(cls)] = {
                        "n_samples": n_c,
                        "backend": "gaussian",
                        "used_pooled": False,
                        "fallback": False,
                    }
                elif self._pooled is not None and n_c >= 1:
                    # Shrinkage: class mean + pooled covariance
                    mean = X_c.mean(axis=0)
                    g = _ClassGaussian(
                        mean=mean,
                        precision=self._pooled.precision.copy(),
                        log_det=self._pooled.log_det,
                        n_samples=n_c,
                        used_pooled=True,
                    )
                    self._gaussians[cls] = g
                    class_meta[str(cls)] = {
                        "n_samples": n_c,
                        "backend": "gaussian_pooled",
                        "used_pooled": True,
                        "fallback": False,
                    }
                else:
                    class_meta[str(cls)] = {
                        "n_samples": n_c,
                        "backend": "uniform",
                        "fallback": True,
                    }

        self._fitted = True
        self.fit_meta_ = {
            "backend": self.backend,
            "n_labeled": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(classes.size),
            "min_samples_full_cov": self.min_samples_full_cov,
            "alpha": self.alpha,
            "per_class": class_meta,
        }
        return self

    def adjust(self, proba: np.ndarray, X: np.ndarray) -> AdjustedScores:
        """Score unlabeled (or any) rows with density-adjusted confidence.

        Parameters
        ----------
        proba:
            Predictive probabilities ``(n, n_classes)``. Column order must
            match ``self.classes_`` when fitted (sorted unique labels).
        X:
            Feature matrix for the same rows, in the same feature space
            used at ``fit`` (typically processed labeled/unlabeled train).

        Returns
        -------
        AdjustedScores with separately logged raw confidence, density factor,
        and adjusted confidence.
        """
        if not self._fitted or self.classes_ is None or self.n_features_ is None:
            raise RuntimeError("CASTDensityAdjuster.fit() must be called before adjust()")

        proba = np.asarray(proba, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        if proba.ndim != 2:
            raise ValueError(f"proba must be 2-D, got shape {proba.shape}")
        if X.ndim != 2 or X.shape[0] != proba.shape[0]:
            raise ValueError("X and proba must share the same number of rows")
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features but adjuster was fit with {self.n_features_}"
            )
        if proba.shape[1] != self.classes_.size:
            raise ValueError(
                f"proba has {proba.shape[1]} classes but adjuster has {self.classes_.size}"
            )

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        n = proba.shape[0]
        if n == 0:
            empty = np.array([], dtype=np.float64)
            return AdjustedScores(
                confidence=empty,
                density_factor=empty,
                adjusted_confidence=empty,
                pseudo_labels=np.array([], dtype=self.classes_.dtype),
                margins=empty,
                meta={"n_samples": 0},
            )

        # Raw scores from probabilities
        order = np.argsort(proba, axis=1)
        top1 = order[:, -1]
        top2 = order[:, -2]
        rows = np.arange(n)
        confidence = proba[rows, top1]
        margins = confidence - proba[rows, top2]
        pseudo_labels = self.classes_[top1]

        # Class-conditional log-density for the predicted class
        log_dens = np.full(n, -np.inf, dtype=np.float64)
        for class_idx, cls in enumerate(self.classes_.tolist()):
            mask = top1 == class_idx
            if not np.any(mask):
                continue
            log_dens[mask] = self._log_density_for_class(cls, X[mask])

        # Convert to a density factor in (0, 1] via min-max over finite scores.
        density_factor = self._to_unit_interval(log_dens)
        # CAST blend: c_adj = c * (α γ + (1-α))
        scale = self.alpha * density_factor + (1.0 - self.alpha)
        adjusted = confidence * scale

        finite_frac = float(np.isfinite(log_dens).mean()) if n else 0.0
        return AdjustedScores(
            confidence=confidence.astype(np.float64),
            density_factor=density_factor.astype(np.float64),
            adjusted_confidence=adjusted.astype(np.float64),
            pseudo_labels=pseudo_labels,
            margins=margins.astype(np.float64),
            meta={
                "n_samples": int(n),
                "alpha": self.alpha,
                "finite_log_density_fraction": finite_frac,
                "density_factor_min": float(density_factor.min()) if n else float("nan"),
                "density_factor_max": float(density_factor.max()) if n else float("nan"),
                "raw_confidence_mean": float(confidence.mean()) if n else float("nan"),
                "adjusted_confidence_mean": float(adjusted.mean()) if n else float("nan"),
                "fit_meta": dict(self.fit_meta_),
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fit_gaussian(self, X: np.ndarray, *, used_pooled: bool) -> _ClassGaussian | None:
        if X.shape[0] == 0:
            return None
        d = X.shape[1]
        mean = X.mean(axis=0)
        if X.shape[0] == 1:
            cov = np.eye(d, dtype=np.float64) * self.reg
        else:
            # Biased covariance; avoid np.cov which can be awkward for d > n
            xc = X - mean
            cov = (xc.T @ xc) / max(X.shape[0], 1)
            cov = cov + np.eye(d, dtype=np.float64) * self.reg
        try:
            precision = np.linalg.inv(cov)
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0 or not np.isfinite(log_det):
                raise np.linalg.LinAlgError("non-SPD covariance")
        except np.linalg.LinAlgError:
            cov = np.eye(d, dtype=np.float64) * max(self.reg, 1e-3)
            precision = np.linalg.inv(cov)
            _, log_det = np.linalg.slogdet(cov)
        return _ClassGaussian(
            mean=mean.astype(np.float64),
            precision=precision.astype(np.float64),
            log_det=float(log_det),
            n_samples=int(X.shape[0]),
            used_pooled=used_pooled,
        )

    def _log_density_for_class(self, cls: Any, X: np.ndarray) -> np.ndarray:
        """Log-density under the class model; -inf triggers uniform factor."""
        n = X.shape[0]
        if self.backend == "kde":
            kde = self._kdes.get(cls)
            if kde is None:
                return np.full(n, -np.inf, dtype=np.float64)
            return np.asarray(kde.score_samples(X), dtype=np.float64)

        g = self._gaussians.get(cls)
        if g is None:
            return np.full(n, -np.inf, dtype=np.float64)
        return self._gaussian_logpdf(X, g)

    def _gaussian_logpdf(self, X: np.ndarray, g: _ClassGaussian) -> np.ndarray:
        d = X.shape[1]
        diff = X - g.mean
        # Mahalanobis via precision: (x-μ)^T Σ^{-1} (x-μ)
        # Avoid forming an n×n matrix: row-wise quadratic forms.
        mahal = np.einsum("ij,jk,ik->i", diff, g.precision, diff)
        return -0.5 * (d * np.log(2.0 * np.pi) + g.log_det + mahal)

    @staticmethod
    def _to_unit_interval(log_dens: np.ndarray) -> np.ndarray:
        """Map log-densities to density_factor ∈ (0, 1].

        Non-finite entries (tiny-class / missing model) get factor 1.0
        (neutral / uniform fallback — do not zero-out confidence solely
        because density estimation was impossible).
        """
        out = np.ones(log_dens.shape[0], dtype=np.float64)
        finite = np.isfinite(log_dens)
        if not np.any(finite):
            return out
        vals = log_dens[finite]
        vmin = float(vals.min())
        vmax = float(vals.max())
        if vmax > vmin:
            scaled = (vals - vmin) / (vmax - vmin)
        else:
            scaled = np.ones_like(vals)
        # Keep strictly in (0, 1]: floor away from 0 so multiplicative
        # adjustment never fully nullifies confidence via numerical underflow.
        scaled = np.clip(scaled, _EPS, 1.0)
        out[finite] = scaled
        return out
