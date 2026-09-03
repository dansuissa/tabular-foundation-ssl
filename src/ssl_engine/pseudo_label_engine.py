"""Shared pseudo-label selection engine for TFM and classical SSL methods.

Scientific intent
-----------------
Centralize conservative, leakage-safe pseudo-label selection so TabPFN,
TabICL, CAST, LoopTabFM-style, and GBDT self-training share identical
gates (confidence, margin, entropy, density, agreement, caps) and
deterministic tie handling. Zero accepted labels is a valid outcome;
selection never fabricates examples to satisfy caps.

Leakage rules
-------------
* Selection uses predictive probabilities and (optionally) unlabeled
  **features** for density. It must never observe unlabeled true labels
  or any test labels.
* Validation labels may be used upstream for temperature fitting only;
  they must not enter ``select``.
* ``post_hoc_pseudo_label_quality`` (diagnostics) may use hidden OpenML
  labels **after** training/prediction only — never call it here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal

import numpy as np

from src.ssl_engine.density import CASTDensityAdjuster

SelectionMode = Literal["accumulate", "refresh"]
_EPS = 1e-12


@dataclass
class SelectionConfig:
    """Configurable gates and caps for pseudo-label selection.

    Default safety rules are conservative for low-label tabular SSL:
    class-balanced selection, majority-domination prevention, optional
    stricter thresholds at budget 50, and hard caps relative to
    ``n_labeled``.
    """

    confidence_threshold: float = 0.90
    margin_threshold: float | None = 0.10
    entropy_threshold: float | None = None
    use_class_balanced: bool = True
    per_class_cap: int | None = 500
    per_round_cap: int | None = 2000
    total_cap: int | None = None
    multiplier_of_n_labeled: float | None = 1.0
    stricter_at_budget_50: bool = True
    require_all_classes: bool = False
    prevent_majority_domination: bool = True
    majority_domination_ratio: float = 2.0
    density_adjuster: CASTDensityAdjuster | None = None
    agreement_required: bool = False
    agreement_min_confidence: float | None = None
    mode: SelectionMode = "accumulate"
    random_state: int = 0
    # When stricter_at_budget_50 and budget == 50, multiply thresholds:
    budget_50_confidence: float = 0.95
    budget_50_margin: float | None = 0.15
    class_thresholds: dict[Any, float] | None = None


@dataclass
class SelectionResult:
    """Accepted pseudo-labels for one selection round."""

    indices: np.ndarray
    pseudo_labels: np.ndarray
    confidences: np.ndarray
    margins: np.ndarray
    densities: np.ndarray
    weights: np.ndarray
    accepted_by_class: dict[Any, int]
    stopping_reason: str
    thresholds_used: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreBundle:
    confidence: np.ndarray
    margin: np.ndarray
    entropy: np.ndarray
    pseudo_labels: np.ndarray
    top1_idx: np.ndarray


def scores_from_proba(
    proba: np.ndarray,
    *,
    classes: np.ndarray | None = None,
) -> ScoreBundle:
    """Derive confidence, top-2 margin, and predictive entropy from ``proba``.

    Parameters
    ----------
    proba:
        Array ``(n, C)`` of class probabilities.
    classes:
        Optional label values for columns. Defaults to ``np.arange(C)``.
    """
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2:
        raise ValueError(f"proba must be 2-D, got shape {proba.shape}")
    n, c = proba.shape
    if c < 2:
        raise ValueError("proba must have at least 2 classes")
    if classes is None:
        classes = np.arange(c)
    else:
        classes = np.asarray(classes)
        if classes.shape[0] != c:
            raise ValueError("classes length must match proba.shape[1]")

    if n == 0:
        empty_f = np.array([], dtype=np.float64)
        empty_i = np.array([], dtype=np.int64)
        return ScoreBundle(
            confidence=empty_f,
            margin=empty_f,
            entropy=empty_f,
            pseudo_labels=np.array([], dtype=classes.dtype),
            top1_idx=empty_i,
        )

    order = np.argsort(proba, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    rows = np.arange(n)
    confidence = proba[rows, top1]
    margin = confidence - proba[rows, top2]
    p = np.clip(proba, _EPS, 1.0)
    entropy = -np.sum(p * np.log(p), axis=1)
    return ScoreBundle(
        confidence=confidence,
        margin=margin,
        entropy=entropy,
        pseudo_labels=classes[top1],
        top1_idx=top1,
    )


def _stable_argsort_desc(values: np.ndarray, random_state: int) -> np.ndarray:
    """Deterministic descending argsort with secondary index key.

    Ties are broken by a seeded permutation of indices so that equal
    scores do not depend on memory order, then by original index for
    full stability.
    """
    n = values.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    rng = np.random.RandomState(int(random_state))
    # Primary: -value; secondary: random rank; tertiary: index
    tie_key = rng.permutation(n)
    # lexsort uses last key as primary
    order = np.lexsort((np.arange(n), tie_key, -values))
    return order.astype(np.int64)


def _effective_thresholds(
    config: SelectionConfig,
    budget: int | None,
) -> dict[str, Any]:
    conf_t = float(config.confidence_threshold)
    margin_t = config.margin_threshold
    entropy_t = config.entropy_threshold
    if config.stricter_at_budget_50 and budget is not None and int(budget) == 50:
        conf_t = float(config.budget_50_confidence)
        if config.budget_50_margin is not None:
            margin_t = float(config.budget_50_margin)
    return {
        "confidence_threshold": conf_t,
        "margin_threshold": margin_t,
        "entropy_threshold": entropy_t,
        "class_thresholds": dict(config.class_thresholds)
        if config.class_thresholds
        else None,
        "stricter_budget_50_applied": bool(
            config.stricter_at_budget_50 and budget is not None and int(budget) == 50
        ),
    }


def _empty_result(
    *,
    reason: str,
    thresholds: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> SelectionResult:
    return SelectionResult(
        indices=np.array([], dtype=np.int64),
        pseudo_labels=np.array([], dtype=np.int64),
        confidences=np.array([], dtype=np.float64),
        margins=np.array([], dtype=np.float64),
        densities=np.array([], dtype=np.float64),
        weights=np.array([], dtype=np.float64),
        accepted_by_class={},
        stopping_reason=reason,
        thresholds_used=thresholds,
        meta=meta or {},
    )


class PseudoLabelEngine:
    """Reusable pseudo-label selector with conservative defaults."""

    def __init__(self, config: SelectionConfig | None = None) -> None:
        self.config = config if config is not None else SelectionConfig()

    def scores_from_proba(
        self,
        proba: np.ndarray,
        *,
        classes: np.ndarray | None = None,
    ) -> ScoreBundle:
        """Instance wrapper around module-level ``scores_from_proba``."""
        return scores_from_proba(proba, classes=classes)

    def select(
        self,
        proba: np.ndarray,
        X: np.ndarray | None = None,
        *,
        n_labeled: int | None = None,
        budget: int | None = None,
        round_idx: int = 0,
        existing_mask: np.ndarray | None = None,
        teacher2_proba: np.ndarray | None = None,
        sample_weights_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        | None = None,
        classes: np.ndarray | None = None,
        labeled_classes: np.ndarray | None = None,
        n_already_accepted: int | None = None,
        config: SelectionConfig | None = None,
    ) -> SelectionResult:
        """Select pseudo-labels from predictive probabilities.

        Parameters
        ----------
        proba:
            Probabilities for the candidate pool (usually unlabeled train).
        X:
            Optional features aligned with ``proba`` (required if a density
            adjuster is configured).
        n_labeled:
            Labeled training size; used with ``multiplier_of_n_labeled``.
        budget:
            Label budget (e.g. 50/100/250/500); triggers stricter thresholds
            when ``stricter_at_budget_50`` is set.
        round_idx:
            Self-training / loop round index (logged only).
        existing_mask:
            Boolean mask over the pool. In ``accumulate`` mode, ``True``
            means already accepted (excluded). In ``refresh`` mode the
            mask is ignored for exclusion (full reselection).
        teacher2_proba:
            Optional second teacher probabilities for agreement gating.
        sample_weights_fn:
            Optional ``(indices, confidences, densities) -> weights``.
        classes:
            Column labels for ``proba``. Defaults to ``0..C-1``.
        labeled_classes:
            Classes present in the labeled set (for ``require_all_classes``
            / majority checks). Defaults to unique of predicted classes
            among candidates that pass gates, or ``classes``.
        n_already_accepted:
            Count already accepted across rounds (for ``total_cap``).
        config:
            Optional per-call override of ``self.config``.

        Returns
        -------
        SelectionResult. Zero indices is a valid outcome; this method
        never forces adding examples.
        """
        cfg = config if config is not None else self.config
        thresholds = _effective_thresholds(cfg, budget)
        proba = np.asarray(proba, dtype=np.float64)
        if proba.ndim != 2:
            raise ValueError(f"proba must be 2-D, got shape {proba.shape}")
        n_pool, n_classes = proba.shape
        if classes is None:
            classes = np.arange(n_classes)
        else:
            classes = np.asarray(classes)
            if classes.shape[0] != n_classes:
                raise ValueError("classes length must match proba.shape[1]")

        if n_pool == 0:
            return _empty_result(
                reason="empty_pool",
                thresholds=thresholds,
                meta={"round_idx": int(round_idx)},
            )

        if not np.all(np.isfinite(proba)):
            return _empty_result(
                reason="non_finite_proba",
                thresholds=thresholds,
                meta={"round_idx": int(round_idx)},
            )

        scores = scores_from_proba(proba, classes=classes)
        confidence = scores.confidence.copy()
        margin = scores.margin.copy()
        entropy = scores.entropy.copy()
        pseudo_labels = scores.pseudo_labels.copy()
        density = np.ones(n_pool, dtype=np.float64)

        # Density-adjusted confidence (CAST); log raw vs adjusted separately.
        raw_confidence = confidence.copy()
        if cfg.density_adjuster is not None:
            if X is None:
                raise ValueError("X is required when density_adjuster is set")
            X_arr = np.asarray(X)
            if X_arr.shape[0] != n_pool:
                raise ValueError("X rows must match proba rows")
            adjusted = cfg.density_adjuster.adjust(proba, X_arr)
            density = adjusted.density_factor
            confidence = adjusted.adjusted_confidence
            # Keep pseudo_labels / margins from probability scores for
            # consistency with teacher argmax; density only reweights confidence.

        # Eligible mask
        eligible = np.ones(n_pool, dtype=bool)

        if cfg.mode == "accumulate" and existing_mask is not None:
            existing = np.asarray(existing_mask, dtype=bool)
            if existing.shape[0] != n_pool:
                raise ValueError("existing_mask length must match proba rows")
            eligible &= ~existing

        conf_t = float(thresholds["confidence_threshold"])
        class_thresholds = thresholds.get("class_thresholds")
        if class_thresholds:
            # Per-class confidence gate; default to global threshold.
            for i in range(n_pool):
                lab = pseudo_labels[i]
                t_i = float(class_thresholds.get(lab, conf_t))
                if confidence[i] < t_i:
                    eligible[i] = False
        else:
            eligible &= confidence >= conf_t

        margin_t = thresholds["margin_threshold"]
        if margin_t is not None:
            eligible &= margin >= float(margin_t)

        entropy_t = thresholds["entropy_threshold"]
        if entropy_t is not None:
            eligible &= entropy <= float(entropy_t)

        # Teacher agreement
        agreement_rate = float("nan")
        if cfg.agreement_required:
            if teacher2_proba is None:
                return _empty_result(
                    reason="agreement_required_but_teacher2_missing",
                    thresholds=thresholds,
                    meta={"round_idx": int(round_idx)},
                )
            t2 = np.asarray(teacher2_proba, dtype=np.float64)
            if t2.shape != proba.shape:
                raise ValueError("teacher2_proba shape must match proba")
            pred1 = scores.top1_idx
            pred2 = np.argmax(t2, axis=1)
            agree = pred1 == pred2
            agreement_rate = float(agree.mean()) if n_pool else float("nan")
            eligible &= agree
            if cfg.agreement_min_confidence is not None:
                t2_conf = t2[np.arange(n_pool), pred2]
                eligible &= t2_conf >= float(cfg.agreement_min_confidence)

        if not np.any(eligible):
            return _empty_result(
                reason="no_candidates",
                thresholds=thresholds,
                meta={
                    "round_idx": int(round_idx),
                    "agreement_rate": agreement_rate,
                    "n_eligible_before_caps": 0,
                    "raw_confidence_mean": float(raw_confidence.mean()),
                    "density_mean": float(density.mean()),
                },
            )

        candidate_idx = np.where(eligible)[0]
        # Rank by adjusted confidence (desc), deterministic ties.
        order = _stable_argsort_desc(confidence[candidate_idx], cfg.random_state)
        candidate_idx = candidate_idx[order]

        # Caps
        already = int(n_already_accepted or 0)
        if cfg.mode == "accumulate" and existing_mask is not None:
            already = max(already, int(np.asarray(existing_mask).sum()))

        round_cap = cfg.per_round_cap if cfg.per_round_cap is not None else candidate_idx.size
        total_remaining = None
        if cfg.total_cap is not None:
            total_remaining = max(int(cfg.total_cap) - already, 0)
        if cfg.multiplier_of_n_labeled is not None and n_labeled is not None:
            mult_cap = int(np.floor(float(cfg.multiplier_of_n_labeled) * int(n_labeled)))
            mult_remaining = max(mult_cap - already, 0)
            total_remaining = (
                mult_remaining
                if total_remaining is None
                else min(total_remaining, mult_remaining)
            )

        if total_remaining is not None and total_remaining <= 0:
            return _empty_result(
                reason="total_cap_reached",
                thresholds=thresholds,
                meta={
                    "round_idx": int(round_idx),
                    "n_already_accepted": already,
                    "agreement_rate": agreement_rate,
                },
            )

        max_take = int(round_cap)
        if total_remaining is not None:
            max_take = min(max_take, int(total_remaining))
        if max_take <= 0:
            return _empty_result(
                reason="round_cap_zero",
                thresholds=thresholds,
                meta={"round_idx": int(round_idx)},
            )

        if labeled_classes is None:
            labeled_classes = classes
        labeled_classes = np.asarray(labeled_classes)
        labeled_set = list(dict.fromkeys(labeled_classes.tolist()))  # stable unique

        selected = self._cap_selection(
            candidate_idx=candidate_idx,
            pseudo_labels=pseudo_labels,
            confidence=confidence,
            max_take=max_take,
            per_class_cap=cfg.per_class_cap,
            use_class_balanced=cfg.use_class_balanced,
            prevent_majority_domination=cfg.prevent_majority_domination,
            majority_domination_ratio=cfg.majority_domination_ratio,
            labeled_classes=labeled_set,
            require_all_classes=cfg.require_all_classes,
        )

        if selected.size == 0:
            reason = (
                "require_all_classes_unsatisfied"
                if cfg.require_all_classes
                else "no_candidates_after_caps"
            )
            return _empty_result(
                reason=reason,
                thresholds=thresholds,
                meta={
                    "round_idx": int(round_idx),
                    "n_eligible_before_caps": int(candidate_idx.size),
                    "agreement_rate": agreement_rate,
                },
            )

        # Preserve ranking order among selected
        rank = {int(i): r for r, i in enumerate(candidate_idx.tolist())}
        selected = np.asarray(
            sorted(selected.tolist(), key=lambda i: rank[int(i)]),
            dtype=np.int64,
        )

        sel_labels = pseudo_labels[selected]
        sel_conf = confidence[selected]
        sel_margin = margin[selected]
        sel_dens = density[selected]
        sel_raw = raw_confidence[selected]

        if sample_weights_fn is not None:
            weights = np.asarray(
                sample_weights_fn(selected, sel_conf, sel_dens),
                dtype=np.float64,
            )
            if weights.shape[0] != selected.shape[0]:
                raise ValueError("sample_weights_fn returned wrong length")
        else:
            # Default: weight by adjusted confidence (clipped).
            weights = np.clip(sel_conf, 0.0, 1.0)

        accepted_by_class = {
            k: int(v) for k, v in Counter(sel_labels.tolist()).items()
        }

        stopping_reason = "selected"
        if total_remaining is not None and selected.size >= total_remaining:
            stopping_reason = "total_cap"
        elif cfg.per_round_cap is not None and selected.size >= int(cfg.per_round_cap):
            stopping_reason = "per_round_cap"

        return SelectionResult(
            indices=selected,
            pseudo_labels=sel_labels,
            confidences=sel_conf,
            margins=sel_margin,
            densities=sel_dens,
            weights=weights,
            accepted_by_class=accepted_by_class,
            stopping_reason=stopping_reason,
            thresholds_used=thresholds,
            meta={
                "round_idx": int(round_idx),
                "mode": cfg.mode,
                "n_pool": int(n_pool),
                "n_eligible_before_caps": int(candidate_idx.size),
                "n_selected": int(selected.size),
                "n_already_accepted": already,
                "agreement_rate": agreement_rate,
                "raw_confidence_mean": float(sel_raw.mean()) if selected.size else float("nan"),
                "raw_confidence_quantiles": _quantiles(sel_raw),
                "adjusted_confidence_mean": float(sel_conf.mean()) if selected.size else float("nan"),
                "density_mean": float(sel_dens.mean()) if selected.size else float("nan"),
                "density_quantiles": _quantiles(sel_dens),
                "budget": budget,
                "n_labeled": n_labeled,
                "density_adjusted": cfg.density_adjuster is not None,
            },
        )

    def with_config(self, **kwargs: Any) -> "PseudoLabelEngine":
        """Return a new engine with updated SelectionConfig fields."""
        return PseudoLabelEngine(replace(self.config, **kwargs))

    # ------------------------------------------------------------------
    # Caps / balancing
    # ------------------------------------------------------------------

    @staticmethod
    def _cap_selection(
        *,
        candidate_idx: np.ndarray,
        pseudo_labels: np.ndarray,
        confidence: np.ndarray,
        max_take: int,
        per_class_cap: int | None,
        use_class_balanced: bool,
        prevent_majority_domination: bool,
        majority_domination_ratio: float,
        labeled_classes: list[Any],
        require_all_classes: bool,
    ) -> np.ndarray:
        if max_take <= 0 or candidate_idx.size == 0:
            return np.array([], dtype=np.int64)

        labels_c = pseudo_labels[candidate_idx]
        classes_present = list(dict.fromkeys(labels_c.tolist()))
        n_cls = max(len(labeled_classes), 1)

        if require_all_classes:
            missing = [c for c in labeled_classes if c not in set(classes_present)]
            if missing:
                # Do not force: return empty rather than a partial majority set.
                return np.array([], dtype=np.int64)

        per_class_limit = per_class_cap if per_class_cap is not None else max_take
        if use_class_balanced:
            # Soft balance: round-robin over classes in labeled order, then
            # by remaining candidates sorted by confidence (already ranked).
            buckets: dict[Any, list[int]] = {c: [] for c in labeled_classes}
            for idx in candidate_idx.tolist():
                lab = pseudo_labels[idx]
                if lab in buckets:
                    buckets[lab].append(int(idx))
                else:
                    buckets.setdefault(lab, []).append(int(idx))

            selected: list[int] = []
            counts: dict[Any, int] = {c: 0 for c in buckets}
            # Fair round-robin
            progressed = True
            while progressed and len(selected) < max_take:
                progressed = False
                for c in list(buckets.keys()):
                    if len(selected) >= max_take:
                        break
                    if counts[c] >= per_class_limit:
                        continue
                    if not buckets[c]:
                        continue
                    if prevent_majority_domination and selected:
                        # Cap any class relative to the current min non-zero
                        nonzero = [counts[k] for k in counts if counts[k] > 0]
                        if nonzero:
                            min_c = min(nonzero)
                            if counts[c] >= max(
                                1, int(np.ceil(majority_domination_ratio * min_c))
                            ) and counts[c] > min_c:
                                # Allow catch-up classes; skip further majority growth
                                # only when this class already leads beyond ratio.
                                leaders = [
                                    k
                                    for k in counts
                                    if counts[k]
                                    >= max(1, int(np.ceil(majority_domination_ratio * min_c)))
                                ]
                                if c in leaders and counts[c] > min_c:
                                    continue
                    selected.append(buckets[c].pop(0))
                    counts[c] += 1
                    progressed = True
            return np.asarray(selected, dtype=np.int64)

        # Non-balanced: greedily take top confidence with per-class caps
        selected = []
        counts = {}
        for idx in candidate_idx.tolist():
            if len(selected) >= max_take:
                break
            lab = pseudo_labels[idx]
            cnt = counts.get(lab, 0)
            if cnt >= per_class_limit:
                continue
            if prevent_majority_domination and selected:
                nonzero = [v for v in counts.values() if v > 0]
                if nonzero:
                    min_c = min(nonzero)
                    if cnt >= max(1, int(np.ceil(majority_domination_ratio * min_c))) and cnt > min_c:
                        continue
            selected.append(int(idx))
            counts[lab] = cnt + 1
        return np.asarray(selected, dtype=np.int64)


def _quantiles(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {"q25": float("nan"), "q50": float("nan"), "q75": float("nan")}
    q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75])
    return {"q25": float(q25), "q50": float(q50), "q75": float(q75)}
