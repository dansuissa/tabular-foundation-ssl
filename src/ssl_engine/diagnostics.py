"""Post-hoc pseudo-label quality diagnostics (evaluation-time only).

Scientific intent
-----------------
After a model has finished training and producing test predictions, we may
compare accepted pseudo-labels against the **hidden** OpenML training labels
for the unlabeled pool. This measures confirmation bias / PL noise for
analysis and never feeds back into selection or fitting.

Leakage rules
-------------
* Call **only after** training and final prediction are complete.
* Never invoke from ``PseudoLabelEngine.select`` or any fit path.
* Results are for logging / reports only.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
)


def post_hoc_pseudo_label_quality(
    y_true_hidden: np.ndarray,
    selected_indices: np.ndarray,
    pseudo_labels: np.ndarray,
    confidences: np.ndarray | None = None,
    densities: np.ndarray | None = None,
    *,
    margins: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    accepted_by_class: dict[Any, int] | None = None,
    round_idx: int | None = None,
    labeled_class_counts: dict[Any, int] | None = None,
    agreement_rate: float | None = None,
    n_labeled: int | None = None,
    classes: np.ndarray | list[Any] | None = None,
) -> dict[str, Any]:
    """Compute post-hoc PL quality against hidden unlabeled true labels.

    Parameters
    ----------
    y_true_hidden:
        True labels for the **full unlabeled pool** (or an array indexable
        by ``selected_indices``). Used only for diagnostics.
    selected_indices:
        Indices into ``y_true_hidden`` that were accepted as pseudo-labels.
    pseudo_labels:
        Predicted hard labels for those indices (same length as selected).
    confidences, densities, margins, weights:
        Optional per-selected arrays for quantile / summary logging.
    accepted_by_class:
        Optional precomputed acceptance counts.
    round_idx:
        Optional round identifier.
    labeled_class_counts:
        Optional labeled prior for class-prior drift.
    agreement_rate:
        Optional teacher agreement rate from selection meta.
    n_labeled:
        Optional labeled size for final PL multiplier.
    classes:
        Optional explicit class list for precision_by_class keys.

    Returns
    -------
    Dictionary of diagnostic metrics suitable for result-row / JSON logging.
    """
    y_true_hidden = np.asarray(y_true_hidden)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    pseudo_labels = np.asarray(pseudo_labels)

    out: dict[str, Any] = {
        "diagnostic_only": True,
        "n_selected": int(selected_indices.size),
        "round_idx": round_idx,
    }

    if selected_indices.size == 0:
        out.update(
            {
                "accuracy": float("nan"),
                "balanced_accuracy": float("nan"),
                "precision_by_class": {},
                "accepted_by_class": dict(accepted_by_class or {}),
                "confidence_quantiles": _empty_quantiles(),
                "density_quantiles": _empty_quantiles(),
                "margin_quantiles": _empty_quantiles(),
                "class_prior_drift": {},
                "agreement_rate": agreement_rate if agreement_rate is not None else float("nan"),
                "final_pseudo_label_multiplier": 0.0
                if n_labeled is not None
                else float("nan"),
            }
        )
        return out

    if pseudo_labels.shape[0] != selected_indices.shape[0]:
        raise ValueError("pseudo_labels and selected_indices length mismatch")
    if np.any(selected_indices < 0) or np.any(selected_indices >= y_true_hidden.shape[0]):
        raise ValueError("selected_indices out of bounds for y_true_hidden")

    y_true_sel = y_true_hidden[selected_indices]
    y_pred_sel = pseudo_labels

    if classes is None:
        classes_arr = np.unique(np.concatenate([y_true_sel, y_pred_sel]))
    else:
        classes_arr = np.asarray(classes)

    acc = float(accuracy_score(y_true_sel, y_pred_sel))
    try:
        bacc = float(
            balanced_accuracy_score(y_true_sel, y_pred_sel)
        )
    except ValueError:
        bacc = float("nan")

    precision_by_class: dict[str, float] = {}
    try:
        prec = precision_score(
            y_true_sel,
            y_pred_sel,
            labels=classes_arr,
            average=None,
            zero_division=0,
        )
        for cls, p in zip(classes_arr.tolist(), np.asarray(prec).tolist()):
            precision_by_class[str(cls)] = float(p)
    except ValueError:
        precision_by_class = {}

    if accepted_by_class is None:
        accepted_by_class = {
            k: int(v) for k, v in Counter(y_pred_sel.tolist()).items()
        }

    conf_q = _quantiles(confidences) if confidences is not None else _empty_quantiles()
    dens_q = _quantiles(densities) if densities is not None else _empty_quantiles()
    marg_q = _quantiles(margins) if margins is not None else _empty_quantiles()

    # Class-prior drift: predicted PL prior vs labeled prior (or true hidden prior)
    pred_counts = Counter(y_pred_sel.tolist())
    n_sel = float(selected_indices.size)
    pred_prior = {str(k): v / n_sel for k, v in pred_counts.items()}
    if labeled_class_counts:
        total_lab = float(sum(labeled_class_counts.values())) or 1.0
        lab_prior = {
            str(k): float(v) / total_lab for k, v in labeled_class_counts.items()
        }
        all_keys = sorted(set(pred_prior) | set(lab_prior))
        class_prior_drift = {
            k: float(pred_prior.get(k, 0.0) - lab_prior.get(k, 0.0)) for k in all_keys
        }
    else:
        true_counts = Counter(y_true_sel.tolist())
        true_prior = {str(k): v / n_sel for k, v in true_counts.items()}
        all_keys = sorted(set(pred_prior) | set(true_prior))
        class_prior_drift = {
            k: float(pred_prior.get(k, 0.0) - true_prior.get(k, 0.0)) for k in all_keys
        }

    if n_labeled is not None and int(n_labeled) > 0:
        multiplier = float(selected_indices.size) / float(n_labeled)
    else:
        multiplier = float("nan")

    correct = y_true_sel == y_pred_sel
    out.update(
        {
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "precision_by_class": precision_by_class,
            "accepted_by_class": {str(k): int(v) for k, v in accepted_by_class.items()},
            "n_correct": int(np.sum(correct)),
            "confidence_mean": float(np.mean(confidences))
            if confidences is not None and len(confidences)
            else float("nan"),
            "confidence_quantiles": conf_q,
            "density_mean": float(np.mean(densities))
            if densities is not None and len(densities)
            else float("nan"),
            "density_quantiles": dens_q,
            "margin_quantiles": marg_q,
            "weight_mean": float(np.mean(weights))
            if weights is not None and len(weights)
            else float("nan"),
            "class_prior_drift": class_prior_drift,
            "agreement_rate": agreement_rate if agreement_rate is not None else float("nan"),
            "final_pseudo_label_multiplier": multiplier,
            "per_class_accuracy": _per_class_accuracy(y_true_sel, y_pred_sel),
        }
    )
    return out


def _per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for cls in np.unique(y_true):
        mask = y_true == cls
        if not np.any(mask):
            continue
        out[str(cls)] = float(np.mean(y_pred[mask] == y_true[mask]))
    return out


def _quantiles(x: np.ndarray | None) -> dict[str, float]:
    if x is None:
        return _empty_quantiles()
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return _empty_quantiles()
    q25, q50, q75 = np.quantile(arr, [0.25, 0.5, 0.75])
    return {
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _empty_quantiles() -> dict[str, float]:
    return {
        "q25": float("nan"),
        "q50": float("nan"),
        "q75": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
    }
