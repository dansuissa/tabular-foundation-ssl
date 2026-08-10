"""Dual-view containers: raw pandas + shared-processed numpy arrays.

Leakage fence: LabelEncoder is fit only on labeled∪validation labels.
Unlabeled true labels and test labels never enter the encoder fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import (
    build_preprocessor,
    fit_label_encoder,
    fit_transform_preprocessor,
    transform_labels,
)
from src.splits import DataSplits

LabelEncoderPolicy = Literal["labeled_plus_val", "labeled_only"]


@dataclass(frozen=True)
class DatasetViews:
    """Immutable dual-view bundle for one (dataset, seed, n_labeled) cell.

    Raw frames keep original column order and dtypes for native TFM input.
    Processed arrays use the shared classical preprocessor.
    Empty validation is represented as empty frames/arrays (not None).
    """

    # Raw feature views (column order preserved)
    X_labeled_raw: pd.DataFrame
    X_unlabeled_raw: pd.DataFrame
    X_validation_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    # Processed feature views
    X_labeled_processed: np.ndarray
    X_unlabeled_processed: np.ndarray
    X_validation_processed: np.ndarray
    X_test_processed: np.ndarray
    # Encoded labels (ints); validation may be length-0
    y_labeled: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder
    class_names: tuple[str, ...]
    # Meta
    dataset: str
    seed: int
    n_labeled: int
    validation_strategy: str
    n_classes: int

    @property
    def has_validation(self) -> bool:
        return int(self.X_validation_raw.shape[0]) > 0

    @property
    def n_features_processed(self) -> int:
        return int(self.X_labeled_processed.shape[1])


@dataclass
class FitContext:
    """Unified method context; hidden unlabeled truth is never included.

    ``DatasetViews.y_test`` is retained for scoring by the runner after
    prediction. Fit implementations must not read it; a source-level leakage
    test enforces that no module under ``src/models`` references ``y_test``.
    """

    views: DatasetViews
    random_state: int
    method_config: dict[str, Any] = field(default_factory=dict)
    method_name: str = ""


@dataclass
class PredictionResult:
    y_pred: np.ndarray
    y_proba: np.ndarray | None = None
    training_meta: dict[str, Any] = field(default_factory=dict)


def _preserve_columns(frame: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    """Return a copy with a fixed column order (raises if columns diverge)."""
    if list(frame.columns) != list(columns):
        missing = [c for c in columns if c not in frame.columns]
        extra = [c for c in frame.columns if c not in columns]
        raise ValueError(
            f"Column order/set mismatch across splits. missing={missing} extra={extra}"
        )
    return frame.loc[:, columns].copy()


def _label_series_for_encoder(
    splits: DataSplits,
    policy: LabelEncoderPolicy,
) -> pd.Series:
    if policy == "labeled_only":
        return splits.y_labeled_train.copy()
    if policy == "labeled_plus_val":
        parts = [splits.y_labeled_train]
        if len(splits.y_val) > 0:
            parts.append(splits.y_val)
        return pd.concat(parts, ignore_index=True)
    raise ValueError(
        f"Unknown label_encoder_classes_policy={policy!r}. "
        "Supported: 'labeled_plus_val', 'labeled_only'."
    )


def _encode_or_empty(encoder: LabelEncoder, y: pd.Series) -> np.ndarray:
    if len(y) == 0:
        return np.empty(0, dtype=np.int64)
    return np.asarray(transform_labels(encoder, y), dtype=np.int64)


def build_dataset_views(
    splits: DataSplits,
    dataset_name: str,
    seed: int,
    n_labeled: int,
    label_encoder_classes_policy: LabelEncoderPolicy = "labeled_plus_val",
) -> DatasetViews:
    """Build leakage-safe raw + processed views from ``DataSplits``.

    Critical policy (default ``labeled_plus_val``):
      LabelEncoder fits ONLY on labeled_train ∪ validation labels.
      Unlabeled true labels are never used for class mapping.
    """
    columns = splits.X_labeled_train.columns

    X_labeled_raw = _preserve_columns(splits.X_labeled_train, columns)
    X_unlabeled_raw = _preserve_columns(splits.X_unlabeled_train, columns)
    X_validation_raw = _preserve_columns(splits.X_val, columns)
    X_test_raw = _preserve_columns(splits.X_test, columns)

    # Label encoder: labeled (+ val) only — never unlabeled y.
    y_for_encoder = _label_series_for_encoder(splits, label_encoder_classes_policy)
    label_encoder = fit_label_encoder(y_for_encoder)
    y_labeled = _encode_or_empty(label_encoder, splits.y_labeled_train)
    y_validation = _encode_or_empty(label_encoder, splits.y_val)
    y_test = _encode_or_empty(label_encoder, splits.y_test)

    # Preprocessor fit on labeled + unlabeled train features only.
    preprocessor = build_preprocessor(X_labeled_raw)
    (
        X_labeled_processed,
        X_unlabeled_processed,
        X_validation_processed,
        X_test_processed,
    ) = fit_transform_preprocessor(
        preprocessor,
        X_labeled_raw,
        X_unlabeled_raw,
        X_labeled_raw,
        X_unlabeled_raw,
        X_validation_raw,
        X_test_raw,
    )

    class_names = tuple(str(c) for c in label_encoder.classes_)
    return DatasetViews(
        X_labeled_raw=X_labeled_raw,
        X_unlabeled_raw=X_unlabeled_raw,
        X_validation_raw=X_validation_raw,
        X_test_raw=X_test_raw,
        X_labeled_processed=X_labeled_processed,
        X_unlabeled_processed=X_unlabeled_processed,
        X_validation_processed=X_validation_processed,
        X_test_processed=X_test_processed,
        y_labeled=y_labeled,
        y_validation=y_validation,
        y_test=y_test,
        label_encoder=label_encoder,
        class_names=class_names,
        dataset=dataset_name,
        seed=int(seed),
        n_labeled=int(n_labeled),
        validation_strategy=splits.validation_strategy,
        n_classes=len(class_names),
    )
