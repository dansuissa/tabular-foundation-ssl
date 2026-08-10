"""Leakage-safe processed feature view and fixed label encoding helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols, categorical_cols = infer_column_types(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("No usable columns found for preprocessing.")

    return ColumnTransformer(transformers=transformers)


def fit_label_encoder(y: pd.Series) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(y.astype(str))
    return encoder


def transform_labels(encoder: LabelEncoder, y: pd.Series) -> np.ndarray:
    return encoder.transform(y.astype(str))


def fit_transform_preprocessor(
    preprocessor: ColumnTransformer,
    X_labeled_train: pd.DataFrame,
    X_unlabeled_train: pd.DataFrame,
    *extra_sets: pd.DataFrame,
) -> tuple[np.ndarray, ...]:
    """Fit preprocessing on labeled + unlabeled train only, then transform all sets."""
    fit_X = pd.concat([X_labeled_train, X_unlabeled_train], axis=0, ignore_index=True)
    preprocessor.fit(fit_X)
    n_features = preprocessor.transform(fit_X[:1]).shape[1]
    return tuple(_transform_or_empty(preprocessor, dataset, n_features) for dataset in extra_sets)


def _transform_or_empty(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    n_features: int,
) -> np.ndarray:
    if len(X) == 0:
        return np.empty((0, n_features), dtype=np.float32)
    return np.asarray(preprocessor.transform(X), dtype=np.float32)
