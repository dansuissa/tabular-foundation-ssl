"""OpenML dataset specifications, loading, and deterministic row capping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    openml_id: int
    target_column: str = "class"
    max_rows: int | None = None


@dataclass
class TabularDataset:
    name: str
    openml_id: int
    target_column: str
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]
    n_classes: int
    task_type: str


def _subsample_max_rows(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=max_rows,
        random_state=seed,
    )
    keep_idx, _ = next(splitter.split(X, y))
    return (
        X.iloc[keep_idx].reset_index(drop=True),
        y.iloc[keep_idx].reset_index(drop=True),
    )


def load_dataset(spec: DatasetSpec) -> TabularDataset:
    """Load a tabular classification dataset from OpenML."""
    bunch = fetch_openml(
        data_id=spec.openml_id,
        as_frame=True,
        parser="auto",
    )
    X = bunch.data.copy()
    y = bunch.target.copy()

    if spec.target_column in X.columns:
        y = X.pop(spec.target_column)

    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(
                f"Expected a single target column for dataset {spec.name}, "
                f"got shape {y.shape}."
            )
        y = y.iloc[:, 0]

    y = y.astype(str)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if spec.max_rows is not None and len(X) > spec.max_rows:
        X, y = _subsample_max_rows(X, y, spec.max_rows)

    n_classes = y.nunique(dropna=True)
    task_type = "binary" if n_classes == 2 else "multiclass"

    return TabularDataset(
        name=spec.name,
        openml_id=spec.openml_id,
        target_column=spec.target_column,
        X=X,
        y=y,
        feature_names=list(X.columns),
        n_classes=int(n_classes),
        task_type=task_type,
    )


def dataset_from_config(entry: dict[str, Any]) -> DatasetSpec:
    max_rows = entry.get("max_rows")
    return DatasetSpec(
        name=entry["name"],
        openml_id=int(entry["openml_id"]),
        target_column=entry.get("target_column", "class"),
        max_rows=int(max_rows) if max_rows is not None else None,
    )
