"""Synthetic datasets and leakage assertions for integration tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_binary(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 6)
    logits = X[:, 0] * 1.2 + X[:, 1] * -0.8 + rng.randn(n) * 0.3
    y = (logits > 0).astype(int)
    cols = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


def make_synthetic_multiclass(n: int = 600, n_classes: int = 4, seed: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 8)
    centers = rng.randn(n_classes, 8) * 2.0
    # Assign by nearest center + noise
    dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    y = dist.argmin(axis=1)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]), pd.Series(y, name="y")


def make_synthetic_mixed(n: int = 500, seed: int = 2) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    num = rng.randn(n, 4)
    cat = rng.choice(["a", "b", "c"], size=(n, 2))
    y = ((num[:, 0] + (cat[:, 0] == "a").astype(float) - 0.5) > 0).astype(int)
    df = pd.DataFrame(num, columns=[f"n{i}" for i in range(4)])
    df["c0"] = cat[:, 0]
    df["c1"] = cat[:, 1]
    return df, pd.Series(y, name="y")


def make_imbalanced_binary(n: int = 800, pos_frac: float = 0.08, seed: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    n_pos = max(2, int(n * pos_frac))
    n_neg = n - n_pos
    X_neg = rng.randn(n_neg, 5)
    X_pos = rng.randn(n_pos, 5) + 1.5
    X = np.vstack([X_neg, X_pos])
    y = np.array([0] * n_neg + [1] * n_pos)
    idx = rng.permutation(n)
    return pd.DataFrame(X[idx], columns=[f"f{i}" for i in range(5)]), pd.Series(y[idx], name="y")


def make_missing_values(n: int = 300, seed: int = 4) -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_synthetic_binary(n=n, seed=seed)
    rng = np.random.RandomState(seed + 9)
    mask = rng.rand(*X.shape) < 0.1
    X = X.mask(mask)
    return X, y


class LeakageProbe:
    """Wrap arrays/frames to detect illicit access during fit."""

    def __init__(self, name: str, data):
        self.name = name
        self.data = data
        self.access_count = 0
        self.allowed = True

    def forbid(self) -> None:
        self.allowed = False

    def allow(self) -> None:
        self.allowed = True

    def get(self):
        self.access_count += 1
        if not self.allowed:
            raise AssertionError(f"Leakage: accessed forbidden view '{self.name}'")
        return self.data
