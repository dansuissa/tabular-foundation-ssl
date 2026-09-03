"""Supervised benchmark baselines with lazy optional boosting dependencies."""
from __future__ import annotations

import numpy as np
from typing import Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def _require_xgboost():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is not installed. Install it with: pip install xgboost"
        ) from exc
    return XGBClassifier


def _require_lightgbm():
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "lightgbm is not installed. Install it with: pip install lightgbm"
        ) from exc
    return LGBMClassifier


def _require_catboost():
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError(
            "catboost is not installed. Install it with: pip install catboost"
        ) from exc
    return CatBoostClassifier


class LogisticRegressionBaseline:
    name = "logistic_regression"

    def __init__(self, random_state: int = 0) -> None:
        self.random_state = random_state
        self.model = LogisticRegression(max_iter=2000, random_state=random_state)
        self.training_meta: dict = {}

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
    ) -> "LogisticRegressionBaseline":
        self.model.fit(X_labeled, y_labeled)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class RandomForestBaseline:
    name = "random_forest"

    def __init__(self, random_state: int = 0) -> None:
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        )
        self.training_meta: dict = {}

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
    ) -> "RandomForestBaseline":
        self.model.fit(X_labeled, y_labeled)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class XGBoostBaseline:
    name = "xgboost"

    def __init__(self, random_state: int = 0, n_classes: int = 2) -> None:
        XGBClassifier = _require_xgboost()
        self.random_state = random_state
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "n_jobs": -1,
            "random_state": random_state,
            "tree_method": "hist",
        }
        if n_classes > 2:
            params["objective"] = "multi:softprob"
            params["num_class"] = n_classes
        self.model = XGBClassifier(**params)
        self.training_meta: dict = {}

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
    ) -> "XGBoostBaseline":
        self.model.fit(X_labeled, y_labeled)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class _LGBMStableInputWrapper:
    """LightGBM wrapper with consistent DataFrame inputs.

    This avoids sklearn's feature-name warning by ensuring that both fit and
    predict/predict_proba see stable column names.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self._feature_names: list[str] | None = None

    def _to_frame(self, X: np.ndarray):
        import pandas as pd

        X = np.asarray(X, dtype=np.float32)
        if self._feature_names is None:
            self._feature_names = [f"f{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=self._feature_names)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_df = self._to_frame(X)
        self.model.fit(X_df, y)
        return self

    @property
    def classes_(self):
        return self.model.classes_

    @property
    def n_classes_(self):
        return getattr(self.model, "n_classes_", None)

    @property
    def n_features_in_(self):
        return getattr(self.model, "n_features_in_", None)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._to_frame(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._to_frame(X))

    def __getattr__(self, name: str):
        return getattr(self.model, name)


class LightGBMBaseline:
    name = "lightgbm"

    def __init__(self, random_state: int = 0) -> None:
        LGBMClassifier = _require_lightgbm()
        self.random_state = random_state
        self.model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.training_meta: dict = {}

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
    ) -> "LightGBMBaseline":
        self.model.fit(np.asarray(X_labeled, dtype=np.float32), y_labeled)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(X, dtype=np.float32))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.asarray(X, dtype=np.float32))


class CatBoostBaseline:
    name = "catboost"

    def __init__(self, random_state: int = 0, n_classes: int = 2) -> None:
        CatBoostClassifier = _require_catboost()
        self.random_state = random_state
        loss_function = "Logloss" if n_classes <= 2 else "MultiClass"
        self.model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            loss_function=loss_function,
        )
        self.training_meta: dict = {}

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
    ) -> "CatBoostBaseline":
        self.model.fit(X_labeled, y_labeled)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class MLPBaseline:
    name = "mlp"

    def __init__(self, random_state: int = 0) -> None:
        self.random_state = random_state
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
        )
        self.training_meta: dict = {}

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
    ) -> "MLPBaseline":
        self.model.fit(X_labeled, y_labeled)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def build_supervised_model(method: str, random_state: int = 0, n_classes: int = 2):
    if method == "logistic_regression":
        return LogisticRegressionBaseline(random_state=random_state)
    if method == "random_forest":
        return RandomForestBaseline(random_state=random_state)
    if method == "xgboost":
        return XGBoostBaseline(random_state=random_state, n_classes=n_classes)
    if method == "lightgbm":
        return LightGBMBaseline(random_state=random_state)
    if method == "catboost":
        return CatBoostBaseline(random_state=random_state, n_classes=n_classes)
    if method == "mlp":
        return MLPBaseline(random_state=random_state)
    raise ValueError(f"Unknown supervised method: {method}")


def build_xgboost_estimator(random_state: int, n_classes: int):
    return build_supervised_model("xgboost", random_state=random_state, n_classes=n_classes).model


def build_lightgbm_estimator(random_state: int):
    LGBMClassifier = _require_lightgbm()
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    return _LGBMStableInputWrapper(model)


def build_catboost_estimator(
    random_state: int,
    n_classes: int,
    iterations: int = 500,
):
    CatBoostClassifier = _require_catboost()
    loss_function = "Logloss" if n_classes <= 2 else "MultiClass"
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        loss_function=loss_function,
    )
