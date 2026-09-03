"""Classical graph and self-training model adapters for the legacy interface."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.graph_ssl import RobustGraphSSLModel
from src.models.self_training_utils import fit_self_training
from src.models.supervised import (
    build_catboost_estimator,
    build_lightgbm_estimator,
    build_xgboost_estimator,
)


class SelfTrainingLR:
    name = "self_training_lr"

    def __init__(self, random_state: int = 0, n_classes: int = 2) -> None:
        self.random_state = random_state
        self.n_classes = n_classes
        self.model = None
        self.training_meta: dict = {}

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
    ) -> "SelfTrainingLR":
        def factory():
            return LogisticRegression(max_iter=2000, random_state=self.random_state)

        self.model, self.training_meta = fit_self_training(
            name=self.name,
            base_estimator_factory=factory,
            X_labeled=X_labeled,
            y_labeled=y_labeled,
            X_unlabeled=X_unlabeled,
            random_state=self.random_state,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class SelfTrainingXGBoost:
    name = "self_training_xgboost"

    def __init__(self, random_state: int = 0, n_classes: int = 2) -> None:
        self.random_state = random_state
        self.n_classes = n_classes
        self.model = None
        self.training_meta: dict = {}

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
    ) -> "SelfTrainingXGBoost":
        def factory():
            return build_xgboost_estimator(self.random_state, self.n_classes)

        self.model, self.training_meta = fit_self_training(
            name=self.name,
            base_estimator_factory=factory,
            X_labeled=X_labeled,
            y_labeled=y_labeled,
            X_unlabeled=X_unlabeled,
            random_state=self.random_state,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class SelfTrainingLightGBM:
    name = "self_training_lightgbm"

    def __init__(self, random_state: int = 0, n_classes: int = 2) -> None:
        self.random_state = random_state
        self.n_classes = n_classes
        self.model = None
        self.training_meta: dict = {}

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
    ) -> "SelfTrainingLightGBM":
        def factory():
            return build_lightgbm_estimator(self.random_state)

        self.model, self.training_meta = fit_self_training(
            name=self.name,
            base_estimator_factory=factory,
            X_labeled=X_labeled,
            y_labeled=y_labeled,
            X_unlabeled=X_unlabeled,
            random_state=self.random_state,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class SelfTrainingCatBoost:
    name = "self_training_catboost"

    def __init__(self, random_state: int = 0, n_classes: int = 2) -> None:
        self.random_state = random_state
        self.n_classes = n_classes
        self.model = None
        self.training_meta: dict = {}

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
    ) -> "SelfTrainingCatBoost":
        def factory():
            return build_catboost_estimator(
                self.random_state,
                self.n_classes,
                iterations=150,
            )

        self.model, self.training_meta = fit_self_training(
            name=self.name,
            base_estimator_factory=factory,
            X_labeled=X_labeled,
            y_labeled=y_labeled,
            X_unlabeled=X_unlabeled,
            random_state=self.random_state,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class LabelSpreadingModel(RobustGraphSSLModel):
    name = "label_spreading"

    def __init__(self, random_state: int = 0, n_classes: int = 2) -> None:
        super().__init__(
            method="label_spreading",
            random_state=random_state,
            n_classes=n_classes,
        )


class LabelPropagationModel(RobustGraphSSLModel):
    name = "label_propagation"

    def __init__(self, random_state: int = 0, n_classes: int = 2) -> None:
        super().__init__(
            method="label_propagation",
            random_state=random_state,
            n_classes=n_classes,
        )


def build_semi_supervised_model(method: str, random_state: int = 0, n_classes: int = 2):
    if method == "self_training_lr":
        return SelfTrainingLR(random_state=random_state, n_classes=n_classes)
    if method == "self_training_xgboost":
        return SelfTrainingXGBoost(random_state=random_state, n_classes=n_classes)
    if method == "self_training_lightgbm":
        return SelfTrainingLightGBM(random_state=random_state, n_classes=n_classes)
    if method == "self_training_catboost":
        return SelfTrainingCatBoost(random_state=random_state, n_classes=n_classes)
    if method == "label_spreading":
        return LabelSpreadingModel(random_state=random_state, n_classes=n_classes)
    if method == "label_propagation":
        return LabelPropagationModel(random_state=random_state, n_classes=n_classes)
    raise ValueError(f"Unknown semi-supervised method: {method}")
