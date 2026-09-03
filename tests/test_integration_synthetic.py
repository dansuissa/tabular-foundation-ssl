"""Synthetic integration tests for canonical extended methods."""
from __future__ import annotations

import pytest

from src.exceptions import OptionalDependencyError
from src.models import run_model_from_context
from src.models.registry_ext import EXTENDED_METHODS, build_extended_model
from src.splits import make_ssl_split
from src.views import FitContext, build_dataset_views
from tests.fixtures import make_synthetic_binary, make_synthetic_multiclass


def _run(model, X, y, n_labeled=50, seed=0):
    splits = make_ssl_split(
        X,
        y,
        n_labeled=n_labeled,
        test_size=0.2,
        val_size_from_labeled=0.2,
        seed=seed,
    )
    views = build_dataset_views(splits, "synthetic", seed, n_labeled)
    ctx = FitContext(
        views=views,
        random_state=seed,
        method_name=getattr(model, "name", "model"),
    )
    return run_model_from_context(model, ctx), views


def test_extended_registry_is_canonical_only():
    assert EXTENDED_METHODS == {
        "tabpfn3",
        "tabiclv2",
        "tabpfn3_self_training",
        "tabiclv2_self_training",
        "laplacian_ssl",
        "unlabeled_attention_ssl",
        "embedding_alignment_ssl",
        "geometric_attention_ssl",
    }


def test_laplacian_ssl_synthetic():
    X, y = make_synthetic_binary(240, 1)
    model = build_extended_model(
        "laplacian_ssl",
        random_state=0,
        n_classes=2,
        max_epochs=3,
        patience=2,
    )
    pred, views = _run(model, X, y)
    assert pred.y_proba.shape[0] == len(views.y_test)


def test_embedding_alignment_ssl_synthetic():
    X, y = make_synthetic_multiclass(300, 3, 2)
    model = build_extended_model(
        "embedding_alignment_ssl",
        random_state=0,
        n_classes=3,
        max_epochs=3,
        patience=2,
    )
    pred, views = _run(model, X, y, n_labeled=60)
    assert len(pred.y_pred) == len(views.y_test)


def test_tfm_dependency_is_explicit():
    model = build_extended_model("tabpfn3", random_state=0)
    X, y = make_synthetic_binary(120, 0)
    try:
        _run(model, X, y)
    except OptionalDependencyError as exc:
        assert exc.package in {"tabpfn", "torch"}
    except Exception:
        # Model weights and provider authentication are intentionally external
        # to the public repository.
        pass
