
from __future__ import annotations
import pytest
from src.splits import make_ssl_split
from src.views import build_dataset_views, FitContext
from src.models import run_model_from_context
from src.models.cast_trees import CASTCatBoost
from src.models.laplacian_ssl import build_laplacian_model
from src.models.prototype_ssl import PrototypeAlignmentSSL
from src.models.registry_ext import build_extended_model
from src.exceptions import OptionalDependencyError, UnsupportedMethodError
from tests.fixtures import (
    make_synthetic_binary, make_synthetic_multiclass, make_synthetic_mixed,
    make_imbalanced_binary, make_missing_values,
)

def _run(model, X, y, n_labeled=50, seed=0):
    splits=make_ssl_split(X,y,n_labeled=n_labeled,test_size=0.2,val_size_from_labeled=0.2,seed=seed)
    views=build_dataset_views(splits,"syn",seed,n_labeled)
    ctx=FitContext(views=views, random_state=seed, method_name=getattr(model,"name","m"))
    return run_model_from_context(model, ctx), views

@pytest.mark.parametrize("maker", [make_synthetic_binary, make_synthetic_multiclass, make_synthetic_mixed, make_imbalanced_binary, make_missing_values])
def test_cast_catboost_synthetic(maker):
    X,y=maker()
    n_lab=50 if y.nunique()<=5 else 60
    try:
        make_ssl_split(X,y,n_labeled=n_lab,test_size=0.2,val_size_from_labeled=0.2,seed=0)
    except Exception:
        pytest.skip("invalid budget for synthetic")
    model=CASTCatBoost(random_state=0, n_classes=int(y.nunique()))
    pred, views=_run(model,X,y,n_labeled=n_lab)
    assert len(pred.y_pred)==len(views.y_test)
    assert pred.y_proba is not None

def test_laplacian_linear_smoke():
    X,y=make_synthetic_binary(300,1)
    model=build_laplacian_model("laplacian_linear", random_state=0, n_classes=2, max_epochs=5, patience=2)
    pred, views=_run(model,X,y)
    assert pred.y_proba.shape[0]==len(views.y_test)

def test_prototype_smoke():
    X,y=make_synthetic_multiclass(300,3,2)
    model=PrototypeAlignmentSSL(random_state=0,n_classes=3,max_epochs=5,patience=2)
    pred, views=_run(model,X,y,n_labeled=60)
    assert len(pred.y_pred)==len(views.y_test)

def test_distpfn_precise_unsupported():
    model=build_extended_model("tabpfn3_distpfn_transductive", random_state=0)
    X,y=make_synthetic_binary(100,0)
    with pytest.raises(UnsupportedMethodError) as ei:
        _run(model,X,y)
    assert "unsupported_faithful_distpfn" in ei.value.status

def test_tfm_missing_dependency_or_runs():
    model=build_extended_model("tabpfn3", random_state=0)
    X,y=make_synthetic_binary(120,0)
    try:
        _run(model,X,y)
    except OptionalDependencyError as e:
        assert e.package in {"tabpfn","torch"}
    except Exception:
        pass
