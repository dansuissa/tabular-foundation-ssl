
from __future__ import annotations
import numpy as np
from src.splits import make_ssl_split
from src.views import build_dataset_views, FitContext
from src.models import run_model, run_model_from_context
from src.models.supervised import build_supervised_model
from tests.fixtures import make_synthetic_binary

def test_logistic_parity_context_vs_ndarray():
    X,y=make_synthetic_binary(250,0)
    splits=make_ssl_split(X,y,n_labeled=50,test_size=0.2,val_size_from_labeled=0.2,seed=0)
    views=build_dataset_views(splits,"b",0,50)
    m1=build_supervised_model("logistic_regression", random_state=0)
    m2=build_supervised_model("logistic_regression", random_state=0)
    r1=run_model(m1, views.X_labeled_processed, views.y_labeled, views.X_unlabeled_processed, views.X_test_processed)
    ctx=FitContext(views=views, random_state=0, method_name="logistic_regression")
    r2=run_model_from_context(m2, ctx)
    assert np.allclose(r1.y_proba, r2.y_proba, atol=1e-10)
    assert np.array_equal(r1.y_pred, r2.y_pred)
