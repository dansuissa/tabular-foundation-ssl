
from __future__ import annotations
import numpy as np
from src.splits import make_ssl_split
from src.views import build_dataset_views
from tests.fixtures import make_synthetic_binary, make_synthetic_mixed

def test_raw_and_processed_views_and_label_encoder_policy():
    X,y = make_synthetic_binary(n=300, seed=0)
    splits = make_ssl_split(X,y,n_labeled=50,test_size=0.2,val_size_from_labeled=0.2,seed=0)
    views = build_dataset_views(splits,"synth",seed=0,n_labeled=50)
    assert views.X_labeled_raw.shape[0]==splits.X_labeled_train.shape[0]
    assert views.X_labeled_processed.dtype==np.float32
    assert views.X_test_processed.shape[0]==len(splits.y_test)
    enc_classes=set(views.label_encoder.classes_)
    lab=set(splits.y_labeled_train.astype(str)) | set(splits.y_val.astype(str))
    assert enc_classes.issubset(lab) or enc_classes==lab
    assert len(views.y_labeled)==len(splits.y_labeled_train)

def test_mixed_types_preserve_raw_columns():
    X,y = make_synthetic_mixed(n=250, seed=2)
    splits = make_ssl_split(X,y,n_labeled=50,test_size=0.2,val_size_from_labeled=0.2,seed=1)
    views = build_dataset_views(splits,"mixed",seed=1,n_labeled=50)
    assert "c0" in views.X_labeled_raw.columns
    assert views.n_features_processed >= 1
