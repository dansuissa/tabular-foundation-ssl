
from __future__ import annotations
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from src.splits import make_ssl_split, DataSplits
from src.views import build_dataset_views
from src.models.sparse_graph import build_sparse_knn_graph
from tests.fixtures import make_synthetic_multiclass

def test_graph_nodes_exclude_val_test():
    X,y = make_synthetic_multiclass(n=400, n_classes=3, seed=0)
    splits = make_ssl_split(X,y,n_labeled=60,test_size=0.2,val_size_from_labeled=0.2,seed=0)
    views = build_dataset_views(splits,"mc",0,60)
    n_train = len(views.X_labeled_processed)+len(views.X_unlabeled_processed)
    Xg = np.vstack([views.X_labeled_processed, views.X_unlabeled_processed])
    g = build_sparse_knn_graph(Xg, k=5, mutual=True, random_state=0)
    assert g.n_nodes == n_train

def test_label_encoder_not_fit_on_unlabeled_only_classes():
    X = pd.DataFrame(np.random.RandomState(0).randn(20,3), columns=list("abc"))
    splits = DataSplits(
        X_labeled_train=X.iloc[:8].reset_index(drop=True),
        y_labeled_train=pd.Series([0,0,0,0,1,1,1,1]),
        X_unlabeled_train=X.iloc[8:16].reset_index(drop=True),
        y_unlabeled_train=pd.Series([0,1,2,2,2,1,0,1]),
        X_val=X.iloc[16:18].reset_index(drop=True),
        y_val=pd.Series([0,1]),
        X_test=X.iloc[18:].reset_index(drop=True),
        y_test=pd.Series([0,1]),
        n_labeled=8,n_unlabeled=8,train_labeled_size=8,val_size=2,test_size=2,
        validation_strategy="stratified_labeled_val",labeled_classes_present=2,
        all_classes_present_in_labeled=False,min_labeled_per_class=4,max_labeled_per_class=4,
        train_pool_class_counts="{}",labeled_class_counts="{}",val_class_counts="{}",
        test_class_counts="{}",unlabeled_class_counts="{}",
    )
    views = build_dataset_views(splits,"fab",0,8)
    assert "2" not in set(views.label_encoder.classes_)


def test_model_implementations_cannot_read_test_labels():
    """Keep evaluation labels out of every model's fit/predict implementation.

    The runner owns metric computation and therefore legitimately reads
    ``views.y_test``. Model modules receive the common context for feature-view
    dispatch, but accessing the test target there would violate the protocol.
    """
    model_root = Path(__file__).resolve().parents[1] / "src" / "models"
    violations: list[str] = []
    for path in model_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "y_test":
                violations.append(f"{path.relative_to(model_root)}:{node.lineno}")
    assert not violations, f"model code references test labels: {violations}"
