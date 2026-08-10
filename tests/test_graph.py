
from __future__ import annotations
import numpy as np
from src.models.sparse_graph import build_sparse_knn_graph, laplacian_loss_from_proba

def test_graph_symmetry_sparsity_no_self():
    rng=np.random.RandomState(0)
    X=rng.randn(40,5)
    g=build_sparse_knn_graph(X,k=5,mutual=True,random_state=0)
    A=g.adjacency
    assert A.shape==(40,40)
    diff=(A-A.T)
    assert diff.nnz==0 or np.allclose(diff.data, 0, atol=1e-8)
    assert np.all(A.diagonal()==0)
    assert np.all(A.data>=0) and np.all(np.isfinite(A.data))
    proba=rng.dirichlet(np.ones(3), size=40)
    loss=laplacian_loss_from_proba(proba,A)
    assert np.isfinite(loss) and loss>=0
