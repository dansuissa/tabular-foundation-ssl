"""Sparse kNN graph construction for Laplacian-regularized SSL (train pool only)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


@dataclass
class SparseGraph:
    adjacency: sparse.csr_matrix  # symmetric, zero diagonal, nonnegative
    n_nodes: int
    n_edges: int  # undirected edge count (upper triangle)
    k: int
    backend: str
    mutual: bool
    build_time_seconds: float
    affinity_mean: float
    affinity_std: float
    n_connected_components: int
    n_isolated_nodes: int
    meta: dict[str, Any]


def _local_scaling_affinities(
    distances: np.ndarray,
    indices: np.ndarray,
    self_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust local-scaling RBF using k-th neighbor distance as scale."""
    n, k = distances.shape
    # Exclude self if present at position 0
    scales = distances[:, -1].copy()
    scales = np.maximum(scales, 1e-8)
    rows = []
    cols = []
    vals = []
    for i in range(n):
        for j_pos in range(k):
            j = int(indices[i, j_pos])
            if j == i:
                continue
            d = float(distances[i, j_pos])
            sigma_i = float(scales[i])
            sigma_j = float(scales[j])
            w = float(np.exp(-(d * d) / (sigma_i * sigma_j)))
            if not np.isfinite(w) or w < 0:
                continue
            rows.append(i)
            cols.append(j)
            vals.append(w)
    return np.asarray(rows), np.asarray(cols), np.asarray(vals, dtype=np.float64)


def _symmetrize_and_clean(
    n: int, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, mutual: bool
) -> sparse.csr_matrix:
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    if mutual:
        # Keep edge only if both directions exist.
        A = A.minimum(A.T)
        A = ((A + A.T) * 0.5).tocsr()
    else:
        A = ((A + A.T) * 0.5).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    # Validate nonnegative finite
    data = A.data
    if data.size and (np.any(~np.isfinite(data)) or np.any(data < 0)):
        data = np.where(np.isfinite(data) & (data >= 0), data, 0.0)
        A.data = data
        A.eliminate_zeros()
    return A


def _graph_stats(A: sparse.csr_matrix) -> tuple[int, int, float, float, int, int]:
    n = A.shape[0]
    # Undirected edges
    n_edges = int(A.nnz // 2)
    data = A.data
    aff_mean = float(data.mean()) if data.size else 0.0
    aff_std = float(data.std()) if data.size else 0.0
    # Connected components via scipy
    n_comp, labels = sparse.csgraph.connected_components(A, directed=False)
    degrees = np.asarray(A.sum(axis=1)).ravel()
    n_isolated = int(np.sum(degrees <= 0))
    return n_edges, n_comp, aff_mean, aff_std, n_isolated, int(n_comp)


def build_sparse_knn_graph(
    X: np.ndarray,
    k: int = 10,
    mutual: bool = True,
    backend: Literal["auto", "sklearn", "pynndescent", "faiss"] = "auto",
    random_state: int = 0,
    metric: str = "euclidean",
) -> SparseGraph:
    """Build sparse mutual/non-mutual kNN graph on labeled∪unlabeled train only.

    Never includes validation or test nodes. Never materializes dense n×n affinity.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n = X.shape[0]
    if n == 0:
        empty = sparse.csr_matrix((0, 0))
        return SparseGraph(
            adjacency=empty,
            n_nodes=0,
            n_edges=0,
            k=k,
            backend="empty",
            mutual=mutual,
            build_time_seconds=0.0,
            affinity_mean=0.0,
            affinity_std=0.0,
            n_connected_components=0,
            n_isolated_nodes=0,
            meta={},
        )
    k_eff = int(min(max(k, 1), max(n - 1, 1)))
    t0 = time.perf_counter()
    used_backend = "sklearn"
    distances: np.ndarray
    indices: np.ndarray

    if backend in ("auto", "pynndescent") and n >= 5000:
        try:
            import pynndescent

            index = pynndescent.NNDescent(
                X, n_neighbors=k_eff + 1, metric=metric, random_state=random_state
            )
            indices, distances = index.neighbor_graph
            used_backend = "pynndescent"
        except Exception:
            used_backend = "sklearn"

    if used_backend == "sklearn" and backend == "faiss":
        try:
            import faiss  # type: ignore

            xb = np.ascontiguousarray(X.astype(np.float32))
            index = faiss.IndexFlatL2(xb.shape[1])
            index.add(xb)
            distances, indices = index.search(xb, k_eff + 1)
            distances = np.sqrt(np.maximum(distances, 0.0))
            used_backend = "faiss"
        except Exception:
            used_backend = "sklearn"

    if used_backend == "sklearn":
        nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, n_jobs=1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

    rows, cols, vals = _local_scaling_affinities(distances, indices, np.arange(n))
    A = _symmetrize_and_clean(n, rows, cols, vals, mutual=mutual)
    n_edges, n_comp, aff_mean, aff_std, n_isolated, _ = _graph_stats(A)
    build_time = time.perf_counter() - t0
    return SparseGraph(
        adjacency=A,
        n_nodes=n,
        n_edges=n_edges,
        k=k_eff,
        backend=used_backend,
        mutual=mutual,
        build_time_seconds=build_time,
        affinity_mean=aff_mean,
        affinity_std=aff_std,
        n_connected_components=n_comp,
        n_isolated_nodes=n_isolated,
        meta={"metric": metric},
    )


def laplacian_loss_from_proba(
    proba: np.ndarray,
    adjacency: sparse.csr_matrix,
    edge_batch_size: int | None = None,
) -> float:
    """L_lap = sum_ij w_ij ||p_i - p_j||^2 / sum_ij w_ij."""
    proba = np.asarray(proba, dtype=np.float64)
    A = adjacency.tocoo()
    if A.nnz == 0:
        return 0.0
    if edge_batch_size is None or A.nnz <= edge_batch_size:
        diff = proba[A.row] - proba[A.col]
        num = float(np.sum(A.data * np.sum(diff * diff, axis=1)))
        den = float(np.sum(A.data))
        return num / max(den, 1e-12)
    # Mini-batch over edges
    num = 0.0
    den = 0.0
    for start in range(0, A.nnz, edge_batch_size):
        end = min(A.nnz, start + edge_batch_size)
        rows = A.row[start:end]
        cols = A.col[start:end]
        w = A.data[start:end]
        diff = proba[rows] - proba[cols]
        num += float(np.sum(w * np.sum(diff * diff, axis=1)))
        den += float(np.sum(w))
    return num / max(den, 1e-12)


def adjacency_to_edge_tensors(adjacency: sparse.csr_matrix):
    """Convert CSR adjacency to (row, col, weight) numpy arrays for torch."""
    A = adjacency.tocoo()
    return (
        np.asarray(A.row, dtype=np.int64),
        np.asarray(A.col, dtype=np.int64),
        np.asarray(A.data, dtype=np.float32),
    )


def laplacian_loss_torch(
    proba,
    row_idx,
    col_idx,
    weights,
    edge_batch_size: int | None = None,
):
    """Differentiable Laplacian smoothness on probability (or embedding) rows.

    L = sum_e w_e ||p_i - p_j||^2 / sum_e w_e
    ``proba`` is (N, C) torch tensor; edge tensors may be torch or numpy.
    """
    import torch

    if isinstance(row_idx, np.ndarray):
        row_idx = torch.as_tensor(row_idx, device=proba.device, dtype=torch.long)
        col_idx = torch.as_tensor(col_idx, device=proba.device, dtype=torch.long)
        weights = torch.as_tensor(weights, device=proba.device, dtype=proba.dtype)
    n_edges = int(row_idx.shape[0])
    if n_edges == 0:
        return proba.new_zeros(())
    if edge_batch_size is None or n_edges <= edge_batch_size:
        diff = proba[row_idx] - proba[col_idx]
        num = (weights * (diff * diff).sum(dim=-1)).sum()
        den = weights.sum().clamp_min(1e-12)
        return num / den
    num = proba.new_zeros(())
    den = proba.new_zeros(())
    for start in range(0, n_edges, edge_batch_size):
        end = min(n_edges, start + edge_batch_size)
        r = row_idx[start:end]
        c = col_idx[start:end]
        w = weights[start:end]
        diff = proba[r] - proba[c]
        num = num + (w * (diff * diff).sum(dim=-1)).sum()
        den = den + w.sum()
    return num / den.clamp_min(1e-12)


def graph_meta_dict(graph: SparseGraph) -> dict[str, Any]:
    """Flatten SparseGraph fields into a training_meta-friendly dict."""
    return {
        "graph_backend": graph.backend,
        "graph_n_nodes": graph.n_nodes,
        "graph_n_edges": graph.n_edges,
        "graph_k": graph.k,
        "graph_mutual": graph.mutual,
        "graph_build_time_seconds": graph.build_time_seconds,
        "graph_affinity_mean": graph.affinity_mean,
        "graph_affinity_std": graph.affinity_std,
        "graph_n_connected_components": graph.n_connected_components,
        "graph_n_isolated_nodes": graph.n_isolated_nodes,
        **{f"graph_{k}": v for k, v in graph.meta.items()},
    }
