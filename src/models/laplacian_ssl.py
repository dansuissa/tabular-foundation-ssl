"""Sparse Laplacian-regularized SSL on processed tabular features.

Graph is built ONLY on labeled ∪ unlabeled train (never val/test).
PyTorch is optional; missing torch raises OptionalDependencyError.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.exceptions import OptionalDependencyError
from src.models.sparse_graph import (
    adjacency_to_edge_tensors,
    build_sparse_knn_graph,
    graph_meta_dict,
    laplacian_loss_torch,
)
from src.models.torch_utils import (
    internal_val_split,
    iterate_minibatches,
    make_mlp,
    normalize_probability_matrix,
    set_global_determinism,
    to_dense_float32,
    validate_probability_matrix,
)

LAPLACIAN_METHODS = {"laplacian_linear", "laplacian_mlp"}

_COMMON_DEFAULTS: dict[str, Any] = {
    "graph_k": 10,
    "graph_mutual": True,
    "graph_backend": "auto",
    "lambda_lap": 0.5,
    "edge_batch_size": 50_000,
    "batch_size": 128,
    "max_epochs": 30,
    "patience": 5,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "device": "cpu",
    "val_fraction": 0.2,
    "ramp_epochs": 5,
}

LINEAR_DEFAULTS = {**_COMMON_DEFAULTS}

MLP_DEFAULTS: dict[str, Any] = {
    **_COMMON_DEFAULTS,
    "hidden_dim": 64,
    "embedding_dim": 32,
    "depth": 2,
    "dropout": 0.1,
    "lambda_emb_lap": 0.1,
    "use_embedding_laplacian": True,
}


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError(
            "torch",
            "Install torch to run laplacian_linear / laplacian_mlp.",
        ) from exc
    return torch


def _ramp_weight(epoch: int, ramp_epochs: int) -> float:
    if ramp_epochs <= 0:
        return 1.0
    return float(min(1.0, (epoch + 1) / float(ramp_epochs)))


def _collapse_diagnostics(proba: np.ndarray, embeddings: np.ndarray | None = None) -> dict[str, Any]:
    proba = np.asarray(proba, dtype=np.float64)
    mean_p = proba.mean(axis=0)
    pred = proba.argmax(axis=1)
    counts = np.bincount(pred, minlength=proba.shape[1]).astype(np.float64)
    frac = counts / max(counts.sum(), 1.0)
    entropy = float(-(proba * np.log(np.clip(proba, 1e-12, 1.0))).sum(axis=1).mean())
    out: dict[str, Any] = {
        "mean_class_proba": mean_p.tolist(),
        "pred_class_fraction": frac.tolist(),
        "max_pred_class_fraction": float(frac.max()) if frac.size else float("nan"),
        "mean_prediction_entropy": entropy,
        "constant_prediction_suspect": bool(frac.max() >= 0.98) if frac.size else False,
    }
    if embeddings is not None and embeddings.size:
        emb = np.asarray(embeddings, dtype=np.float64)
        var = emb.var(axis=0)
        out["embedding_variance_mean"] = float(var.mean())
        out["embedding_variance_min"] = float(var.min())
        out["representation_collapse_suspect"] = bool(var.mean() < 1e-6)
    else:
        out["embedding_variance_mean"] = float("nan")
        out["representation_collapse_suspect"] = False
    return out


def _prepare_xy(X_labeled, y_labeled, X_unlabeled):
    X_labeled = to_dense_float32(X_labeled)
    y_labeled = np.asarray(y_labeled)
    classes = np.unique(y_labeled)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_local = np.array([class_to_idx[c] for c in y_labeled], dtype=np.int64)
    if X_unlabeled is None or len(X_unlabeled) == 0:
        X_unlabeled = np.empty((0, X_labeled.shape[1]), dtype=np.float32)
    else:
        X_unlabeled = to_dense_float32(X_unlabeled)
    return X_labeled, y_local, X_unlabeled, classes


def _resolve_val(X_lab, y_local, X_val, y_val, val_fraction, seed):
    if X_val is not None and y_val is not None and len(X_val) > 0:
        X_val = to_dense_float32(X_val)
        y_val = np.asarray(y_val)
        # Map external labels into local indices if they match class set.
        return X_lab, y_local, X_val, y_val, "external_val"
    return internal_val_split(X_lab, y_local, val_fraction=val_fraction, seed=seed)


class _LaplacianSSLBase:
    name = "laplacian_ssl"

    def __init__(self, random_state: int = 0, n_classes: int = 2, **params: Any) -> None:
        self.random_state = int(random_state)
        self.n_classes = int(n_classes)
        self.cfg = dict(params)
        self.classes_: np.ndarray | None = None
        self.model_ = None
        self.training_meta: dict[str, Any] = {}
        self._device = None

    def fit_context(self, ctx) -> "_LaplacianSSLBase":
        views = ctx.views
        cfg = {**self.cfg, **getattr(ctx, "method_config", {})}
        self.cfg = cfg
        X_val = views.X_validation_processed if views.has_validation else None
        y_val = views.y_validation if views.has_validation else None
        return self.fit(
            views.X_labeled_processed,
            views.y_labeled,
            X_unlabeled=views.X_unlabeled_processed,
            X_val=X_val,
            y_val=y_val,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LaplacianLinearSSL(_LaplacianSSLBase):
    """Linear softmax + Laplacian smoothness on train-pool probabilities."""

    name = "laplacian_linear"

    def __init__(self, random_state: int = 0, n_classes: int = 2, **params: Any) -> None:
        cfg = dict(LINEAR_DEFAULTS)
        cfg.update(params)
        super().__init__(random_state=random_state, n_classes=n_classes, **cfg)

    def fit(
        self,
        X_labeled,
        y_labeled,
        X_unlabeled=None,
        X_val=None,
        y_val=None,
    ) -> "LaplacianLinearSSL":
        torch = _require_torch()
        set_global_determinism(self.random_state)
        cfg = self.cfg
        device = torch.device(cfg["device"])
        self._device = device
        rng = np.random.default_rng(self.random_state)

        X_lab, y_local, X_unlab, self.classes_ = _prepare_xy(X_labeled, y_labeled, X_unlabeled)
        # Remap external val labels to local indices when provided.
        if X_val is not None and y_val is not None and len(X_val) > 0:
            y_val_arr = np.asarray(y_val)
            class_to_idx = {c: i for i, c in enumerate(self.classes_)}
            if set(np.unique(y_val_arr)).issubset(set(self.classes_)):
                y_val = np.array([class_to_idx[c] for c in y_val_arr], dtype=np.int64)
            else:
                # Already local ints covering [0, C)
                y_val = y_val_arr.astype(np.int64)

        X_tr, y_tr, X_v, y_v, val_strategy = _resolve_val(
            X_lab, y_local, X_val, y_val, cfg["val_fraction"], self.random_state
        )

        # Graph on full labeled∪unlabeled train (NOT val carve-out features from
        # labeled stay in the pool — use original labeled+unlabeled).
        X_pool = (
            np.vstack([X_lab, X_unlab]).astype(np.float32)
            if X_unlab.shape[0] > 0
            else X_lab.astype(np.float32)
        )
        n_lab = X_lab.shape[0]
        graph = build_sparse_knn_graph(
            X_pool,
            k=int(cfg["graph_k"]),
            mutual=bool(cfg["graph_mutual"]),
            backend=cfg["graph_backend"],
            random_state=self.random_state,
        )
        row_np, col_np, w_np = adjacency_to_edge_tensors(graph.adjacency)

        n_classes = len(self.classes_)
        linear = torch.nn.Linear(X_lab.shape[1], n_classes).to(device)
        torch.nn.init.xavier_uniform_(linear.weight)
        torch.nn.init.zeros_(linear.bias)
        self.model_ = linear
        optim = torch.optim.Adam(
            linear.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )
        ce = torch.nn.CrossEntropyLoss()

        X_tr_t = torch.from_numpy(X_tr).to(device)
        y_tr_t = torch.from_numpy(y_tr).to(device)
        X_pool_t = torch.from_numpy(X_pool).to(device)
        X_v_t = torch.from_numpy(X_v).to(device) if X_v is not None else None
        y_v_t = torch.from_numpy(np.asarray(y_v, dtype=np.int64)).to(device) if y_v is not None else None

        row_t = torch.from_numpy(row_np).to(device)
        col_t = torch.from_numpy(col_np).to(device)
        w_t = torch.from_numpy(w_np).to(device)

        best_loss = math.inf
        best_epoch = -1
        best_state = None
        bad = 0
        loss_hist: list[dict[str, float]] = []
        epochs_trained = 0

        for epoch in range(int(cfg["max_epochs"])):
            linear.train()
            ramp = _ramp_weight(epoch, int(cfg["ramp_epochs"]))
            epoch_ce = 0.0
            epoch_lap = 0.0
            n_batches = 0
            for batch_idx in iterate_minibatches(len(X_tr_t), cfg["batch_size"], rng, shuffle=True):
                xb = X_tr_t[batch_idx]
                yb = y_tr_t[batch_idx]
                optim.zero_grad()
                logits_b = linear(xb)
                loss_ce = ce(logits_b, yb)
                # Full-pool Laplacian (edge-batched inside helper).
                logits_pool = linear(X_pool_t)
                proba_pool = torch.softmax(logits_pool, dim=1)
                loss_lap = laplacian_loss_torch(
                    proba_pool, row_t, col_t, w_t, edge_batch_size=cfg["edge_batch_size"]
                )
                loss = loss_ce + float(cfg["lambda_lap"]) * ramp * loss_lap
                loss.backward()
                optim.step()
                epoch_ce += float(loss_ce.detach().cpu())
                epoch_lap += float(loss_lap.detach().cpu())
                n_batches += 1
            epochs_trained = epoch + 1
            mean_ce = epoch_ce / max(n_batches, 1)
            mean_lap = epoch_lap / max(n_batches, 1)
            loss_hist.append({"ce": mean_ce, "lap": mean_lap, "ramp": ramp})

            linear.eval()
            with torch.no_grad():
                if X_v_t is not None:
                    monitor = float(ce(linear(X_v_t), y_v_t).cpu())
                else:
                    monitor = mean_ce + float(cfg["lambda_lap"]) * mean_lap

            if monitor < best_loss - 1e-5:
                best_loss = monitor
                best_epoch = epoch + 1
                best_state = {k: v.detach().clone() for k, v in linear.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= int(cfg["patience"]):
                    break

        if best_state is not None:
            linear.load_state_dict(best_state)

        with torch.no_grad():
            proba_pool = torch.softmax(linear(X_pool_t), dim=1).cpu().numpy()
        collapse = _collapse_diagnostics(proba_pool)

        self.training_meta = {
            "method": self.name,
            "method_fidelity": "reimplementation",
            "reference_family": "Laplacian / graph SSL",
            "protocol": "inductive",
            "uses_unlabeled_data": bool(X_unlab.shape[0] > 0),
            "n_labeled": int(n_lab),
            "n_unlabeled": int(X_unlab.shape[0]),
            "best_epoch": best_epoch,
            "epochs_trained": epochs_trained,
            "best_val_loss": best_loss if math.isfinite(best_loss) else float("nan"),
            "neural_validation_strategy": val_strategy,
            "lambda_lap": float(cfg["lambda_lap"]),
            "loss_components": loss_hist[-1] if loss_hist else {},
            "loss_history_tail": loss_hist[-5:],
            "collapse_diagnostics": collapse,
            **graph_meta_dict(graph),
        }
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch = _require_torch()
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("LaplacianLinearSSL is not fitted.")
        device = self._device or torch.device(self.cfg["device"])
        X = to_dense_float32(X)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X).to(device))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        proba = normalize_probability_matrix(proba)
        return validate_probability_matrix(proba, n_classes=len(self.classes_))


class LaplacianMLPSSL(_LaplacianSSLBase):
    """Encoder+classifier with prediction and optional embedding Laplacian."""

    name = "laplacian_mlp"

    def __init__(self, random_state: int = 0, n_classes: int = 2, **params: Any) -> None:
        cfg = dict(MLP_DEFAULTS)
        cfg.update(params)
        super().__init__(random_state=random_state, n_classes=n_classes, **cfg)
        self.encoder = None
        self.classifier = None

    def fit(
        self,
        X_labeled,
        y_labeled,
        X_unlabeled=None,
        X_val=None,
        y_val=None,
    ) -> "LaplacianMLPSSL":
        torch = _require_torch()
        set_global_determinism(self.random_state)
        cfg = self.cfg
        device = torch.device(cfg["device"])
        self._device = device
        rng = np.random.default_rng(self.random_state)

        X_lab, y_local, X_unlab, self.classes_ = _prepare_xy(X_labeled, y_labeled, X_unlabeled)
        if X_val is not None and y_val is not None and len(X_val) > 0:
            y_val_arr = np.asarray(y_val)
            class_to_idx = {c: i for i, c in enumerate(self.classes_)}
            if set(np.unique(y_val_arr)).issubset(set(self.classes_)):
                y_val = np.array([class_to_idx[c] for c in y_val_arr], dtype=np.int64)
            else:
                y_val = y_val_arr.astype(np.int64)

        X_tr, y_tr, X_v, y_v, val_strategy = _resolve_val(
            X_lab, y_local, X_val, y_val, cfg["val_fraction"], self.random_state
        )

        X_pool = (
            np.vstack([X_lab, X_unlab]).astype(np.float32)
            if X_unlab.shape[0] > 0
            else X_lab.astype(np.float32)
        )
        n_lab = X_lab.shape[0]
        graph = build_sparse_knn_graph(
            X_pool,
            k=int(cfg["graph_k"]),
            mutual=bool(cfg["graph_mutual"]),
            backend=cfg["graph_backend"],
            random_state=self.random_state,
        )
        row_np, col_np, w_np = adjacency_to_edge_tensors(graph.adjacency)

        n_classes = len(self.classes_)
        encoder = make_mlp(
            torch,
            in_dim=X_lab.shape[1],
            hidden_dim=cfg["hidden_dim"],
            out_dim=cfg["embedding_dim"],
            depth=cfg["depth"],
            dropout=cfg["dropout"],
        ).to(device)
        classifier = torch.nn.Linear(cfg["embedding_dim"], n_classes).to(device)
        self.encoder = encoder
        self.classifier = classifier
        self.model_ = (encoder, classifier)

        params = list(encoder.parameters()) + list(classifier.parameters())
        optim = torch.optim.Adam(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        ce = torch.nn.CrossEntropyLoss()

        X_tr_t = torch.from_numpy(X_tr).to(device)
        y_tr_t = torch.from_numpy(y_tr).to(device)
        X_pool_t = torch.from_numpy(X_pool).to(device)
        X_v_t = torch.from_numpy(X_v).to(device) if X_v is not None else None
        y_v_t = torch.from_numpy(np.asarray(y_v, dtype=np.int64)).to(device) if y_v is not None else None
        row_t = torch.from_numpy(row_np).to(device)
        col_t = torch.from_numpy(col_np).to(device)
        w_t = torch.from_numpy(w_np).to(device)

        best_loss = math.inf
        best_epoch = -1
        best_state = None
        bad = 0
        loss_hist: list[dict[str, float]] = []
        epochs_trained = 0

        for epoch in range(int(cfg["max_epochs"])):
            encoder.train()
            classifier.train()
            ramp = _ramp_weight(epoch, int(cfg["ramp_epochs"]))
            sums = {"ce": 0.0, "pred_lap": 0.0, "emb_lap": 0.0}
            n_batches = 0
            for batch_idx in iterate_minibatches(len(X_tr_t), cfg["batch_size"], rng, shuffle=True):
                xb = X_tr_t[batch_idx]
                yb = y_tr_t[batch_idx]
                optim.zero_grad()
                emb_b = encoder(xb)
                logits_b = classifier(emb_b)
                loss_ce = ce(logits_b, yb)

                emb_pool = encoder(X_pool_t)
                logits_pool = classifier(emb_pool)
                proba_pool = torch.softmax(logits_pool, dim=1)
                loss_pred_lap = laplacian_loss_torch(
                    proba_pool, row_t, col_t, w_t, edge_batch_size=cfg["edge_batch_size"]
                )
                loss = loss_ce + float(cfg["lambda_lap"]) * ramp * loss_pred_lap
                loss_emb_lap = logits_b.new_zeros(())
                if cfg.get("use_embedding_laplacian", True) and float(cfg["lambda_emb_lap"]) > 0:
                    loss_emb_lap = laplacian_loss_torch(
                        emb_pool, row_t, col_t, w_t, edge_batch_size=cfg["edge_batch_size"]
                    )
                    loss = loss + float(cfg["lambda_emb_lap"]) * ramp * loss_emb_lap

                loss.backward()
                optim.step()
                sums["ce"] += float(loss_ce.detach().cpu())
                sums["pred_lap"] += float(loss_pred_lap.detach().cpu())
                sums["emb_lap"] += float(loss_emb_lap.detach().cpu())
                n_batches += 1

            epochs_trained = epoch + 1
            means = {k: v / max(n_batches, 1) for k, v in sums.items()}
            means["ramp"] = ramp
            loss_hist.append(means)

            encoder.eval()
            classifier.eval()
            with torch.no_grad():
                if X_v_t is not None:
                    monitor = float(ce(classifier(encoder(X_v_t)), y_v_t).cpu())
                else:
                    monitor = means["ce"] + float(cfg["lambda_lap"]) * means["pred_lap"]

            if monitor < best_loss - 1e-5:
                best_loss = monitor
                best_epoch = epoch + 1
                best_state = {
                    "encoder": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
                    "classifier": {k: v.detach().clone() for k, v in classifier.state_dict().items()},
                }
                bad = 0
            else:
                bad += 1
                if bad >= int(cfg["patience"]):
                    break

        if best_state is not None:
            encoder.load_state_dict(best_state["encoder"])
            classifier.load_state_dict(best_state["classifier"])

        with torch.no_grad():
            emb_pool = encoder(X_pool_t).cpu().numpy()
            proba_pool = torch.softmax(classifier(encoder(X_pool_t)), dim=1).cpu().numpy()
        collapse = _collapse_diagnostics(proba_pool, emb_pool)

        self.training_meta = {
            "method": self.name,
            "method_fidelity": "reimplementation",
            "reference_family": "Laplacian / graph SSL",
            "protocol": "inductive",
            "uses_unlabeled_data": bool(X_unlab.shape[0] > 0),
            "n_labeled": int(n_lab),
            "n_unlabeled": int(X_unlab.shape[0]),
            "best_epoch": best_epoch,
            "epochs_trained": epochs_trained,
            "best_val_loss": best_loss if math.isfinite(best_loss) else float("nan"),
            "neural_validation_strategy": val_strategy,
            "lambda_lap": float(cfg["lambda_lap"]),
            "lambda_emb_lap": float(cfg["lambda_emb_lap"]),
            "use_embedding_laplacian": bool(cfg.get("use_embedding_laplacian", True)),
            "loss_components": loss_hist[-1] if loss_hist else {},
            "loss_history_tail": loss_hist[-5:],
            "collapse_diagnostics": collapse,
            **graph_meta_dict(graph),
        }
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch = _require_torch()
        if self.encoder is None or self.classifier is None or self.classes_ is None:
            raise RuntimeError("LaplacianMLPSSL is not fitted.")
        device = self._device or torch.device(self.cfg["device"])
        X = to_dense_float32(X)
        self.encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(self.encoder(torch.from_numpy(X).to(device)))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        proba = normalize_probability_matrix(proba)
        return validate_probability_matrix(proba, n_classes=len(self.classes_))


def build_laplacian_model(name: str, random_state: int = 0, n_classes: int = 2, **params: Any):
    if name == "laplacian_ssl":
        name = "laplacian_mlp"
    """Factory for laplacian_linear / laplacian_mlp."""
    key = name.lower().strip()
    if key == "laplacian_linear":
        return LaplacianLinearSSL(random_state=random_state, n_classes=n_classes, **params)
    if key == "laplacian_mlp":
        return LaplacianMLPSSL(random_state=random_state, n_classes=n_classes, **params)
    raise ValueError(f"Unknown Laplacian model '{name}'. Expected one of {sorted(LAPLACIAN_METHODS)}.")
