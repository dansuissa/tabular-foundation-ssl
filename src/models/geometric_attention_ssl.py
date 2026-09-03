"""Combined geometric-attention SSL with ablation flags.

One implementation; registry names are flag presets:

  geometric_attention_supervised
  geometric_attention_laplacian
  geometric_attention_prototype
  geometric_attention_retrieval
  geometric_attention_ssl
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
    feature_shuffle_corrupt,
    internal_val_split,
    iterate_minibatches,
    make_mlp,
    normalize_probability_matrix,
    set_global_determinism,
    to_dense_float32,
    validate_probability_matrix,
)

GEOMETRIC_METHODS = {
    "geometric_attention_supervised",
    "geometric_attention_laplacian",
    "geometric_attention_prototype",
    "geometric_attention_retrieval",
    "geometric_attention_ssl",
}

ABLATION_FLAGS: dict[str, dict[str, bool | str]] = {
    "geometric_attention_supervised": {
        "use_laplacian": False,
        "use_prototype": False,
        "use_retrieval": False,
        "use_pseudo": False,
        "use_consistency": False,
        "use_margin": False,
        "attention_memory": "labeled_only",
    },
    "geometric_attention_laplacian": {
        "use_laplacian": True,
        "use_prototype": False,
        "use_retrieval": False,
        "use_pseudo": False,
        "use_consistency": False,
        "use_margin": False,
        "attention_memory": "labeled_only",
    },
    "geometric_attention_prototype": {
        "use_laplacian": False,
        "use_prototype": True,
        "use_retrieval": False,
        "use_pseudo": False,
        "use_consistency": False,
        "use_margin": True,
        "attention_memory": "labeled_only",
    },
    "geometric_attention_retrieval": {
        "use_laplacian": False,
        "use_prototype": False,
        "use_retrieval": True,
        "use_pseudo": False,
        "use_consistency": False,
        "use_margin": False,
        "attention_memory": "labeled_only",
    },
    "geometric_attention_ssl": {
        "use_laplacian": True,
        "use_prototype": True,
        "use_retrieval": True,
        "use_pseudo": True,
        "use_consistency": True,
        "use_margin": True,
        # The registered complete method must genuinely attend over the
        # unlabeled training pool. Instability is handled by the Phase-B gate,
        # not by silently changing the scientific method.
        "attention_memory": "labeled_plus_unlabeled",
    },
}

DEFAULTS: dict[str, Any] = {
    "hidden_dim": 64,
    "embedding_dim": 32,
    "depth": 2,
    "dropout": 0.1,
    "batch_size": 64,
    "max_epochs": 25,
    "patience": 5,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "device": "auto",
    "val_fraction": 0.2,
    "ramp_epochs": 5,
    "supervised_warmup_epochs": 1,
    "diagnostic_batch_size": 512,
    "graph_k": 10,
    "graph_mutual": True,
    "graph_backend": "auto",
    "edge_batch_size": 50_000,
    "attention_k": 8,
    "attention_labeled_fraction": 0.5,
    "ema_momentum": 0.99,
    "confidence_threshold": 0.9,
    "margin": 1.0,
    "corruption_rate": 0.3,
    "prototype_momentum": 0.9,
    "n_heads": 2,
    "lambda_pl": 0.3,
    "lambda_lap": 0.3,
    "lambda_proto": 0.3,
    "lambda_margin": 0.1,
    "lambda_consistency": 0.1,
    # Component flags (overridden by ablation presets)
    "use_laplacian": True,
    "use_prototype": True,
    "use_retrieval": True,
    "use_pseudo": True,
    "use_consistency": True,
    "use_margin": True,
    # Retrieval memory: labeled_only avoids early collapse on huge unlabeled pools.
    "attention_memory": "labeled_plus_unlabeled",
}


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError(
            "torch",
            "Install torch to run geometric_attention_* methods.",
        ) from exc
    return torch


def _ramp_weight(epoch: int, ramp_epochs: int) -> float:
    if ramp_epochs <= 0:
        return 1.0
    return float(min(1.0, (epoch + 1) / float(ramp_epochs)))


def _prepare(X_labeled, y_labeled, X_unlabeled):
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


def _remap_val(y_val, classes):
    y_val = np.asarray(y_val)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    if set(np.unique(y_val)).issubset(set(classes)):
        return np.array([class_to_idx[c] for c in y_val], dtype=np.int64)
    return y_val.astype(np.int64)


def _topk_neighbors(query_emb, memory_emb, k: int, exclude_idx=None):
    import torch

    q2 = (query_emb ** 2).sum(dim=1, keepdim=True)
    m2 = (memory_emb ** 2).sum(dim=1).unsqueeze(0)
    dist = q2 + m2 - 2.0 * query_emb @ memory_emb.t()
    if exclude_idx is not None:
        b = query_emb.shape[0]
        n_mem = int(memory_emb.shape[0])
        excl = exclude_idx.long()
        # Only mask indices that exist in the current memory bank (labeled-only
        # memory must not use unlabeled offsets; OOB would IndexError / poison).
        valid = (excl >= 0) & (excl < n_mem)
        if valid.any():
            rows = torch.arange(b, device=query_emb.device)[valid]
            dist[rows, excl[valid]] = float("inf")
        n_excl = 1 if valid.any() else 0
    else:
        n_excl = 0
    k_eff = min(k, max(memory_emb.shape[0] - n_excl, 1))
    return dist.topk(k_eff, dim=1, largest=False)


def _balanced_topk_neighbors(
    query_emb,
    memory_emb,
    memory_is_labeled,
    k: int,
    labeled_fraction: float,
    exclude_idx=None,
):
    """Retrieve a fixed labeled/unlabeled mixture from training memory.

    Without stratification, a large unlabeled pool can occupy every retrieved
    slot purely through cardinality. The final memory still contains and
    attends to both labeled and unlabeled rows, but neither type can silently
    eliminate the other.
    """
    import torch

    n_mem = int(memory_emb.shape[0])
    if n_mem == 0:
        raise ValueError("retrieval memory is empty")
    q2 = (query_emb ** 2).sum(dim=1, keepdim=True)
    m2 = (memory_emb ** 2).sum(dim=1).unsqueeze(0)
    dist = q2 + m2 - 2.0 * query_emb @ memory_emb.t()
    if exclude_idx is not None:
        excl = exclude_idx.long()
        valid = (excl >= 0) & (excl < n_mem)
        if valid.any():
            rows = torch.arange(query_emb.shape[0], device=query_emb.device)[valid]
            dist[rows, excl[valid]] = float("inf")

    lab_idx = torch.where(memory_is_labeled)[0]
    unlab_idx = torch.where(~memory_is_labeled)[0]
    if len(unlab_idx) == 0:
        return dist[:, lab_idx].topk(
            min(k, len(lab_idx)), dim=1, largest=False
        )[0], lab_idx[
            dist[:, lab_idx].topk(min(k, len(lab_idx)), dim=1, largest=False)[1]
        ]
    if len(lab_idx) == 0:
        vals, local = dist[:, unlab_idx].topk(
            min(k, len(unlab_idx)), dim=1, largest=False
        )
        return vals, unlab_idx[local]

    k_lab = max(1, min(len(lab_idx), int(round(k * labeled_fraction))))
    k_unlab = max(1, min(len(unlab_idx), k - k_lab))
    # Fill any slots lost to a small stratum from the other stratum.
    remaining = k - k_lab - k_unlab
    if remaining > 0:
        add_lab = min(remaining, len(lab_idx) - k_lab)
        k_lab += add_lab
        remaining -= add_lab
        k_unlab += min(remaining, len(unlab_idx) - k_unlab)

    lab_vals, lab_local = dist[:, lab_idx].topk(k_lab, dim=1, largest=False)
    unlab_vals, unlab_local = dist[:, unlab_idx].topk(
        k_unlab, dim=1, largest=False
    )
    vals = torch.cat([lab_vals, unlab_vals], dim=1)
    idx = torch.cat([lab_idx[lab_local], unlab_idx[unlab_local]], dim=1)
    order = vals.argsort(dim=1)
    return vals.gather(1, order), idx.gather(1, order)


class GeometricAttentionSSL:
    """Unified geometric SSL model; components toggled via flags."""

    name = "geometric_attention_ssl"

    def __init__(
        self,
        random_state: int = 0,
        n_classes: int = 2,
        method_name: str | None = None,
        **params: Any,
    ) -> None:
        self.random_state = int(random_state)
        self.n_classes = int(n_classes)
        cfg = dict(DEFAULTS)
        if method_name is not None and method_name in ABLATION_FLAGS:
            cfg.update(ABLATION_FLAGS[method_name])
            self.name = method_name
        cfg.update(params)
        self.cfg = cfg
        self.classes_: np.ndarray | None = None
        self.encoder = None
        self.classifier = None
        self.attn = None
        self.class_emb = None
        self.type_emb = None
        self.proto_tokens = None
        self.gate = None
        self._teacher = None
        self._memory_X = None
        self._memory_is_labeled = None
        self._memory_y = None
        self.training_meta: dict[str, Any] = {}
        self._device = None

    def fit_context(self, ctx) -> "GeometricAttentionSSL":
        views = ctx.views
        self.cfg = {**self.cfg, **getattr(ctx, "method_config", {})}
        X_val = views.X_validation_processed if views.has_validation else None
        y_val = views.y_validation if views.has_validation else None
        return self.fit(
            views.X_labeled_processed,
            views.y_labeled,
            X_unlabeled=views.X_unlabeled_processed,
            X_val=X_val,
            y_val=y_val,
        )

    def _build(self, torch, input_dim, n_classes, device):
        cfg = self.cfg
        d = int(cfg["embedding_dim"])
        encoder = make_mlp(
            torch,
            in_dim=input_dim,
            hidden_dim=cfg["hidden_dim"],
            out_dim=d,
            depth=cfg["depth"],
            dropout=cfg["dropout"],
        ).to(device)
        classifier = torch.nn.Linear(d, n_classes).to(device)
        attn = class_emb = type_emb = proto_tokens = gate = None
        if cfg["use_retrieval"]:
            n_heads = int(cfg["n_heads"])
            while d % n_heads != 0 and n_heads > 1:
                n_heads -= 1
            attn = torch.nn.MultiheadAttention(
                d, n_heads, batch_first=True, dropout=float(cfg["dropout"])
            ).to(device)
            class_emb = torch.nn.Embedding(n_classes + 1, d).to(device)
            type_emb = torch.nn.Embedding(2, d).to(device)
            proto_tokens = torch.nn.Parameter(torch.randn(n_classes, d, device=device) * 0.02)
            gate = torch.nn.Sequential(
                torch.nn.Linear(2 * d, d),
                torch.nn.ReLU(),
                torch.nn.Linear(d, d),
                torch.nn.Sigmoid(),
            ).to(device)
        return encoder, classifier, attn, class_emb, type_emb, proto_tokens, gate

    def fit(
        self,
        X_labeled,
        y_labeled,
        X_unlabeled=None,
        X_val=None,
        y_val=None,
    ) -> "GeometricAttentionSSL":
        torch = _require_torch()
        set_global_determinism(self.random_state)
        cfg = self.cfg
        # Prefer CUDA when available unless an explicit device was requested.
        req = str(cfg.get("device", "auto"))
        if req in {"auto", "cuda", "gpu"} and torch.cuda.is_available():
            device = torch.device("cuda")
        elif req.startswith("cuda") and torch.cuda.is_available():
            device = torch.device(req)
        else:
            device = torch.device("cpu")
        self._device = device
        cfg = {**cfg, "device": str(device)}
        self.cfg = cfg
        rng = np.random.default_rng(self.random_state)

        X_lab, y_local, X_unlab, self.classes_ = _prepare(X_labeled, y_labeled, X_unlabeled)
        n_classes = len(self.classes_)

        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_val = to_dense_float32(X_val)
            y_val = _remap_val(y_val, self.classes_)
            X_tr, y_tr, X_v, y_v, val_strategy = X_lab, y_local, X_val, y_val, "external_val"
        else:
            X_tr, y_tr, X_v, y_v, val_strategy = internal_val_split(
                X_lab, y_local, val_fraction=cfg["val_fraction"], seed=self.random_state
            )

        X_pool = (
            np.vstack([X_lab, X_unlab]).astype(np.float32)
            if X_unlab.shape[0] > 0
            else X_lab.astype(np.float32)
        )
        graph = None
        row_t = col_t = w_t = None
        if cfg["use_laplacian"]:
            graph = build_sparse_knn_graph(
                X_pool,
                k=int(cfg["graph_k"]),
                mutual=bool(cfg["graph_mutual"]),
                backend=cfg["graph_backend"],
                random_state=self.random_state,
            )
            row_np, col_np, w_np = adjacency_to_edge_tensors(graph.adjacency)
            row_t = torch.from_numpy(row_np).to(device)
            col_t = torch.from_numpy(col_np).to(device)
            w_t = torch.from_numpy(w_np).to(device)

        # Memory bank (train only; never val/test). Retrieval ablations may use labeled-only.
        mem_mode = str(cfg.get("attention_memory", "labeled_plus_unlabeled"))
        if mem_mode == "labeled_only" or X_unlab.shape[0] == 0:
            X_mem = X_lab
            is_lab = np.ones(len(X_lab), dtype=bool)
            y_mem = y_local.copy()
        else:
            X_mem = X_pool
            is_lab = np.concatenate(
                [np.ones(len(X_lab), dtype=bool), np.zeros(len(X_unlab), dtype=bool)]
            )
            y_mem = np.concatenate([y_local, np.full(len(X_unlab), -1, dtype=np.int64)])
        self._memory_X = X_mem
        self._memory_is_labeled = is_lab
        self._memory_y = y_mem

        (
            encoder,
            classifier,
            attn,
            class_emb,
            type_emb,
            proto_tokens,
            gate,
        ) = self._build(torch, X_lab.shape[1], n_classes, device)
        self.encoder = encoder
        self.classifier = classifier
        self.attn = attn
        self.class_emb = class_emb
        self.type_emb = type_emb
        self.proto_tokens = proto_tokens
        self.gate = gate

        import copy

        teacher = copy.deepcopy(encoder)
        for p in teacher.parameters():
            p.requires_grad_(False)
        self._teacher = teacher

        params = list(encoder.parameters()) + list(classifier.parameters())
        if cfg["use_retrieval"]:
            params += (
                list(attn.parameters())
                + list(class_emb.parameters())
                + list(type_emb.parameters())
                + [proto_tokens]
                + list(gate.parameters())
            )
        optim = torch.optim.Adam(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        ce = torch.nn.CrossEntropyLoss()

        X_tr_t = torch.from_numpy(X_tr).to(device)
        y_tr_t = torch.from_numpy(y_tr).to(device)
        X_pool_t = torch.from_numpy(X_pool).to(device)
        X_lab_t = torch.from_numpy(X_lab).to(device)
        y_lab_t = torch.from_numpy(y_local).to(device)
        X_mem_t = torch.from_numpy(X_mem).to(device)
        is_lab_t = torch.from_numpy(is_lab).to(device)
        y_mem_t = torch.from_numpy(y_mem).to(device)
        X_unlab_t = torch.from_numpy(X_unlab).to(device) if X_unlab.shape[0] > 0 else None
        X_v_t = torch.from_numpy(X_v).to(device) if X_v is not None else None
        y_v_t = torch.from_numpy(np.asarray(y_v, dtype=np.int64)).to(device) if y_v is not None else None

        prototypes = torch.zeros(n_classes, cfg["embedding_dim"], device=device)
        with torch.no_grad():
            e0 = encoder(X_lab_t)
            for c in range(n_classes):
                m = y_lab_t == c
                if m.any():
                    prototypes[c] = e0[m].mean(0)

        from sklearn.neighbors import NearestNeighbors

        nn_lab = NearestNeighbors(n_neighbors=1).fit(X_lab)
        tr_to_lab = nn_lab.kneighbors(X_tr, return_distance=False).ravel()

        best_loss = math.inf
        best_epoch = -1
        best_state = None
        bad = 0
        loss_hist: list[dict[str, float]] = []
        attn_stats_last = {}
        retrieval_state = {
            "strength": 0.0,
            "confidence_threshold": 0.99,
        }
        epochs_trained = 0
        k_att = int(cfg["attention_k"])
        mom_ema = float(cfg["ema_momentum"])
        mom_proto = float(cfg["prototype_momentum"])

        def encode_query(xb, excl=None):
            q = encoder(xb)
            stats = {
                "labeled_attention_mass": 0.0,
                "unlabeled_attention_mass": 0.0,
                "attention_entropy": 0.0,
            }
            if not cfg["use_retrieval"]:
                return classifier(q), q, stats
            # Teacher memory embeddings are detached; class/type embeddings MUST
            # remain outside no_grad so they can learn (otherwise memory tokens
            # are frozen random noise and retrieval collapses to chance).
            with torch.no_grad():
                mem_emb = teacher(X_mem_t)
                unknown = class_emb.num_embeddings - 1
                type_ids = (~is_lab_t).long()
                class_ids = torch.full(
                    (X_mem_t.shape[0],), unknown, device=device, dtype=torch.long
                )
                class_ids[is_lab_t] = y_mem_t[is_lab_t].clamp(0, unknown - 1)
                if (~is_lab_t).any():
                    tp = torch.softmax(classifier(mem_emb), dim=1)
                    conf, yhat = tp.max(dim=1)
                    rel = (~is_lab_t) & (
                        conf >= float(retrieval_state["confidence_threshold"])
                    )
                    class_ids[rel] = yhat[rel]
            mem_tokens = (
                mem_emb.detach() + type_emb(type_ids) + class_emb(class_ids)
            )
            if mem_mode == "labeled_plus_unlabeled":
                dist_vals, nn_idx = _balanced_topk_neighbors(
                    q.detach(),
                    mem_emb.detach(),
                    is_lab_t,
                    k=k_att,
                    labeled_fraction=float(cfg["attention_labeled_fraction"]),
                    exclude_idx=excl,
                )
            else:
                dist_vals, nn_idx = _topk_neighbors(
                    q.detach(), mem_emb.detach(), k=k_att, exclude_idx=excl
                )
            neigh = mem_tokens[nn_idx]
            q_in = q.unsqueeze(1)
            attn_out, attn_w = attn(
                q_in, neigh, neigh, need_weights=True, average_attn_weights=True
            )
            attn_out = attn_out.squeeze(1)
            attn_w = attn_w.squeeze(1)
            neigh_lab = is_lab_t[nn_idx].float()
            stats = {
                "labeled_attention_mass": float(
                    (attn_w * neigh_lab).sum(1).mean().detach().cpu()
                ),
                "unlabeled_attention_mass": float(
                    (attn_w * (1.0 - neigh_lab)).sum(1).mean().detach().cpu()
                ),
                "attention_entropy": float(
                    (
                        -(attn_w.clamp_min(1e-12) * attn_w.clamp_min(1e-12).log())
                        .sum(1)
                        .mean()
                    )
                    .detach()
                    .cpu()
                ),
            }
            combined = attn_out
            # Prototype-token bank is part of the prototype ablation, not retrieval-only.
            if cfg.get("use_prototype", False):
                proto = proto_tokens.unsqueeze(0).expand(q.shape[0], -1, -1)
                proto_out, _ = attn(q_in, proto, proto, need_weights=False)
                combined = attn_out + proto_out.squeeze(1)
            g = gate(torch.cat([q, combined], dim=1))
            strength = float(retrieval_state["strength"])
            z = q + strength * g * (combined - q)
            return classifier(z), z, stats

        for epoch in range(int(cfg["max_epochs"])):
            encoder.train()
            classifier.train()
            if cfg["use_retrieval"]:
                attn.train()
                gate.train()
            ramp = _ramp_weight(epoch, int(cfg["ramp_epochs"]))
            warmup = int(cfg["supervised_warmup_epochs"])
            retrieval_ramp = _ramp_weight(
                max(epoch - warmup, 0), int(cfg["ramp_epochs"])
            )
            if epoch < warmup:
                retrieval_ramp = 0.0
            retrieval_state["strength"] = retrieval_ramp
            retrieval_state["confidence_threshold"] = float(
                0.99
                - retrieval_ramp
                * (0.99 - float(cfg["confidence_threshold"]))
            )
            sums = {
                "sup": 0.0,
                "pl": 0.0,
                "lap": 0.0,
                "proto": 0.0,
                "margin": 0.0,
                "consistency": 0.0,
                "gradient_norm": 0.0,
            }
            n_batches = 0
            mass_acc = {
                "labeled_attention_mass": 0.0,
                "unlabeled_attention_mass": 0.0,
                "attention_entropy": 0.0,
            }

            for batch_pos in iterate_minibatches(len(X_tr_t), cfg["batch_size"], rng, shuffle=True):
                xb = X_tr_t[batch_pos]
                yb = y_tr_t[batch_pos]
                excl = torch.from_numpy(tr_to_lab[batch_pos]).to(device)
                optim.zero_grad()

                logits, emb, stats = encode_query(xb, excl)
                loss_sup = ce(logits, yb)
                loss = loss_sup
                loss_pl = logits.new_zeros(())
                loss_lap = logits.new_zeros(())
                loss_proto = logits.new_zeros(())
                loss_margin = logits.new_zeros(())
                loss_cons = logits.new_zeros(())

                # Pseudo + prototype attraction on high-conf unlabeled
                if X_unlab_t is not None and (cfg["use_pseudo"] or cfg["use_prototype"]):
                    u_n = min(len(xb), X_unlab_t.shape[0])
                    u_idx = rng.integers(0, X_unlab_t.shape[0], size=u_n)
                    xu = X_unlab_t[torch.from_numpy(u_idx).to(device)]
                    # Self-exclusion only when unlabeled rows live in the memory bank.
                    if mem_mode != "labeled_only" and X_unlab.shape[0] > 0:
                        excl_u = torch.from_numpy(u_idx + len(X_lab)).to(device)
                    else:
                        excl_u = None
                    with torch.no_grad():
                        logits_u0, _, _ = encode_query(xu, excl_u)
                        pu = torch.softmax(logits_u0, dim=1)
                        conf, yhat = pu.max(dim=1)
                        reliable = conf >= float(cfg["confidence_threshold"])
                    if reliable.any():
                        excl_rel = excl_u[reliable] if excl_u is not None else None
                        logits_u, emb_u, _ = encode_query(xu[reliable], excl_rel)
                        if cfg["use_pseudo"] and float(cfg["lambda_pl"]) > 0:
                            loss_pl = ce(logits_u, yhat[reliable])
                            loss = loss + float(cfg["lambda_pl"]) * ramp * loss_pl
                        if cfg["use_prototype"] and float(cfg["lambda_proto"]) > 0:
                            target = prototypes[yhat[reliable]].detach()
                            # Mean per embedding dimension keeps this component
                            # commensurate with supervised cross-entropy.
                            dist = ((emb_u - target) ** 2).mean(dim=1)
                            loss_proto = (conf[reliable] * dist).mean()
                            loss = loss + float(cfg["lambda_proto"]) * ramp * loss_proto
                            with torch.no_grad():
                                for c in range(n_classes):
                                    m = yhat[reliable] == c
                                    if m.any():
                                        prototypes[c] = (
                                            mom_proto * prototypes[c]
                                            + (1.0 - mom_proto) * emb_u[m].detach().mean(0)
                                        )

                # Laplacian on train-pool predictions (encoder path; not full retrieval)
                if cfg["use_laplacian"] and row_t is not None and float(cfg["lambda_lap"]) > 0:
                    emb_pool = encoder(X_pool_t)
                    proba_pool = torch.softmax(classifier(emb_pool), dim=1)
                    loss_lap = laplacian_loss_torch(
                        proba_pool, row_t, col_t, w_t, edge_batch_size=cfg["edge_batch_size"]
                    )
                    loss = loss + float(cfg["lambda_lap"]) * ramp * loss_lap

                # Prototype separation
                if cfg["use_margin"] and cfg["use_prototype"] and float(cfg["lambda_margin"]) > 0:
                    seps = []
                    for i in range(n_classes):
                        for j in range(i + 1, n_classes):
                            d = torch.norm(prototypes[i] - prototypes[j], p=2)
                            seps.append(torch.relu(float(cfg["margin"]) - d) ** 2)
                    if seps:
                        loss_margin = torch.stack(seps).mean()
                        loss = loss + float(cfg["lambda_margin"]) * ramp * loss_margin

                # Consistency
                if cfg["use_consistency"] and float(cfg["lambda_consistency"]) > 0:
                    mask = (torch.rand_like(xb) < float(cfg["corruption_rate"])).float()
                    x_cor = feature_shuffle_corrupt(torch, xb, mask)
                    p1 = torch.softmax(encode_query(xb, excl)[0], dim=1)
                    p2 = torch.softmax(encode_query(x_cor, excl)[0], dim=1)
                    loss_cons = ((p1 - p2) ** 2).mean()
                    loss = loss + float(cfg["lambda_consistency"]) * ramp * loss_cons

                # Labeled prototype EMA refresh
                if cfg["use_prototype"]:
                    with torch.no_grad():
                        for c in range(n_classes):
                            m = yb == c
                            if m.any():
                                prototypes[c] = mom_proto * prototypes[c] + (1.0 - mom_proto) * emb[m].detach().mean(0)

                loss.backward()
                grad_sq = 0.0
                for parameter in params:
                    if parameter.grad is not None:
                        grad_sq += float(parameter.grad.detach().norm(2).cpu()) ** 2
                grad_norm = math.sqrt(grad_sq)
                optim.step()

                if cfg["use_retrieval"]:
                    with torch.no_grad():
                        for p_t, p_s in zip(teacher.parameters(), encoder.parameters()):
                            p_t.data.mul_(mom_ema).add_(p_s.data, alpha=1.0 - mom_ema)

                sums["sup"] += float(loss_sup.detach().cpu())
                sums["pl"] += float(loss_pl.detach().cpu())
                sums["lap"] += float(loss_lap.detach().cpu())
                sums["proto"] += float(loss_proto.detach().cpu())
                sums["margin"] += float(loss_margin.detach().cpu())
                sums["consistency"] += float(loss_cons.detach().cpu())
                sums["gradient_norm"] += grad_norm
                mass_acc["labeled_attention_mass"] += stats["labeled_attention_mass"]
                mass_acc["unlabeled_attention_mass"] += stats["unlabeled_attention_mass"]
                mass_acc["attention_entropy"] += stats["attention_entropy"]
                n_batches += 1

            epochs_trained = epoch + 1
            means = {k: v / max(n_batches, 1) for k, v in sums.items()}
            means["ramp"] = ramp
            means["retrieval_strength"] = retrieval_ramp
            means["weighted_pl"] = float(cfg["lambda_pl"]) * ramp * means["pl"]
            means["weighted_lap"] = float(cfg["lambda_lap"]) * ramp * means["lap"]
            means["weighted_proto"] = float(cfg["lambda_proto"]) * ramp * means["proto"]
            means["weighted_margin"] = float(cfg["lambda_margin"]) * ramp * means["margin"]
            means["weighted_consistency"] = (
                float(cfg["lambda_consistency"]) * ramp * means["consistency"]
            )
            loss_hist.append(means)
            attn_stats_last = {k: v / max(n_batches, 1) for k, v in mass_acc.items()}

            encoder.eval()
            classifier.eval()
            with torch.no_grad():
                if X_v_t is not None:
                    logits_v, _, _ = encode_query(X_v_t, None)
                    monitor = float(ce(logits_v, y_v_t).cpu())
                else:
                    monitor = means["sup"]

            eligible = (not cfg["use_retrieval"]) or retrieval_ramp > 0.0
            if eligible and monitor < best_loss - 1e-5:
                best_loss = monitor
                best_epoch = epoch + 1
                best_state = {
                    "encoder": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
                    "classifier": {k: v.detach().clone() for k, v in classifier.state_dict().items()},
                    "prototypes": prototypes.detach().clone(),
                    "teacher": {k: v.detach().clone() for k, v in teacher.state_dict().items()},
                    "retrieval_strength": retrieval_ramp,
                }
                if cfg["use_retrieval"]:
                    best_state["attn"] = {k: v.detach().clone() for k, v in attn.state_dict().items()}
                    best_state["class_emb"] = {
                        k: v.detach().clone() for k, v in class_emb.state_dict().items()
                    }
                    best_state["type_emb"] = {
                        k: v.detach().clone() for k, v in type_emb.state_dict().items()
                    }
                    best_state["gate"] = {k: v.detach().clone() for k, v in gate.state_dict().items()}
                    best_state["proto_tokens"] = proto_tokens.detach().clone()
                bad = 0
            else:
                bad += 1
                if bad >= int(cfg["patience"]):
                    break

        if best_state is not None:
            encoder.load_state_dict(best_state["encoder"])
            classifier.load_state_dict(best_state["classifier"])
            prototypes = best_state["prototypes"]
            teacher.load_state_dict(best_state["teacher"])
            retrieval_state["strength"] = float(best_state["retrieval_strength"])
            if cfg["use_retrieval"]:
                attn.load_state_dict(best_state["attn"])
                class_emb.load_state_dict(best_state["class_emb"])
                type_emb.load_state_dict(best_state["type_emb"])
                gate.load_state_dict(best_state["gate"])
                with torch.no_grad():
                    proto_tokens.copy_(best_state["proto_tokens"])

        # Collapse diagnostics. Batch the full train pool to avoid invalid CUDA
        # launch configurations on large datasets such as jannis.
        with torch.no_grad():
            proba_parts = []
            emb_parts = []
            diag_bs = int(cfg["diagnostic_batch_size"])
            for start in range(0, len(X_pool_t), diag_bs):
                logits_p, emb_p, _ = encode_query(
                    X_pool_t[start : start + diag_bs], None
                )
                proba_parts.append(torch.softmax(logits_p, dim=1).cpu().numpy())
                emb_parts.append(emb_p.cpu().numpy())
            proba_p = np.vstack(proba_parts)
            emb_np = np.vstack(emb_parts)
            frac = np.bincount(proba_p.argmax(1), minlength=n_classes).astype(np.float64)
            frac = frac / max(frac.sum(), 1.0)
            var = emb_np.var(axis=0)
            collapse = {
                "pred_class_fraction": frac.tolist(),
                "max_pred_class_fraction": float(frac.max()),
                "constant_prediction_suspect": bool(frac.max() >= 0.98),
                "embedding_variance_mean": float(var.mean()),
                "representation_collapse_suspect": bool(var.mean() < 1e-6),
                "effective_embedding_rank": int(
                    np.sum(
                        np.linalg.svd(
                            emb_np - emb_np.mean(0, keepdims=True),
                            compute_uv=False,
                        )
                        > 1e-6
                    )
                ),
            }

        meta = {
            "method": self.name,
            "method_fidelity": "reimplementation",
            "reference_family": "Geometric attention SSL",
            "protocol": "inductive",
            "uses_unlabeled_data": bool(X_unlab.shape[0] > 0),
            "component_flags": {
                "use_laplacian": cfg["use_laplacian"],
                "use_prototype": cfg["use_prototype"],
                "use_retrieval": cfg["use_retrieval"],
                "use_pseudo": cfg["use_pseudo"],
                "use_consistency": cfg["use_consistency"],
                "use_margin": cfg["use_margin"],
            },
            "n_labeled": int(X_lab.shape[0]),
            "n_unlabeled": int(X_unlab.shape[0]),
            "best_epoch": best_epoch,
            "epochs_trained": epochs_trained,
            "best_val_loss": best_loss if math.isfinite(best_loss) else float("nan"),
            "neural_validation_strategy": val_strategy,
            "loss_components": loss_hist[-1] if loss_hist else {},
            "loss_history_tail": loss_hist[-5:],
            "attention_k": k_att if cfg["use_retrieval"] else 0,
            "attention_memory": mem_mode,
            "attention_labeled_fraction": float(cfg["attention_labeled_fraction"]),
            "selected_retrieval_strength": float(retrieval_state["strength"]),
            "labeled_attention_mass": attn_stats_last.get("labeled_attention_mass", 0.0),
            "unlabeled_attention_mass": attn_stats_last.get("unlabeled_attention_mass", 0.0),
            "attention_entropy": attn_stats_last.get("attention_entropy", 0.0),
            "collapse_diagnostics": collapse,
            "no_val_test_in_memory": True,
        }
        if graph is not None:
            meta.update(graph_meta_dict(graph))
        self.training_meta = meta

        # Stash encode_query for predict via bound method recreation
        self._encode_query_fn = encode_query
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch = _require_torch()
        if self.encoder is None or self.classes_ is None:
            raise RuntimeError("GeometricAttentionSSL is not fitted.")
        device = self._device or torch.device(self.cfg["device"])
        X = to_dense_float32(X)
        X_t = torch.from_numpy(X).to(device)
        self.encoder.eval()
        self.classifier.eval()
        proba_parts = []
        bs = int(self.cfg["batch_size"])
        with torch.no_grad():
            for start in range(0, len(X_t), bs):
                xb = X_t[start : start + bs]
                logits, _, _ = self._encode_query_fn(xb, None)
                proba_parts.append(torch.softmax(logits, dim=1).cpu().numpy())
        proba = np.vstack(proba_parts)
        proba = normalize_probability_matrix(proba)
        return validate_probability_matrix(proba, n_classes=len(self.classes_))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_geometric_attention_model(
    name: str,
    random_state: int = 0,
    n_classes: int = 2,
    **params: Any,
) -> GeometricAttentionSSL:
    key = name.lower().strip()
    if key not in GEOMETRIC_METHODS:
        raise ValueError(
            f"Unknown geometric model '{name}'. Expected one of {sorted(GEOMETRIC_METHODS)}."
        )
    return GeometricAttentionSSL(
        random_state=random_state,
        n_classes=n_classes,
        method_name=key,
        **params,
    )
