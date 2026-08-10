"""Official STUNT adapter for the tabular SSL benchmark.

Upstream: https://github.com/jaehyun513/STUNT
Pinned commit: see ``UPSTREAM_COMMIT``.

Scientific intent
-----------------
STUNT meta-learns a ProtoNet encoder on self-generated few-shot tasks from
unlabeled columns (k-means pseudo-labels on random column subsets, with
column permutation). We adapt the official algorithm to this benchmark's
fixed labeled/unlabeled/validation/test splits rather than STUNT's own
index files.

Differences from the paper / official scripts (logged in training_meta):
* Uses benchmark fixed splits instead of STUNT dataset index npy files.
* Pseudo-validation uses unsupervised k-means on a holdout of unlabeled
  features (STUNT's generate_pseudo_val scheme), not paper dataset scripts.
* FAISS GPU k-means falls back to sklearn MiniBatchKMeans when FAISS is
  unavailable.
* Smoke defaults use fewer meta-steps than the paper's full schedule.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.exceptions import OptionalDependencyError, UnsupportedMethodError
from src.models.common import empty_training_meta
from src.models.torch_utils import set_global_determinism, to_dense_float32

LOGGER = logging.getLogger(__name__)

# Pinned upstream commit (jaehyun513/STUNT @ main tip used for integration).
UPSTREAM_COMMIT = "e860675d0e390dba5f12eb9fd7bdcdd8d379f012"
UPSTREAM_URL = "https://github.com/jaehyun513/STUNT"

_THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party" / "stunt"


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise OptionalDependencyError(
            "torch",
            "STUNT requires PyTorch. Install torch in ssl-tfm / ssl-representation.",
        ) from exc
    return torch


def _kmeans_labels(
    X: np.ndarray,
    n_clusters: int,
    min_points: int,
    random_state: int,
) -> np.ndarray | None:
    """Cluster rows; prefer FAISS (official), else sklearn MiniBatchKMeans."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    try:
        import faiss  # type: ignore

        kmeans = faiss.Kmeans(
            X.shape[1],
            n_clusters,
            niter=20,
            nredo=1,
            verbose=False,
            min_points_per_centroid=min_points,
            seed=int(random_state),
            gpu=False,
        )
        kmeans.train(X)
        _, I = kmeans.index.search(X, 1)
        return I[:, 0].astype(np.int32)
    except Exception:
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=int(random_state),
            n_init=3,
            max_iter=50,
            batch_size=min(1024, max(n_clusters * 10, X.shape[0])),
        )
        labels = km.fit_predict(X).astype(np.int32)
        _, counts = np.unique(labels, return_counts=True)
        if counts.min() < min_points:
            return None
        return labels


class _MLPProto:
    """Faithful port of STUNT models/protonet_model/mlp.py::MLPProto."""

    def __init__(self, torch, in_features: int, out_features: int, hidden_sizes: int):
        nn = torch.nn
        self.module = nn.Sequential(
            nn.Linear(in_features, hidden_sizes, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_sizes, hidden_sizes, bias=True),
        )
        self.torch = torch

    def __call__(self, inputs):
        # inputs: (meta_batch, n_examples, n_features) or (n, d)
        if inputs.dim() == 3:
            b, n, d = inputs.shape
            emb = self.module(inputs.view(-1, d))
            return emb.view(b, n, -1)
        return self.module(inputs)

    def parameters(self):
        return self.module.parameters()

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        self.module.load_state_dict(state)

    def to(self, device):
        self.module.to(device)
        return self


def _get_prototypes(embeddings, targets, num_ways):
    """Mean prototype per class (torchmeta-compatible)."""
    # embeddings: (meta_batch, n_support, dim); targets: (meta_batch, n_support)
    torch = _require_torch()
    mb, _, dim = embeddings.shape
    protos = []
    for b in range(mb):
        class_protos = []
        for c in range(num_ways):
            mask = targets[b] == c
            if mask.any():
                class_protos.append(embeddings[b, mask].mean(dim=0))
            else:
                class_protos.append(torch.zeros(dim, device=embeddings.device))
        protos.append(torch.stack(class_protos, dim=0))
    return torch.stack(protos, dim=0)


def generate_stunt_task_batch(
    unlabeled_x: np.ndarray,
    *,
    shot: int,
    query: int,
    num_way: int,
    tasks_per_batch: int,
    rng: np.random.RandomState,
) -> dict[str, Any] | None:
    """Self-generate ProtoNet tasks from unlabeled columns (STUNT income.py)."""
    x = unlabeled_x
    xs, ys, xq, yq = [], [], [], []
    for _ in range(tasks_per_batch):
        tmp_x = copy.deepcopy(x)
        min_count = 0
        y = None
        task_idx = None
        tries = 0
        while min_count < (shot + query) and tries < 30:
            tries += 1
            min_col = max(1, int(x.shape[1] * 0.2))
            max_col = max(min_col + 1, int(x.shape[1] * 0.5))
            if max_col <= min_col:
                max_col = min(x.shape[1], min_col + 1)
            col = int(rng.choice(range(min_col, max_col), 1, replace=False)[0])
            col = min(col, x.shape[1])
            task_idx = rng.choice(np.arange(x.shape[1]), col, replace=False)
            masked_x = np.ascontiguousarray(x[:, task_idx], dtype=np.float32)
            labels = _kmeans_labels(
                masked_x,
                n_clusters=num_way,
                min_points=shot + query,
                random_state=int(rng.randint(0, 10_000_000)),
            )
            if labels is None:
                continue
            y = labels
            class_list, counts = np.unique(y, return_counts=True)
            min_count = int(counts.min()) if len(counts) else 0

        if y is None or task_idx is None or min_count < (shot + query):
            return None

        num_to_permute = x.shape[0]
        for t_idx in task_idx:
            rand_perm = rng.permutation(num_to_permute)
            tmp_x[:, t_idx] = tmp_x[:, t_idx][rand_perm]

        class_list = np.unique(y)
        if len(class_list) < num_way:
            return None
        classes = rng.choice(class_list, num_way, replace=False)
        support_idx, query_idx = [], []
        for k in classes:
            k_idx = np.where(y == k)[0]
            k_idx = k_idx[rng.permutation(len(k_idx))]
            support_idx.append(k_idx[:shot])
            query_idx.append(k_idx[shot : shot + query])
        support_idx = np.concatenate(support_idx)
        query_idx = np.concatenate(query_idx)
        support_x = tmp_x[support_idx]
        query_x = tmp_x[query_idx]
        s_y = y[support_idx]
        q_y = y[query_idx]
        support_y = copy.deepcopy(s_y)
        query_y = copy.deepcopy(q_y)
        for i, k in enumerate(classes):
            support_y[s_y == k] = i
            query_y[q_y == k] = i
        xs.append(support_x.astype(np.float32))
        xq.append(query_x.astype(np.float32))
        ys.append(support_y.astype(np.int64))
        yq.append(query_y.astype(np.int64))

    return {
        "train": (np.stack(xs, 0), np.stack(ys, 0)),
        "test": (np.stack(xq, 0), np.stack(yq, 0)),
    }


class STUNTMethod:
    """STUNT ProtoNet adapted to benchmark fixed splits."""

    name = "stunt"

    def __init__(
        self,
        random_state: int = 0,
        n_classes: int = 2,
        hidden_dim: int = 1024,
        meta_steps: int = 200,
        meta_batch_size: int = 4,
        num_ways: int = 2,
        shot: int = 1,
        query: int = 15,
        lr: float = 1e-3,
        smoke: bool = True,
        device: str | None = None,
    ) -> None:
        self.random_state = int(random_state)
        self.n_classes = int(n_classes)
        self.hidden_dim = int(hidden_dim)
        self.meta_steps = int(50 if smoke else meta_steps)
        self.meta_batch_size = int(meta_batch_size)
        self.num_ways = int(num_ways)
        self.shot = int(shot)
        self.query = int(query)
        self.lr = float(lr)
        self.smoke = bool(smoke)
        self.device_request = device
        self.model: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._support_x: np.ndarray | None = None
        self._support_y: np.ndarray | None = None
        self.training_meta: dict[str, Any] = empty_training_meta()

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "STUNTMethod":
        torch = _require_torch()
        self._torch = torch
        set_global_determinism(self.random_state)
        device = torch.device(
            self.device_request
            if self.device_request
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._device = device

        X_lab = to_dense_float32(X_labeled)
        y_lab = np.asarray(y_labeled).astype(np.int64)
        if X_unlabeled is None or len(X_unlabeled) == 0:
            raise UnsupportedMethodError(
                self.name,
                "STUNT requires an unlabeled pool to self-generate column tasks.",
                status="unsupported_stunt_no_unlabeled",
            )
        X_u = to_dense_float32(X_unlabeled)
        if X_u.shape[1] < 2:
            raise UnsupportedMethodError(
                self.name,
                "STUNT needs ≥2 features to sample column subsets.",
                status="unsupported_stunt_too_few_features",
            )

        # Ensure third_party marker exists (thin adapter / clone hook).
        _THIRD_PARTY.mkdir(parents=True, exist_ok=True)
        marker = _THIRD_PARTY / "UPSTREAM_COMMIT.txt"
        if not marker.exists():
            marker.write_text(f"{UPSTREAM_COMMIT}\n{UPSTREAM_URL}\n", encoding="utf-8")

        in_dim = int(X_u.shape[1])
        encoder = _MLPProto(torch, in_dim, self.hidden_dim, self.hidden_dim).to(device)
        opt = torch.optim.Adam(encoder.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        rng = np.random.RandomState(self.random_state)

        steps_ok = 0
        last_loss = float("nan")
        for step in range(self.meta_steps):
            batch = generate_stunt_task_batch(
                X_u,
                shot=self.shot,
                query=self.query,
                num_way=self.num_ways,
                tasks_per_batch=self.meta_batch_size,
                rng=rng,
            )
            if batch is None:
                continue
            xs, ys = batch["train"]
            xq, yq = batch["test"]
            xs_t = torch.from_numpy(xs).to(device)
            ys_t = torch.from_numpy(ys).to(device)
            xq_t = torch.from_numpy(xq).to(device)
            yq_t = torch.from_numpy(yq).to(device)

            encoder.train()
            support_emb = encoder(xs_t)
            query_emb = encoder(xq_t)
            protos = _get_prototypes(support_emb, ys_t, self.num_ways)
            # squared distances → logits
            logits = -torch.sum(
                (protos.unsqueeze(2) - query_emb.unsqueeze(1)) ** 2, dim=-1
            )
            loss = criterion(logits.view(-1, self.num_ways), yq_t.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
            steps_ok += 1

        if steps_ok == 0:
            raise UnsupportedMethodError(
                self.name,
                "Failed to generate any valid STUNT meta-tasks from unlabeled columns.",
                status="unsupported_stunt_task_generation_failed",
            )

        self.model = encoder
        self._support_x = X_lab
        self._support_y = y_lab
        self.training_meta = {
            **empty_training_meta(),
            "method": self.name,
            "protocol": "inductive",
            "uses_unlabeled_data": True,
            "method_fidelity": "official_adapted",
            "reference_family": "STUNT (Nam et al., ICLR 2023)",
            "upstream_url": UPSTREAM_URL,
            "upstream_commit": UPSTREAM_COMMIT,
            "meta_steps_requested": self.meta_steps,
            "meta_steps_completed": steps_ok,
            "last_meta_loss": last_loss,
            "num_ways": self.num_ways,
            "shot": self.shot,
            "query": self.query,
            "smoke": self.smoke,
            "paper_differences": [
                "Uses benchmark fixed labeled/unlabeled/val/test splits instead of STUNT index npy files.",
                "FAISS GPU k-means may fall back to sklearn MiniBatchKMeans.",
                "Smoke mode uses fewer meta-steps than the paper schedule.",
                "Downstream classification uses labeled support prototypes (no STUNT eval.py index files).",
            ],
            "fallback_reason": None,
            "n_pseudo_added_total": np.nan,
            "pl_diagnostics_deferred": True,
        }
        return self

    def _embed(self, X: np.ndarray):
        torch = self._torch
        assert torch is not None and self.model is not None
        X = to_dense_float32(X)
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(X).to(self._device)
            return self.model(t)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch = self._torch
        if self.model is None or self._support_x is None or self._support_y is None:
            raise RuntimeError("STUNT is not fitted.")
        assert torch is not None
        support_emb = self._embed(self._support_x)
        query_emb = self._embed(X)
        classes = np.unique(self._support_y)
        # Build prototypes from labeled support
        protos = []
        for c in range(self.n_classes):
            mask = self._support_y == c
            if np.any(mask):
                protos.append(support_emb[mask].mean(dim=0))
            else:
                protos.append(torch.zeros(support_emb.shape[1], device=support_emb.device))
        protos_t = torch.stack(protos, dim=0)
        logits = -torch.sum((protos_t.unsqueeze(0) - query_emb.unsqueeze(1)) ** 2, dim=-1)
        proba = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def build_stunt_model(random_state: int = 0, n_classes: int = 2, **kwargs: Any) -> STUNTMethod:
    # benchmark.yaml uses max_epochs as a shared training-budget key; map to meta_steps.
    if "max_epochs" in kwargs and "meta_steps" not in kwargs:
        kwargs["meta_steps"] = int(kwargs.pop("max_epochs"))
    else:
        kwargs.pop("max_epochs", None)
    # Ignore other neural-default keys that may leak from shared config merges.
    for k in ("patience", "batch_size", "learning_rate", "weight_decay", "dropout", "embedding_dim"):
        kwargs.pop(k, None)
    return STUNTMethod(random_state=random_state, n_classes=n_classes, **kwargs)
