"""Inductive D2R2-c: diffusion representation + random distance matching + mean prototypes.

Reference: https://github.com/Carol-cloud-project/D2R2
Paper: Fang et al., NeurIPS 2024 — D2R2.

Primary method ``d2r2_c`` / ``d2r2c_inductive`` uses **mean** support prototypes
(D2R2-c). It never refines prototypes with query/test statistics.

The transductive instance-wise iterative prototype (IP) variant is exposed
separately as ``d2r2_transductive`` and must not enter the primary inductive
ranking.

Official upstream is JAX/Flax-heavy; this module provides a faithful PyTorch
port of the inductive D2R2-c recipe with conservative smoke defaults.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.exceptions import OptionalDependencyError, UnsupportedMethodError
from src.models.common import empty_training_meta
from src.models.torch_utils import set_global_determinism, to_dense_float32

LOGGER = logging.getLogger(__name__)

UPSTREAM_URL = "https://github.com/Carol-cloud-project/D2R2"


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise OptionalDependencyError(
            "torch",
            "D2R2-c requires PyTorch. Install torch in ssl-tfm / ssl-representation.",
        ) from exc
    return torch, nn, F


def _vp_beta_schedule(T: int) -> np.ndarray:
    """Variance-preserving beta schedule (matches D2R2 vp option qualitatively)."""
    t = np.arange(1, T + 1, dtype=np.float64)
    return (1.0 - np.exp(-0.1 * t / T)).astype(np.float64) * 0.02 + 1e-4


def _make_mlp(nn, in_dim: int, hidden: int, out_dim: int, depth: int = 2, dropout: float = 0.0):
    layers = []
    prev = in_dim
    for i in range(depth):
        layers.append(nn.Linear(prev, hidden))
        layers.append(nn.Mish())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = hidden
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class D2R2CInductive:
    """Inductive D2R2-c with mean support prototypes."""

    name = "d2r2_c"

    def __init__(
        self,
        random_state: int = 0,
        n_classes: int = 2,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        diffusion_steps: int = 10,
        train_steps: int = 200,
        batch_size: int = 128,
        lr: float = 3e-4,
        alpha_rdm: float = 0.1,
        col_select_ratio: float = 0.5,
        smoke: bool = True,
        device: str | None = None,
    ) -> None:
        self.random_state = int(random_state)
        self.n_classes = int(n_classes)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.diffusion_steps = int(diffusion_steps)
        self.train_steps = int(80 if smoke else train_steps)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.alpha_rdm = float(alpha_rdm)
        self.col_select_ratio = float(col_select_ratio)
        self.smoke = bool(smoke)
        self.device_request = device
        self.embed_model: Any | None = None
        self.noise_model: Any | None = None
        self._support_x: np.ndarray | None = None
        self._support_y: np.ndarray | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._alpha_hats: Any | None = None
        self.training_meta: dict[str, Any] = empty_training_meta()

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "D2R2CInductive":
        torch, nn, F = _require_torch()
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
        if X_unlabeled is not None and len(X_unlabeled) > 0:
            X_u = to_dense_float32(X_unlabeled)
            X_train = np.vstack([X_lab, X_u])
        else:
            X_train = X_lab

        x_dim = int(X_train.shape[1])
        # Treat all dims as numeric for the PyTorch port (processed view).
        num_idx = np.arange(x_dim)
        cate_idx = np.array([], dtype=int)

        embed_model = _make_mlp(nn, x_dim + 1, self.hidden_dim, self.embed_dim, depth=2).to(device)
        # noise predictor: concat(Xt, t_emb, embed) → eps
        noise_in = x_dim + 1 + self.embed_dim
        noise_model = _make_mlp(nn, noise_in, self.hidden_dim, x_dim, depth=2).to(device)
        opt = torch.optim.Adam(
            list(embed_model.parameters()) + list(noise_model.parameters()), lr=self.lr
        )

        betas = np.concatenate([[0.0], _vp_beta_schedule(self.diffusion_steps)])
        alphas = 1.0 - betas
        alpha_hats = np.cumprod(alphas)
        self._alpha_hats = torch.tensor(alpha_hats, dtype=torch.float32, device=device)

        rng = np.random.RandomState(self.random_state)
        std_scale = X_train.std(axis=0) + 1e-6
        std_scale_t = torch.tensor(std_scale, dtype=torch.float32, device=device)

        X_t_all = torch.from_numpy(X_train).to(device)
        n = X_train.shape[0]
        last_loss = float("nan")

        for step in range(self.train_steps):
            idx = rng.choice(n, size=min(self.batch_size, n), replace=False)
            Xb = X_t_all[idx]
            bsz = Xb.shape[0]
            t = torch.randint(1, self.diffusion_steps + 1, (bsz, 1), device=device)
            eps = torch.randn_like(Xb)
            a_hat = self._alpha_hats[t.view(-1)].view(-1, 1)
            Xt = torch.sqrt(a_hat) * Xb + torch.sqrt(1.0 - a_hat) * eps

            # Embedding with small noise (as in upstream)
            z_noise = torch.randn_like(Xb) * std_scale_t
            embed_in = torch.cat([Xb + z_noise, t.float() / self.diffusion_steps], dim=1)
            embed = embed_model(embed_in)

            noise_in_t = torch.cat([Xt, t.float() / self.diffusion_steps, embed], dim=1)
            pred_eps = noise_model(noise_in_t)
            noise_loss = ((pred_eps - eps) ** 2).sum(dim=1).mean()

            # Random distance matching (RDM)
            p = self.col_select_ratio
            A = float(np.sqrt(3 * p * (1 - p) + 1e-12))
            n_num = len(num_idx)
            W1 = (torch.rand(n_num, n_num, device=device) * 2 - 1) * A
            W1 = W1 / (n_num ** 0.5 + 1e-6)
            num_map = Xb[:, num_idx] @ W1
            if len(cate_idx):
                # Bernoulli map for categorical (none in processed numeric path)
                rand_proj = num_map
            else:
                rand_proj = num_map
            normed_proj = rand_proj / (rand_proj.norm(dim=1, keepdim=True) + 1e-6)
            normed_embed = embed / (embed.norm(dim=1, keepdim=True).detach() + 1e-6)
            perm = torch.arange(bsz, device=device) - 1
            perm = perm % bsz
            embed_dist = ((normed_embed - normed_embed[perm]) ** 2).sum(dim=1)
            rand_dist = ((normed_proj - normed_proj[perm]) ** 2).sum(dim=1)
            rdm_loss = ((embed_dist - rand_dist) ** 2).mean()

            loss = noise_loss + self.alpha_rdm * rdm_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())

        self.embed_model = embed_model
        self.noise_model = noise_model
        self._support_x = X_lab
        self._support_y = y_lab
        self.training_meta = {
            **empty_training_meta(),
            "method": self.name,
            "protocol": "inductive",
            "uses_unlabeled_data": bool(X_unlabeled is not None and len(X_unlabeled) > 0),
            "method_fidelity": "faithful_reimplementation",
            "reference_family": "D2R2-c (Fang et al., NeurIPS 2024)",
            "upstream_url": UPSTREAM_URL,
            "prototype_mode": "mean_support",
            "not_instance_wise_iterative_prototypes": True,
            "train_steps": self.train_steps,
            "diffusion_steps": self.diffusion_steps,
            "alpha_rdm": self.alpha_rdm,
            "last_train_loss": last_loss,
            "smoke": self.smoke,
            "paper_differences": [
                "PyTorch port of JAX/Flax official code; VP schedule simplified.",
                "Processed numeric features only (no native mixed-type cat maps).",
                "Conservative smoke step counts by default.",
                "Classification uses mean support prototypes (D2R2-c), not IP.",
            ],
            "fallback_reason": None,
            "pl_diagnostics_deferred": True,
        }
        return self

    def _embed(self, X: np.ndarray):
        torch = self._torch
        assert torch is not None and self.embed_model is not None
        X = to_dense_float32(X)
        t = torch.ones((X.shape[0], 1), device=self._device) * float(self.diffusion_steps)
        inp = torch.cat(
            [torch.from_numpy(X).to(self._device), t / self.diffusion_steps], dim=1
        )
        self.embed_model.eval()
        with torch.no_grad():
            return self.embed_model(inp)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch = self._torch
        if self.embed_model is None or self._support_x is None or self._support_y is None:
            raise RuntimeError("D2R2-c is not fitted.")
        assert torch is not None
        q = self._embed(X)
        s = self._embed(self._support_x)
        protos = []
        for c in range(self.n_classes):
            mask = self._support_y == c
            if np.any(mask):
                protos.append(s[mask].mean(dim=0))
            else:
                protos.append(torch.zeros(s.shape[1], device=s.device))
        protos_t = torch.stack(protos, dim=0)
        logits = -torch.sum((protos_t.unsqueeze(0) - q.unsqueeze(1)) ** 2, dim=-1)
        return torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class D2R2Transductive(D2R2CInductive):
    """Transductive IP variant — excluded from primary inductive ranking.

    Raises unless explicitly enabled, because iterative prototypes using
    query/test embeddings violate the inductive protocol.
    """

    name = "d2r2_transductive"

    def __init__(self, allow_transductive: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.allow_transductive = bool(allow_transductive)
        self.name = "d2r2_transductive"

    def fit(self, *args: Any, **kwargs: Any) -> "D2R2Transductive":
        if not self.allow_transductive:
            self.training_meta = {
                **empty_training_meta(),
                "method": self.name,
                "protocol": "transductive",
                "method_group": "transductive_exploratory",
                "method_fidelity": "unsupported",
                "reference_family": "D2R2 IP (transductive)",
            }
            raise UnsupportedMethodError(
                self.name,
                (
                    "d2r2_transductive uses instance-wise iterative prototypes that "
                    "adapt using query/test embeddings. It is excluded from the "
                    "primary inductive ranking. Pass allow_transductive=True only "
                    "inside the transductive_exploratory group."
                ),
                status="unsupported_transductive_not_in_primary",
            )
        # Even when allowed, we still document protocol and do not implement
        # query leakage by default — raise until a verified IP path is wired.
        raise UnsupportedMethodError(
            self.name,
            "Faithful D2R2 IP transductive path is not enabled in this build.",
            status="unsupported_d2r2_ip_not_enabled",
        )


def build_d2r2_model(
    method: str = "d2r2_c",
    random_state: int = 0,
    n_classes: int = 2,
    **kwargs: Any,
) -> Any:
    if "max_epochs" in kwargs and "train_steps" not in kwargs:
        kwargs["train_steps"] = int(kwargs.pop("max_epochs"))
    else:
        kwargs.pop("max_epochs", None)
    for k in ("patience", "learning_rate", "weight_decay", "dropout", "embedding_dim"):
        kwargs.pop(k, None)
    if method in {"d2r2_c", "d2r2c_inductive", "d2r2_c_inductive"}:
        return D2R2CInductive(random_state=random_state, n_classes=n_classes, **kwargs)
    if method == "d2r2_transductive":
        return D2R2Transductive(random_state=random_state, n_classes=n_classes, **kwargs)
    raise KeyError(f"Unknown D2R2 method: {method}")
