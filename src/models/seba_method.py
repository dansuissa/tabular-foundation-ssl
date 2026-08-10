"""SeBA: Separated-at-Birth Alignment for tabular SS-FSL.

Official: https://github.com/kacper3615/SeBA

This module prefers the official trainer/models when the vendored tree is
importable; otherwise it runs a faithful reimplementation of Separated-at-Birth
Alignment (feature/target view split, NN alignment in the target view,
conditioned projector, InfoNCE) followed by a linear probe on labeled data.

Fidelity is labeled accurately in ``training_meta['method_fidelity']``.
Do NOT replace SeBA with generic contrastive learning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.exceptions import OptionalDependencyError, UnsupportedMethodError
from src.models.common import empty_training_meta
from src.models.torch_utils import set_global_determinism, to_dense_float32

LOGGER = logging.getLogger(__name__)

UPSTREAM_URL = "https://github.com/kacper3615/SeBA"
_THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party" / "seba"


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise OptionalDependencyError(
            "torch",
            "SeBA requires PyTorch. Install torch in ssl-tfm / ssl-representation.",
        ) from exc
    return torch, nn, F


def _try_import_official() -> dict[str, Any] | None:
    """Attempt to import vendored official SeBA modules."""
    if not _THIRD_PARTY.exists():
        return None
    # Official layout: models/models.py, trainers/pretrainer.py
    models_py = _THIRD_PARTY / "models" / "models.py"
    pretrainer_py = _THIRD_PARTY / "trainers" / "pretrainer.py"
    if not (models_py.exists() and pretrainer_py.exists()):
        return None
    import importlib.util
    import sys

    # Add third_party/seba to path temporarily
    root = str(_THIRD_PARTY)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from models.models import FNetwork, PNetwork, Classifier  # type: ignore
        from trainers.pretrainer import (  # type: ignore
            generate_mask,
            compute_nearest_neighbor_indices,
            compute_contrastive_loss,
        )

        return {
            "FNetwork": FNetwork,
            "PNetwork": PNetwork,
            "Classifier": Classifier,
            "generate_mask": generate_mask,
            "compute_nearest_neighbor_indices": compute_nearest_neighbor_indices,
            "compute_contrastive_loss": compute_contrastive_loss,
            "source": "official_vendored",
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.info("Official SeBA import failed (%s); using faithful reimplementation.", exc)
        return None


# ---------------------------------------------------------------------------
# Faithful reimplementation (mirrors official models + pretrainer)
# ---------------------------------------------------------------------------


def _build_faithful_modules(torch, nn, F, input_dim: int, hidden_dim: int, embed_dim: int):
    class FNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.layers(x)

    class PNetwork(nn.Module):
        """Conditioned projector: concat(encoder_z, separation_mask)."""

        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(hidden_dim + input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )

        def forward(self, z, mask):
            h = self.layers(torch.cat([z, mask], dim=-1))
            return F.normalize(h, p=2, dim=1)

    class Classifier(nn.Module):
        def __init__(self, f: nn.Module, num_classes: int):
            super().__init__()
            self.f = f
            self.head = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            return self.head(self.f(x))

    return FNetwork(), PNetwork(), Classifier


def _generate_mask(torch, X, masked_ratio: float, fill_mode: str = "zero"):
    """Column-group separation into feature (masked) and target (kept) views."""
    n_features = X.shape[1]
    k = max(1, int(masked_ratio * n_features))
    k = min(k, n_features - 1) if n_features > 1 else 1
    chosen = torch.randperm(n_features, device=X.device)[:k]
    X_masked = X.clone()
    mask = torch.ones_like(X)
    if fill_mode == "marginal":
        perm = torch.randperm(X.shape[0], device=X.device)
        X_masked[:, chosen] = X[perm][:, chosen]
    else:
        X_masked[:, chosen] = 0.0
    mask[:, chosen] = 0.0
    return X_masked, mask


def _nn_indices(torch, orig, k: int = 1):
    dists = torch.cdist(orig, orig)
    dists.fill_diagonal_(float("inf"))
    knn = dists.topk(k, largest=False).indices
    if k == 1:
        return knn[:, 0]
    choice = torch.randint(0, k, (orig.size(0),), device=orig.device)
    return knn[torch.arange(orig.size(0), device=orig.device), choice]


def _infonce(F, h, nn_idx, temperature: float):
    sim = h @ h.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    return F.cross_entropy(sim, nn_idx)


class SeBAMethod:
    """SeBA pretrain (Separated-at-Birth Alignment) + linear probe."""

    name = "seba"

    def __init__(
        self,
        random_state: int = 0,
        n_classes: int = 2,
        hidden_dim: int = 1024,
        embed_dim: int = 256,
        masked_ratio: float = 0.2,
        temperature: float = 0.2,
        pretrain_epochs: int = 50,
        probe_epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 20,
        smoke: bool = True,
        device: str | None = None,
        prefer_official: bool = True,
    ) -> None:
        self.random_state = int(random_state)
        self.n_classes = int(n_classes)
        self.hidden_dim = int(hidden_dim)
        self.embed_dim = int(embed_dim)
        self.masked_ratio = float(masked_ratio)
        self.temperature = float(temperature)
        self.pretrain_epochs = int(20 if smoke else pretrain_epochs)
        self.probe_epochs = int(30 if smoke else probe_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.patience = int(patience)
        self.smoke = bool(smoke)
        self.device_request = device
        self.prefer_official = bool(prefer_official)
        self.classifier_: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self.training_meta: dict[str, Any] = empty_training_meta()

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "SeBAMethod":
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
        if X_unlabeled is None or len(X_unlabeled) == 0:
            # Still allow labeled-only probe after trivial init — but SeBA needs unlabeled.
            raise UnsupportedMethodError(
                self.name,
                "SeBA pretraining requires an unlabeled feature pool.",
                status="unsupported_seba_no_unlabeled",
            )
        X_u = to_dense_float32(X_unlabeled)
        # Pretrain pool: unlabeled (+ labeled features without using labels for SSL)
        X_pre = np.vstack([X_u, X_lab])

        official = _try_import_official() if self.prefer_official else None
        fidelity = "official" if official else "faithful_reimplementation"

        in_dim = int(X_pre.shape[1])
        if official:
            f_model = official["FNetwork"](in_dim, self.hidden_dim).to(device)
            p_model = official["PNetwork"](self.hidden_dim, in_dim, self.embed_dim).to(device)
            gen_mask = official["generate_mask"]
            nn_fn = official["compute_nearest_neighbor_indices"]
            loss_fn = official["compute_contrastive_loss"]
            # Official generate_mask expects feature_groups dict
            feature_groups = {f"f{i}": [i] for i in range(in_dim)}

            def make_mask(Xb):
                return gen_mask(Xb, self.masked_ratio, feature_groups, fill_mode="zero")

            def make_nn(orig):
                return nn_fn(orig)

            def make_loss(h, nn_idx):
                return loss_fn(h, nn_idx, self.temperature)
        else:
            f_model, p_model, ClassifierFactory = _build_faithful_modules(
                torch, nn, F, in_dim, self.hidden_dim, self.embed_dim
            )
            f_model = f_model.to(device)
            p_model = p_model.to(device)

            def make_mask(Xb):
                return _generate_mask(torch, Xb, self.masked_ratio, fill_mode="zero")

            def make_nn(orig):
                return _nn_indices(torch, orig, k=1)

            def make_loss(h, nn_idx):
                return _infonce(F, h, nn_idx, self.temperature)

            ClassifierFactory = ClassifierFactory  # noqa: F841

        opt = torch.optim.Adam(
            list(f_model.parameters()) + list(p_model.parameters()), lr=self.lr
        )

        # Optional unsupervised val from unlabeled holdout for early stopping
        rng = np.random.RandomState(self.random_state)
        n_pre = X_pre.shape[0]
        perm = rng.permutation(n_pre)
        n_val = max(1, int(0.1 * n_pre))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        X_pre_tr = X_pre[train_idx]
        X_pre_va = X_pre[val_idx]

        best_state = None
        best_val = float("inf")
        no_improve = 0
        last_val = float("nan")

        def _batches(X_np, bs):
            n = X_np.shape[0]
            order = rng.permutation(n)
            for start in range(0, n, bs):
                yield X_np[order[start : start + bs]]

        for epoch in range(1, self.pretrain_epochs + 1):
            f_model.train()
            p_model.train()
            for xb in _batches(X_pre_tr, self.batch_size):
                if xb.shape[0] < 2:
                    continue
                Xt = torch.from_numpy(xb).to(device)
                X_masked, mask = make_mask(Xt)
                Z = f_model(X_masked)
                H = p_model(Z, mask)
                with torch.no_grad():
                    orig = Xt * (1.0 - mask)  # target view
                    nn_idx = make_nn(orig)
                loss = make_loss(H, nn_idx)
                opt.zero_grad()
                loss.backward()
                opt.step()

            # Val InfoNCE
            f_model.eval()
            p_model.eval()
            val_losses = []
            with torch.no_grad():
                for xb in _batches(X_pre_va, self.batch_size):
                    if xb.shape[0] < 2:
                        continue
                    Xt = torch.from_numpy(xb).to(device)
                    X_masked, mask = make_mask(Xt)
                    Z = f_model(X_masked)
                    H = p_model(Z, mask)
                    orig = Xt * (1.0 - mask)
                    nn_idx = make_nn(orig)
                    val_losses.append(float(make_loss(H, nn_idx).item()))
            last_val = float(np.mean(val_losses)) if val_losses else float("nan")
            if np.isfinite(last_val) and last_val < best_val - 1e-6:
                best_val = last_val
                no_improve = 0
                best_state = {
                    "f": {k: v.detach().cpu().clone() for k, v in f_model.state_dict().items()},
                    "p": {k: v.detach().cpu().clone() for k, v in p_model.state_dict().items()},
                }
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            f_model.load_state_dict(best_state["f"])
            p_model.load_state_dict(best_state["p"])

        # Freeze encoder; linear probe on labeled
        for p in f_model.parameters():
            p.requires_grad = False
        clf = nn.Linear(self.hidden_dim, self.n_classes).to(device)
        opt_clf = torch.optim.Adam(clf.parameters(), lr=self.lr)
        ce = nn.CrossEntropyLoss()

        X_lab_t = torch.from_numpy(X_lab).to(device)
        y_lab_t = torch.from_numpy(y_lab).to(device)
        # Use labeled val if provided for early stopping of probe
        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        if has_val:
            X_val_t = torch.from_numpy(to_dense_float32(X_val)).to(device)
            y_val_t = torch.from_numpy(np.asarray(y_val).astype(np.int64)).to(device)

        best_probe = None
        best_probe_loss = float("inf")
        f_model.eval()
        for epoch in range(1, self.probe_epochs + 1):
            clf.train()
            with torch.no_grad():
                z = f_model(X_lab_t)
            logits = clf(z)
            loss = ce(logits, y_lab_t)
            opt_clf.zero_grad()
            loss.backward()
            opt_clf.step()
            if has_val:
                clf.eval()
                with torch.no_grad():
                    zv = f_model(X_val_t)
                    lv = float(ce(clf(zv), y_val_t).item())
                if lv < best_probe_loss:
                    best_probe_loss = lv
                    best_probe = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
        if best_probe is not None:
            clf.load_state_dict(best_probe)

        class _FrozenProbe(torch.nn.Module):
            def __init__(self, enc, head):
                super().__init__()
                self.enc = enc
                self.head = head

            def forward(self, x):
                return self.head(self.enc(x))

            def predict_proba_np(self, X_np):
                self.eval()
                with torch.no_grad():
                    t = torch.from_numpy(to_dense_float32(X_np)).to(device)
                    logits = self.forward(t)
                    return torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)

        self.classifier_ = _FrozenProbe(f_model, clf)
        self.training_meta = {
            **empty_training_meta(),
            "method": self.name,
            "protocol": "inductive",
            "uses_unlabeled_data": True,
            "method_fidelity": fidelity,
            "reference_family": "SeBA / Separated-at-Birth Alignment (Jurek et al.)",
            "upstream_url": UPSTREAM_URL,
            "pretrain_epochs": self.pretrain_epochs,
            "probe_epochs": self.probe_epochs,
            "masked_ratio": self.masked_ratio,
            "temperature": self.temperature,
            "best_pretrain_val_infonce": best_val if np.isfinite(best_val) else last_val,
            "smoke": self.smoke,
            "algorithm_components": [
                "feature_target_view_split",
                "nearest_neighbor_alignment_in_target_view",
                "conditioned_projector",
                "InfoNCE",
                "linear_probe_on_labeled",
            ],
            "not_generic_contrastive": True,
            "fallback_reason": None,
            "pl_diagnostics_deferred": True,
        }
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classifier_ is None:
            raise RuntimeError("SeBA is not fitted.")
        return self.classifier_.predict_proba_np(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def build_seba_model(random_state: int = 0, n_classes: int = 2, **kwargs: Any) -> SeBAMethod:
    if "max_epochs" in kwargs and "pretrain_epochs" not in kwargs:
        kwargs["pretrain_epochs"] = int(kwargs.pop("max_epochs"))
    else:
        kwargs.pop("max_epochs", None)
    for k in ("patience", "learning_rate", "weight_decay", "dropout", "embedding_dim"):
        kwargs.pop(k, None)
    return SeBAMethod(random_state=random_state, n_classes=n_classes, **kwargs)
