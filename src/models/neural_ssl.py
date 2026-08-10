"""CPU-friendly neural SSL baselines: SSLAE, VIME-lite and SCARF.

These are intentionally lightweight, CPU-first implementations that consume the
same fixed sklearn-preprocessed numeric matrices as the classical first-wave
methods. PyTorch is imported lazily so this module is importable without torch;
training raises a clean ImportError that the runner records per-run.

VIME-lite is inspired by the VIME-style feature/mask reconstruction objective
and is NOT a faithful reproduction of the original paper.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.models.torch_utils import (
    feature_shuffle_corrupt,
    internal_val_split,
    iterate_minibatches,
    make_mlp,
    normalize_probability_matrix,
    require_torch,
    set_global_determinism,
    to_dense_float32,
    validate_probability_matrix,
)

# Faithful-core methods plus the experimental vime_lite ablation. Dispatch
# accepts all of these; configs/benchmark.yaml controls which are run by group.
NEURAL_SSL_METHODS = {"sslae", "vime", "vime_lite", "scarf"}

# Conservative, CPU-friendly defaults. The runner may override these from
# configs/benchmark.yaml (neural_ssl_defaults + per-method blocks).
NEURAL_DEFAULTS: dict[str, Any] = {
    "hidden_dim": 128,
    "embedding_dim": 64,
    "depth": 2,
    "dropout": 0.1,
    "batch_size": 256,
    "max_epochs": 100,
    "pretrain_epochs": 50,
    "patience": 10,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "device": "cpu",
    "num_workers": 0,
    "val_fraction": 0.2,
}

SSLAE_DEFAULTS = {"recon_weight": 1.0}
VIME_LITE_DEFAULTS = {"mask_prob": 0.3, "mask_loss_weight": 1.0, "consistency_weight": 0.1}
VIME_DEFAULTS = {
    "mask_prob": 0.3,
    "mask_loss_weight": 1.0,
    "feature_loss_weight": 1.0,
    "consistency_weight": 0.1,
}
SCARF_DEFAULTS = {"corruption_rate": 0.3, "temperature": 0.1, "projection_dim": 64}


class _NeuralSSLBase:
    """Shared scaffolding for the neural SSL baselines."""

    method = "neural_ssl"

    def __init__(self, random_state: int = 0, n_classes: int = 2, **params: Any) -> None:
        self.random_state = int(random_state)
        self.n_classes = int(n_classes)
        cfg = dict(NEURAL_DEFAULTS)
        cfg.update(params)
        self.cfg = cfg
        self.classes_: np.ndarray | None = None
        self.encoder = None
        self.classifier = None
        self.training_meta: dict[str, Any] = {}

    # -- shared helpers -------------------------------------------------
    def _prepare_inputs(self, X_labeled, y_labeled, X_unlabeled):
        X_labeled = to_dense_float32(X_labeled)
        y_labeled = np.asarray(y_labeled)
        self.classes_ = np.unique(y_labeled)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_local = np.array([class_to_idx[c] for c in y_labeled], dtype=np.int64)
        X_unlabeled = (
            to_dense_float32(X_unlabeled)
            if X_unlabeled is not None and len(X_unlabeled) > 0
            else np.empty((0, X_labeled.shape[1]), dtype=np.float32)
        )
        return X_labeled, y_local, X_unlabeled

    def _unsupervised_matrix(self, X_labeled, X_unlabeled):
        if X_unlabeled.shape[0] > 0:
            return np.vstack([X_labeled, X_unlabeled]).astype(np.float32)
        return X_labeled.astype(np.float32)

    def _build_encoder_classifier(self, torch, input_dim: int, n_local_classes: int):
        cfg = self.cfg
        encoder = make_mlp(
            torch,
            in_dim=input_dim,
            hidden_dim=cfg["hidden_dim"],
            out_dim=cfg["embedding_dim"],
            depth=cfg["depth"],
            dropout=cfg["dropout"],
        )
        classifier = torch.nn.Linear(cfg["embedding_dim"], n_local_classes)
        return encoder, classifier

    def _adam(self, torch, modules: list):
        params: list = []
        for module in modules:
            params += list(module.parameters())
        return torch.optim.Adam(
            params,
            lr=self.cfg["learning_rate"],
            weight_decay=self.cfg["weight_decay"],
        )

    def _train_classifier(
        self,
        torch,
        X_labeled: np.ndarray,
        y_local: np.ndarray,
        X_unlabeled: np.ndarray,
        consistency_weight: float = 0.0,
        corruption_rate: float = 0.0,
    ) -> dict[str, Any]:
        """Finetune encoder+classifier with optional unlabeled consistency.

        Uses an internal stratified validation holdout for early stopping when
        possible; otherwise trains for max_epochs with training-loss patience.
        """
        cfg = self.cfg
        device = torch.device(cfg["device"])
        rng = np.random.default_rng(self.random_state)

        X_tr, y_tr, X_val, y_val, val_strategy = internal_val_split(
            X_labeled,
            y_local,
            val_fraction=cfg["val_fraction"],
            seed=self.random_state,
        )

        ce = torch.nn.CrossEntropyLoss()
        optim = self._adam(torch, [self.encoder, self.classifier])

        X_tr_t = torch.from_numpy(np.asarray(X_tr, dtype=np.float32)).to(device)
        y_tr_t = torch.from_numpy(np.asarray(y_tr, dtype=np.int64)).to(device)
        X_val_t = (
            torch.from_numpy(np.asarray(X_val, dtype=np.float32)).to(device)
            if X_val is not None
            else None
        )
        y_val_t = (
            torch.from_numpy(np.asarray(y_val, dtype=np.int64)).to(device)
            if y_val is not None
            else None
        )
        X_unlab_t = (
            torch.from_numpy(np.asarray(X_unlabeled, dtype=np.float32)).to(device)
            if (consistency_weight > 0 and X_unlabeled.shape[0] > 0)
            else None
        )

        best_loss = math.inf
        best_epoch = -1
        best_state: dict | None = None
        epochs_trained = 0
        patience = cfg["patience"]
        bad_epochs = 0
        final_train_loss = math.nan

        for epoch in range(cfg["max_epochs"]):
            self.encoder.train()
            self.classifier.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch_idx in iterate_minibatches(
                len(X_tr_t), cfg["batch_size"], rng, shuffle=True
            ):
                xb = X_tr_t[batch_idx]
                yb = y_tr_t[batch_idx]
                optim.zero_grad()
                emb = self.encoder(xb)
                logits = self.classifier(emb)
                loss = ce(logits, yb)
                if consistency_weight > 0 and corruption_rate > 0:
                    loss = loss + consistency_weight * self._consistency_loss(
                        torch, xb, corruption_rate
                    )
                    if X_unlab_t is not None:
                        u_idx = rng.integers(0, X_unlab_t.shape[0], size=min(len(xb), X_unlab_t.shape[0]))
                        xu = X_unlab_t[torch.from_numpy(u_idx).to(device)]
                        loss = loss + consistency_weight * self._consistency_loss(
                            torch, xu, corruption_rate
                        )
                loss.backward()
                optim.step()
                epoch_loss += float(loss.detach().cpu())
                n_batches += 1
            epochs_trained = epoch + 1
            final_train_loss = epoch_loss / max(n_batches, 1)

            # Early stopping signal: val CE if available, else train loss.
            if X_val_t is not None:
                self.encoder.eval()
                self.classifier.eval()
                with torch.no_grad():
                    val_logits = self.classifier(self.encoder(X_val_t))
                    monitor = float(ce(val_logits, y_val_t).cpu())
            else:
                monitor = final_train_loss

            if monitor < best_loss - 1e-5:
                best_loss = monitor
                best_epoch = epoch + 1
                best_state = {
                    "encoder": {k: v.detach().clone() for k, v in self.encoder.state_dict().items()},
                    "classifier": {
                        k: v.detach().clone() for k, v in self.classifier.state_dict().items()
                    },
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        if best_state is not None:
            self.encoder.load_state_dict(best_state["encoder"])
            self.classifier.load_state_dict(best_state["classifier"])

        return {
            "finetune_epochs_trained": epochs_trained,
            "best_epoch": best_epoch,
            "best_val_loss": (best_loss if X_val_t is not None and math.isfinite(best_loss) else math.nan),
            "final_train_loss": final_train_loss,
            "neural_validation_strategy": val_strategy,
        }

    def _consistency_loss(self, torch, x, corruption_rate: float):
        mask = (torch.rand_like(x) < corruption_rate).float()
        x_corrupt = feature_shuffle_corrupt(torch, x, mask)
        p_clean = torch.softmax(self.classifier(self.encoder(x)), dim=1)
        p_corrupt = torch.softmax(self.classifier(self.encoder(x_corrupt)), dim=1)
        return torch.mean((p_clean - p_corrupt) ** 2)

    # -- prediction -----------------------------------------------------
    def _forward_proba(self, X: np.ndarray) -> np.ndarray:
        torch = require_torch()
        device = torch.device(self.cfg["device"])
        X = to_dense_float32(X)
        self.encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            xb = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
            logits = self.classifier(self.encoder(xb))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        # float32 softmax can sum to ~1±1e-7; renormalize in float64 then validate
        # so probability metrics (log_loss, roc_auc) never see malformed rows.
        proba = normalize_probability_matrix(proba)
        return validate_probability_matrix(proba, n_classes=len(self.classes_))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self._forward_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


class SSLAEModel(_NeuralSSLBase):
    """Semi-supervised supervised autoencoder.

    Reconstruction loss on labeled + unlabeled; classification loss on labeled
    only; trained jointly end-to-end.
    """

    name = "sslae"
    method = "sslae"

    def __init__(self, random_state: int = 0, n_classes: int = 2, **params: Any) -> None:
        merged = dict(SSLAE_DEFAULTS)
        merged.update(params)
        super().__init__(random_state=random_state, n_classes=n_classes, **merged)

    def fit(self, X_labeled, y_labeled, X_unlabeled=None) -> "SSLAEModel":
        torch = require_torch()
        set_global_determinism(self.random_state)
        cfg = self.cfg
        device = torch.device(cfg["device"])
        rng = np.random.default_rng(self.random_state)

        X_lab, y_local, X_unlab = self._prepare_inputs(X_labeled, y_labeled, X_unlabeled)
        input_dim = X_lab.shape[1]
        n_local = len(self.classes_)

        self.encoder, self.classifier = self._build_encoder_classifier(torch, input_dim, n_local)
        self.decoder = make_mlp(
            torch,
            in_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
            out_dim=input_dim,
            depth=cfg["depth"],
            dropout=cfg["dropout"],
        )

        X_tr, y_tr, X_val, y_val, val_strategy = internal_val_split(
            X_lab, y_local, val_fraction=cfg["val_fraction"], seed=self.random_state
        )

        ce = torch.nn.CrossEntropyLoss()
        mse = torch.nn.MSELoss()
        optim = self._adam(torch, [self.encoder, self.classifier, self.decoder])

        X_tr_t = torch.from_numpy(np.asarray(X_tr, dtype=np.float32)).to(device)
        y_tr_t = torch.from_numpy(np.asarray(y_tr, dtype=np.int64)).to(device)
        X_unlab_t = (
            torch.from_numpy(np.asarray(X_unlab, dtype=np.float32)).to(device)
            if X_unlab.shape[0] > 0
            else None
        )
        X_val_t = (
            torch.from_numpy(np.asarray(X_val, dtype=np.float32)).to(device)
            if X_val is not None
            else None
        )
        y_val_t = (
            torch.from_numpy(np.asarray(y_val, dtype=np.int64)).to(device)
            if y_val is not None
            else None
        )
        recon_weight = cfg["recon_weight"]

        best_loss = math.inf
        best_epoch = -1
        best_state: dict | None = None
        epochs_trained = 0
        bad_epochs = 0
        final_train_loss = math.nan

        for epoch in range(cfg["max_epochs"]):
            self.encoder.train()
            self.classifier.train()
            self.decoder.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch_idx in iterate_minibatches(len(X_tr_t), cfg["batch_size"], rng, shuffle=True):
                xb = X_tr_t[batch_idx]
                yb = y_tr_t[batch_idx]
                optim.zero_grad()
                emb = self.encoder(xb)
                cls_loss = ce(self.classifier(emb), yb)
                recon_loss = mse(self.decoder(emb), xb)
                loss = cls_loss + recon_weight * recon_loss
                loss.backward()
                optim.step()
                epoch_loss += float(loss.detach().cpu())
                n_batches += 1
            # Unlabeled reconstruction-only pass.
            if X_unlab_t is not None:
                for batch_idx in iterate_minibatches(
                    len(X_unlab_t), cfg["batch_size"], rng, shuffle=True
                ):
                    xu = X_unlab_t[batch_idx]
                    optim.zero_grad()
                    recon_loss = recon_weight * mse(self.decoder(self.encoder(xu)), xu)
                    recon_loss.backward()
                    optim.step()
            epochs_trained = epoch + 1
            final_train_loss = epoch_loss / max(n_batches, 1)

            if X_val_t is not None:
                self.encoder.eval()
                self.classifier.eval()
                with torch.no_grad():
                    monitor = float(ce(self.classifier(self.encoder(X_val_t)), y_val_t).cpu())
            else:
                monitor = final_train_loss

            if monitor < best_loss - 1e-5:
                best_loss = monitor
                best_epoch = epoch + 1
                best_state = {
                    "encoder": {k: v.detach().clone() for k, v in self.encoder.state_dict().items()},
                    "classifier": {
                        k: v.detach().clone() for k, v in self.classifier.state_dict().items()
                    },
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg["patience"]:
                    break

        if best_state is not None:
            self.encoder.load_state_dict(best_state["encoder"])
            self.classifier.load_state_dict(best_state["classifier"])

        self.training_meta = {
            "neural_method": "sslae",
            "method_fidelity": "style_baseline",
            "reference_family": "supervised_autoencoder",
            "best_epoch": best_epoch,
            "epochs_trained": epochs_trained,
            "pretrain_epochs_trained": np.nan,
            "finetune_epochs_trained": np.nan,
            "best_val_loss": (best_loss if X_val_t is not None and math.isfinite(best_loss) else np.nan),
            "final_train_loss": final_train_loss,
            "neural_validation_strategy": val_strategy,
            "recon_weight": recon_weight,
        }
        return self


class VIMELiteModel(_NeuralSSLBase):
    """VIME-lite: feature/mask reconstruction pretraining + classifier finetune.

    Inspired by VIME-style feature/mask reconstruction; not a faithful
    reproduction of the original method.
    """

    name = "vime_lite"
    method = "vime_lite"

    def __init__(self, random_state: int = 0, n_classes: int = 2, **params: Any) -> None:
        merged = dict(VIME_LITE_DEFAULTS)
        merged.update(params)
        super().__init__(random_state=random_state, n_classes=n_classes, **merged)

    def fit(self, X_labeled, y_labeled, X_unlabeled=None) -> "VIMELiteModel":
        torch = require_torch()
        set_global_determinism(self.random_state)
        cfg = self.cfg
        device = torch.device(cfg["device"])
        rng = np.random.default_rng(self.random_state)

        X_lab, y_local, X_unlab = self._prepare_inputs(X_labeled, y_labeled, X_unlabeled)
        input_dim = X_lab.shape[1]
        n_local = len(self.classes_)

        self.encoder, self.classifier = self._build_encoder_classifier(torch, input_dim, n_local)
        recon_head = make_mlp(
            torch, cfg["embedding_dim"], cfg["hidden_dim"], input_dim, cfg["depth"], cfg["dropout"]
        )
        mask_head = make_mlp(
            torch, cfg["embedding_dim"], cfg["hidden_dim"], input_dim, cfg["depth"], cfg["dropout"]
        )

        X_unsup = self._unsupervised_matrix(X_lab, X_unlab)
        X_unsup_t = torch.from_numpy(X_unsup).to(device)

        mse = torch.nn.MSELoss()
        bce = torch.nn.BCEWithLogitsLoss()
        pre_optim = self._adam(torch, [self.encoder, recon_head, mask_head])

        mask_prob = cfg["mask_prob"]
        pretrain_epochs_trained = 0
        for _ in range(cfg["pretrain_epochs"]):
            self.encoder.train()
            recon_head.train()
            mask_head.train()
            for batch_idx in iterate_minibatches(
                len(X_unsup_t), cfg["batch_size"], rng, shuffle=True
            ):
                xb = X_unsup_t[batch_idx]
                mask = (torch.rand_like(xb) < mask_prob).float()
                x_corrupt = feature_shuffle_corrupt(torch, xb, mask)
                pre_optim.zero_grad()
                emb = self.encoder(x_corrupt)
                recon_loss = mse(recon_head(emb), xb)
                mask_loss = bce(mask_head(emb), mask)
                loss = recon_loss + cfg["mask_loss_weight"] * mask_loss
                loss.backward()
                pre_optim.step()
            pretrain_epochs_trained += 1

        finetune_meta = self._train_classifier(
            torch,
            X_lab,
            y_local,
            X_unlab,
            consistency_weight=cfg["consistency_weight"],
            corruption_rate=mask_prob,
        )

        self.training_meta = {
            "neural_method": "vime_lite",
            "method_fidelity": "lite_ablation",
            "reference_family": "VIME",
            "best_epoch": finetune_meta["best_epoch"],
            "epochs_trained": finetune_meta["finetune_epochs_trained"],
            "pretrain_epochs_trained": pretrain_epochs_trained,
            "finetune_epochs_trained": finetune_meta["finetune_epochs_trained"],
            "best_val_loss": finetune_meta["best_val_loss"],
            "final_train_loss": finetune_meta["final_train_loss"],
            "neural_validation_strategy": finetune_meta["neural_validation_strategy"],
            "mask_prob": mask_prob,
            "mask_loss_weight": cfg["mask_loss_weight"],
            "consistency_weight": cfg["consistency_weight"],
        }
        return self


class VIMEModel(_NeuralSSLBase):
    """Faithful-core VIME (Yoon et al., 2020).

    Stage 1 — self-supervised pretraining on labeled + unlabeled X:
      sample a Bernoulli mask M, build a corrupted view X_tilde (masked features
      replaced by values resampled from the feature marginals), and train the
      encoder with two heads to (a) recover the mask M (BCE) and (b) reconstruct
      the original features X (MSE).

    Stage 2 — semi-supervised fine-tuning:
      cross-entropy on labeled data plus a consistency loss between the predicted
      probabilities of two independently corrupted views of each unlabeled batch.

    This is a faithful reproduction of VIME's core objectives (mask estimation +
    feature reconstruction pretraining, consistency-regularized fine-tuning),
    not the lightweight ``vime_lite`` ablation.
    """

    name = "vime"
    method = "vime"

    def __init__(self, random_state: int = 0, n_classes: int = 2, **params: Any) -> None:
        merged = dict(VIME_DEFAULTS)
        merged.update(params)
        super().__init__(random_state=random_state, n_classes=n_classes, **merged)

    def _proba(self, torch, x):
        return torch.softmax(self.classifier(self.encoder(x)), dim=1)

    def fit(self, X_labeled, y_labeled, X_unlabeled=None) -> "VIMEModel":
        torch = require_torch()
        set_global_determinism(self.random_state)
        cfg = self.cfg
        device = torch.device(cfg["device"])
        rng = np.random.default_rng(self.random_state)

        X_lab, y_local, X_unlab = self._prepare_inputs(X_labeled, y_labeled, X_unlabeled)
        input_dim = X_lab.shape[1]
        n_local = len(self.classes_)

        self.encoder, self.classifier = self._build_encoder_classifier(torch, input_dim, n_local)
        recon_head = make_mlp(
            torch, cfg["embedding_dim"], cfg["hidden_dim"], input_dim, cfg["depth"], cfg["dropout"]
        )
        mask_head = make_mlp(
            torch, cfg["embedding_dim"], cfg["hidden_dim"], input_dim, cfg["depth"], cfg["dropout"]
        )

        # --- Stage 1: self-supervised mask + feature reconstruction ---------
        X_unsup = self._unsupervised_matrix(X_lab, X_unlab)
        X_unsup_t = torch.from_numpy(X_unsup).to(device)

        mse = torch.nn.MSELoss()
        bce = torch.nn.BCEWithLogitsLoss()
        pre_optim = self._adam(torch, [self.encoder, recon_head, mask_head])

        mask_prob = cfg["mask_prob"]
        mask_w = cfg["mask_loss_weight"]
        feat_w = cfg["feature_loss_weight"]
        pretrain_epochs_trained = 0
        for _ in range(cfg["pretrain_epochs"]):
            self.encoder.train()
            recon_head.train()
            mask_head.train()
            for batch_idx in iterate_minibatches(
                len(X_unsup_t), cfg["batch_size"], rng, shuffle=True
            ):
                xb = X_unsup_t[batch_idx]
                mask = (torch.rand_like(xb) < mask_prob).float()
                x_tilde = feature_shuffle_corrupt(torch, xb, mask)
                pre_optim.zero_grad()
                emb = self.encoder(x_tilde)
                mask_loss = bce(mask_head(emb), mask)
                feature_loss = mse(recon_head(emb), xb)
                loss = mask_w * mask_loss + feat_w * feature_loss
                loss.backward()
                pre_optim.step()
            pretrain_epochs_trained += 1

        # --- Stage 2: semi-supervised fine-tuning with consistency ----------
        X_tr, y_tr, X_val, y_val, val_strategy = internal_val_split(
            X_lab, y_local, val_fraction=cfg["val_fraction"], seed=self.random_state
        )
        ce = torch.nn.CrossEntropyLoss()
        optim = self._adam(torch, [self.encoder, self.classifier])

        X_tr_t = torch.from_numpy(np.asarray(X_tr, dtype=np.float32)).to(device)
        y_tr_t = torch.from_numpy(np.asarray(y_tr, dtype=np.int64)).to(device)
        X_val_t = (
            torch.from_numpy(np.asarray(X_val, dtype=np.float32)).to(device)
            if X_val is not None
            else None
        )
        y_val_t = (
            torch.from_numpy(np.asarray(y_val, dtype=np.int64)).to(device)
            if y_val is not None
            else None
        )
        consistency_weight = cfg["consistency_weight"]
        X_unlab_t = (
            torch.from_numpy(np.asarray(X_unlab, dtype=np.float32)).to(device)
            if (consistency_weight > 0 and X_unlab.shape[0] > 0)
            else None
        )

        best_loss = math.inf
        best_epoch = -1
        best_state: dict | None = None
        finetune_epochs_trained = 0
        bad_epochs = 0
        final_train_loss = math.nan

        for epoch in range(cfg["max_epochs"]):
            self.encoder.train()
            self.classifier.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch_idx in iterate_minibatches(len(X_tr_t), cfg["batch_size"], rng, shuffle=True):
                xb = X_tr_t[batch_idx]
                yb = y_tr_t[batch_idx]
                optim.zero_grad()
                sup_loss = ce(self.classifier(self.encoder(xb)), yb)
                loss = sup_loss
                if X_unlab_t is not None:
                    u_idx = rng.integers(
                        0, X_unlab_t.shape[0], size=min(len(xb), X_unlab_t.shape[0])
                    )
                    xu = X_unlab_t[torch.from_numpy(u_idx).to(device)]
                    m1 = (torch.rand_like(xu) < mask_prob).float()
                    m2 = (torch.rand_like(xu) < mask_prob).float()
                    p1 = self._proba(torch, feature_shuffle_corrupt(torch, xu, m1))
                    p2 = self._proba(torch, feature_shuffle_corrupt(torch, xu, m2))
                    consistency = torch.mean((p1 - p2) ** 2)
                    loss = loss + consistency_weight * consistency
                loss.backward()
                optim.step()
                epoch_loss += float(sup_loss.detach().cpu())
                n_batches += 1
            finetune_epochs_trained = epoch + 1
            final_train_loss = epoch_loss / max(n_batches, 1)

            if X_val_t is not None:
                self.encoder.eval()
                self.classifier.eval()
                with torch.no_grad():
                    monitor = float(ce(self.classifier(self.encoder(X_val_t)), y_val_t).cpu())
            else:
                monitor = final_train_loss

            if monitor < best_loss - 1e-5:
                best_loss = monitor
                best_epoch = epoch + 1
                best_state = {
                    "encoder": {k: v.detach().clone() for k, v in self.encoder.state_dict().items()},
                    "classifier": {
                        k: v.detach().clone() for k, v in self.classifier.state_dict().items()
                    },
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg["patience"]:
                    break

        if best_state is not None:
            self.encoder.load_state_dict(best_state["encoder"])
            self.classifier.load_state_dict(best_state["classifier"])

        self.training_meta = {
            "neural_method": "vime",
            "method_fidelity": "paper_faithful_core",
            "reference_family": "VIME",
            "best_epoch": best_epoch,
            "epochs_trained": finetune_epochs_trained,
            "pretrain_epochs_trained": pretrain_epochs_trained,
            "finetune_epochs_trained": finetune_epochs_trained,
            "best_val_loss": (
                best_loss if X_val_t is not None and math.isfinite(best_loss) else np.nan
            ),
            "final_train_loss": final_train_loss,
            "neural_validation_strategy": val_strategy,
            "mask_prob": mask_prob,
            "mask_loss_weight": mask_w,
            "feature_loss_weight": feat_w,
            "consistency_weight": consistency_weight,
        }
        return self


class SCARFModel(_NeuralSSLBase):
    """SCARF-style contrastive tabular pretraining + classifier finetune."""

    name = "scarf"
    method = "scarf"

    def __init__(self, random_state: int = 0, n_classes: int = 2, **params: Any) -> None:
        merged = dict(SCARF_DEFAULTS)
        merged.update(params)
        super().__init__(random_state=random_state, n_classes=n_classes, **merged)

    def _nt_xent(self, torch, z1, z2, temperature: float):
        batch = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        z = torch.nn.functional.normalize(z, dim=1)
        sim = torch.matmul(z, z.t()) / temperature
        # Mask self-similarity.
        diag = torch.eye(2 * batch, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(diag, float("-inf"))
        targets = torch.cat(
            [
                torch.arange(batch, 2 * batch, device=z.device),
                torch.arange(0, batch, device=z.device),
            ]
        )
        return torch.nn.functional.cross_entropy(sim, targets)

    def fit(self, X_labeled, y_labeled, X_unlabeled=None) -> "SCARFModel":
        torch = require_torch()
        set_global_determinism(self.random_state)
        cfg = self.cfg
        device = torch.device(cfg["device"])
        rng = np.random.default_rng(self.random_state)

        X_lab, y_local, X_unlab = self._prepare_inputs(X_labeled, y_labeled, X_unlabeled)
        input_dim = X_lab.shape[1]
        n_local = len(self.classes_)

        self.encoder, self.classifier = self._build_encoder_classifier(torch, input_dim, n_local)
        projection = make_mlp(
            torch,
            cfg["embedding_dim"],
            cfg["hidden_dim"],
            cfg["projection_dim"],
            depth=1,
            dropout=cfg["dropout"],
        )

        X_unsup = self._unsupervised_matrix(X_lab, X_unlab)
        X_unsup_t = torch.from_numpy(X_unsup).to(device)

        pre_optim = self._adam(torch, [self.encoder, projection])
        corruption_rate = cfg["corruption_rate"]
        temperature = cfg["temperature"]
        # Prefer batches of at least 128 for a useful contrastive objective.
        contrastive_bs = max(cfg["batch_size"], 128)

        pretrain_epochs_trained = 0
        for _ in range(cfg["pretrain_epochs"]):
            self.encoder.train()
            projection.train()
            for batch_idx in iterate_minibatches(
                len(X_unsup_t), contrastive_bs, rng, shuffle=True
            ):
                if len(batch_idx) < 2:
                    continue
                xb = X_unsup_t[batch_idx]
                m1 = (torch.rand_like(xb) < corruption_rate).float()
                m2 = (torch.rand_like(xb) < corruption_rate).float()
                v1 = feature_shuffle_corrupt(torch, xb, m1)
                v2 = feature_shuffle_corrupt(torch, xb, m2)
                pre_optim.zero_grad()
                z1 = projection(self.encoder(v1))
                z2 = projection(self.encoder(v2))
                loss = self._nt_xent(torch, z1, z2, temperature)
                loss.backward()
                pre_optim.step()
            pretrain_epochs_trained += 1

        finetune_meta = self._train_classifier(torch, X_lab, y_local, X_unlab)

        self.training_meta = {
            "neural_method": "scarf",
            "method_fidelity": "paper_faithful_core",
            "reference_family": "SCARF",
            "best_epoch": finetune_meta["best_epoch"],
            "epochs_trained": finetune_meta["finetune_epochs_trained"],
            "pretrain_epochs_trained": pretrain_epochs_trained,
            "finetune_epochs_trained": finetune_meta["finetune_epochs_trained"],
            "best_val_loss": finetune_meta["best_val_loss"],
            "final_train_loss": finetune_meta["final_train_loss"],
            "neural_validation_strategy": finetune_meta["neural_validation_strategy"],
            "corruption_rate": corruption_rate,
            "temperature": temperature,
        }
        return self


def build_neural_ssl_model(method: str, random_state: int = 0, n_classes: int = 2, **params: Any):
    if method == "sslae":
        return SSLAEModel(random_state=random_state, n_classes=n_classes, **params)
    if method == "vime":
        return VIMEModel(random_state=random_state, n_classes=n_classes, **params)
    if method == "vime_lite":
        return VIMELiteModel(random_state=random_state, n_classes=n_classes, **params)
    if method == "scarf":
        return SCARFModel(random_state=random_state, n_classes=n_classes, **params)
    raise ValueError(f"Unknown neural SSL method: {method}")
