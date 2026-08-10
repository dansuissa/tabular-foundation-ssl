import os
import torch
import torch.nn.functional as F


def generate_mask(X, masked_ratio, feature_groups, fill_mode="zero"):
    num_original = len(feature_groups)
    k = int(masked_ratio * num_original)
    group_names = list(feature_groups.keys())
    chosen_group_idxs = torch.randperm(num_original)[:k]

    X_masked = X.clone()
    mask = torch.ones_like(X)

    for g_idx in chosen_group_idxs:
        cols = feature_groups[group_names[g_idx]]
        if fill_mode == "marginal":
            perm = torch.randperm(X.shape[0], device=X.device)
            X_masked[:, cols] = X[perm][:, cols]
        elif fill_mode == "zero":
            X_masked[:, cols] = 0.0
        mask[:, cols] = 0.0

    return X_masked, mask


def compute_nearest_neighbor_indices(orig, k=1):
    dists = torch.cdist(orig, orig)
    dists.fill_diagonal_(float("inf"))
    knn_indices = dists.topk(k, largest=False).indices
    rand_choice = torch.randint(0, k, (orig.size(0),), device=orig.device)
    nn_idx = knn_indices[torch.arange(orig.size(0)), rand_choice]
    return nn_idx


def compute_contrastive_loss(h, nn_idx, temperature):
    sim = h @ h.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    return F.cross_entropy(sim, nn_idx)


def validate_pretrainer(P, f_model, p_model, val_loader, feature_groups=None):
    f_model.eval()
    p_model.eval()
    total_loss = total_cnt = 0
    with torch.no_grad():
        for X, _ in val_loader:
            X = X.to(P.device)
            X_masked, mask = generate_mask(
                X, P.masked_ratio, feature_groups, P.fill_mode
            )
            Z = f_model(X_masked)
            H = p_model(Z, mask)
            orig = X * (1 - mask)
            nn_idx = compute_nearest_neighbor_indices(orig)
            loss = compute_contrastive_loss(H, nn_idx, P.temperature)
            total_loss += loss.item() * X.size(0)
            total_cnt += X.size(0)
    f_model.train()
    p_model.train()
    return total_loss / total_cnt


def pretrainer(
    P, f_model, p_model, optimizer, train_loader, val_loader, feature_groups
):
    best_val_loss = float("inf")
    no_improve = 0

    os.makedirs(
        f"{P.checkpoint_dir}/{P.dataset}_{P.masked_ratio}_{P.index}", exist_ok=True
    )
    best_ckpt = os.path.join(
        P.checkpoint_dir,
        f"{P.dataset}_{P.masked_ratio}_{P.index}",
        "pretrainer_best.pth",
    )

    for epoch in range(1, P.pretrainer_epochs + 1):
        for X, _ in train_loader:
            X = X.to(P.device)
            X_masked, mask = generate_mask(
                X, P.masked_ratio, feature_groups, P.fill_mode
            )
            Z = f_model(X_masked)
            H = p_model(Z, mask)

            with torch.no_grad():
                orig = X * (1 - mask)
                nn_idx = compute_nearest_neighbor_indices(orig)
            loss = compute_contrastive_loss(H, nn_idx, P.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = validate_pretrainer(P, f_model, p_model, val_loader, feature_groups)
        if epoch % P.log_interval == 0:
            print(
                f"[Pretrain] Epoch {epoch:02d}/{P.pretrainer_epochs}, val_loss={val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(
                {
                    "f_state": f_model.state_dict(),
                    "p_state": p_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "epoch": epoch,
                },
                best_ckpt,
            )
            print(
                f"---> New best pretrainer checkpoint saved (loss={best_val_loss:.4f})"
            )
        else:
            no_improve += 1
            if no_improve >= P.patience:
                print(
                    f"[Pretrain] Early stopping after {epoch} epochs (no improvement)."
                )
                break

    ckpt = torch.load(best_ckpt, map_location=P.device)
    f_model.load_state_dict(ckpt["f_state"])
    p_model.load_state_dict(ckpt["p_state"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(
        f"Best pretrainer (epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}) reloaded from {best_ckpt}"
    )
