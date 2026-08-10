import os
import torch
import torch.nn as nn


def train_classifier(P, clf, x_sup, y_sup):
    for p in clf.f.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(clf.head.parameters(), lr=P.learning_rate)
    criterion = nn.CrossEntropyLoss()

    save_dir = os.path.join(P.checkpoint_dir, f"{P.dataset}_{P.masked_ratio}_{P.index}")

    os.makedirs(save_dir, exist_ok=True)
    best_ckpt = os.path.join(save_dir, "classifier_best.pth")

    best_train_acc = 0.0
    no_improve = 0
    iters = 0

    while iters < P.trainer_epochs and no_improve < P.patience:
        iters += 1

        optimizer.zero_grad()
        logits = clf(x_sup)
        loss = criterion(logits, y_sup)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(1)
        train_acc = (preds == y_sup).float().mean().item()

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(clf.state_dict(), best_ckpt)
            no_improve = 0
        else:
            no_improve += 1

        if best_train_acc == 1.0:
            break

    clf.load_state_dict(torch.load(best_ckpt, map_location=P.device))
    return clf
