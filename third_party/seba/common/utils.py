import random
import numpy as np
import torch
import os

from models import FNetwork, Classifier
from trainers import train_classifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_f_encoder(P, masked_ratio):
    ckpt = os.path.join(
        P.checkpoint_dir, f"{P.dataset}_{masked_ratio}_{P.index}", "pretrainer_best.pth"
    )
    state = torch.load(ckpt, map_location=P.device)
    f = FNetwork(P.input_dim, P.hidden_dim).to(P.device)
    f.load_state_dict(state["f_state"])

    f.train()
    return f


def train_classifier_ensemble_for_fs(P, fs, ratios, x_sup, y_sup):
    clfs = []

    for f, r in zip(fs, ratios):
        clf = Classifier(f, P.hidden_dim, P.num_classes).to(P.device)
        clf = train_classifier(P, clf, x_sup, y_sup)
        clfs.append(clf)

    return clfs
