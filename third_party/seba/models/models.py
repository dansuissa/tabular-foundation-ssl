import torch
import torch.nn as nn
import torch.nn.functional as F


class FNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
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
    def __init__(self, hidden_dim, input_dim, embed_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z, mask):
        zp = torch.cat([z, mask], dim=-1)
        h = self.layers(zp)
        return F.normalize(h, p=2, dim=1)


class Classifier(nn.Module):
    def __init__(self, f: nn.Module, hidden_dim: int, num_classes: int):
        super().__init__()
        self.f = f
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        z = self.f(x)
        return self.head(z)
