import torch
import torch.nn as nn


class DynamicsMLP(nn.Module):
    def __init__(self, in_dim=8, out_dim=2, hidden_dim=64, num_hidden_layers=3):
        super().__init__()

        layers = []
        dim = in_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.Tanh())
            dim = hidden_dim

        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)