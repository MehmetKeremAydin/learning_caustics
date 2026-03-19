import matplotlib.pyplot as plt
import numpy as np
import torch

from sim.dataset import build_training_arrays, RayAccelerationDataset
from models.dynamics_mlp import DynamicsMLP


ckpt = torch.load("checkpoints/dynamics_mlp_best.pt", map_location="cpu")

model = DynamicsMLP(in_dim=5, out_dim=2, hidden_dim=64, num_hidden_layers=3)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

norm = ckpt["normalization"]
x_mean = norm["x_mean"]
x_std = norm["x_std"]
y_mean = norm["y_mean"]
y_std = norm["y_std"]

field_kwargs = ckpt["field_kwargs"]

X, Y = build_training_arrays(
    num_fields=1,
    rays_nx=64,
    rays_ny=64,
    num_steps=64,
    zmax=1.0,
    field_kwargs=field_kwargs,
    seeds=[999],
    jitter_std=0.005,
)

idx = np.random.choice(len(X), size=5000, replace=False)
X_sub = torch.tensor(X[idx], dtype=torch.float32)
Y_sub = torch.tensor(Y[idx], dtype=torch.float32)

Xn = (X_sub - x_mean) / x_std

with torch.no_grad():
    pred_n = model(Xn)
    pred = pred_n * y_std + y_mean

pred = pred.numpy()
Y_sub = Y_sub.numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].scatter(Y_sub[:, 0], pred[:, 0], s=2, alpha=0.4)
axes[0].set_title("ax: target vs pred")
axes[0].set_xlabel("target")
axes[0].set_ylabel("pred")

axes[1].scatter(Y_sub[:, 1], pred[:, 1], s=2, alpha=0.4)
axes[1].set_title("ay: target vs pred")
axes[1].set_xlabel("target")
axes[1].set_ylabel("pred")

plt.tight_layout()
plt.show()