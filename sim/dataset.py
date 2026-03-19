import numpy as np
import torch
from torch.utils.data import Dataset

from sim.raytrace import make_entrance_rays, trace_rays_semi_implicit
from sim.fields import RandomGaussianMixtureField


def build_training_arrays(
    num_fields=20,
    rays_nx=64,
    rays_ny=64,
    num_steps=64,
    zmax=1.0,
    field_kwargs=None,
    seeds=None,
    jitter_std=0.005,
):
    if field_kwargs is None:
        field_kwargs = {}

    if seeds is None:
        seeds = list(range(num_fields))
    else:
        num_fields = len(seeds)

    inputs_all = []
    targets_all = []

    for seed in seeds:
        field = RandomGaussianMixtureField(seed=seed, **field_kwargs)

        x0, y0, vx0, vy0 = make_entrance_rays(
            nx=rays_nx,
            ny=rays_ny,
            xlim=(-1, 1),
            ylim=(-1, 1),
            vx0=0.0,
            vy0=0.0,
        )

        if jitter_std > 0:
            x0 = x0 + np.random.normal(scale=jitter_std, size=x0.shape)
            y0 = y0 + np.random.normal(scale=jitter_std, size=y0.shape)

        result = trace_rays_semi_implicit(
            field=field,
            x0=x0,
            y0=y0,
            vx0=vx0,
            vy0=vy0,
            zmax=zmax,
            num_steps=num_steps,
            bend_scale=1.0,
            return_acceleration=True,
        )

        x = result["x"][:-1]
        y = result["y"][:-1]
        vx = result["vx"][:-1]
        vy = result["vy"][:-1]
        ax = result["ax"]
        ay = result["ay"]
        z = result["z"][:-1]

        z_grid = np.broadcast_to(z[:, None, None], x.shape)

        n = field.refractive_index(x, y, z_grid)
        gx, gy = field.transverse_grad_refractive_index(x, y, z_grid)

        inputs = np.stack([x, y, z_grid, vx, vy, n, gx, gy], axis=-1)
        targets = np.stack([ax, ay], axis=-1)

        inputs_all.append(inputs.reshape(-1, 8))
        targets_all.append(targets.reshape(-1, 2))

    X = np.concatenate(inputs_all, axis=0).astype(np.float32)
    Y = np.concatenate(targets_all, axis=0).astype(np.float32)

    return X, Y


class RayAccelerationDataset(Dataset):
    def __init__(self, X, Y, normalize=True):
        self.X = np.asarray(X, dtype=np.float32)
        self.Y = np.asarray(Y, dtype=np.float32)

        self.normalize = normalize

        if normalize:
            self.x_mean = self.X.mean(axis=0, keepdims=True)
            self.x_std = self.X.std(axis=0, keepdims=True) + 1e-8
            self.y_mean = self.Y.mean(axis=0, keepdims=True)
            self.y_std = self.Y.std(axis=0, keepdims=True) + 1e-8
        else:
            self.x_mean = np.zeros((1, self.X.shape[1]), dtype=np.float32)
            self.x_std = np.ones((1, self.X.shape[1]), dtype=np.float32)
            self.y_mean = np.zeros((1, self.Y.shape[1]), dtype=np.float32)
            self.y_std = np.ones((1, self.Y.shape[1]), dtype=np.float32)

        self.Xn = (self.X - self.x_mean) / self.x_std
        self.Yn = (self.Y - self.y_mean) / self.y_std

    def __len__(self):
        return len(self.Xn)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.Xn[idx])
        y = torch.from_numpy(self.Yn[idx])
        return x, y

    def get_normalization_tensors(self, device="cpu"):
        return {
            "x_mean": torch.tensor(self.x_mean, dtype=torch.float32, device=device),
            "x_std": torch.tensor(self.x_std, dtype=torch.float32, device=device),
            "y_mean": torch.tensor(self.y_mean, dtype=torch.float32, device=device),
            "y_std": torch.tensor(self.y_std, dtype=torch.float32, device=device),
        }