import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from sim.dataset import build_training_arrays, RayAccelerationDataset
from models.dynamics_mlp import DynamicsMLP


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    field_kwargs = dict(
        num_blobs=12,
        beta=0.16,
        sinusoid_amplitude=0.06,
        num_sinusoids=3,
        z_mod_amplitude=0.25,
        z_mod_k=3.0,
    )

    X, Y = build_training_arrays(
        num_fields=20,
        rays_nx=64,
        rays_ny=64,
        num_steps=64,
        zmax=1.0,
        field_kwargs=field_kwargs,
        seeds=list(range(20)),
        jitter_std=0.005,
    )

    print("Dataset shape:", X.shape, Y.shape)

    dataset = RayAccelerationDataset(X, Y, normalize=True)

    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )

    train_loader = DataLoader(train_set, batch_size=4096, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=4096, shuffle=False, num_workers=0)

    model = DynamicsMLP(in_dim=8, out_dim=2, hidden_dim=64, num_hidden_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val = float("inf")
    history = {"train": [], "val": []}

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, 11):
        model.train()
        train_loss = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False, dynamic_ncols=True)
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            train_loss += loss.item() * bs
            train_count += bs

            pbar.set_postfix(loss=f"{loss.item():.3e}")

        train_loss /= train_count

        model.eval()
        val_loss = 0.0
        val_count = 0

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch:03d} [val]", leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for xb, yb in pbar_val:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = criterion(pred, yb)

                bs = xb.shape[0]
                val_loss += loss.item() * bs
                val_count += bs

                pbar_val.set_postfix(loss=f"{loss.item():.3e}")

        val_loss /= val_count

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(f"Epoch {epoch:03d} | train={train_loss:.6e} | val={val_loss:.6e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "field_kwargs": field_kwargs,
                    "normalization": dataset.get_normalization_tensors(device="cpu"),
                },
                "checkpoints/dynamics_mlp_best.pt",
            )

    np.savez("checkpoints/dynamics_history.npz", **history)


if __name__ == "__main__":
    main()