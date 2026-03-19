import os
import numpy as np
import torch

from sim.fields import RandomGaussianMixtureField
from sim.raytrace import make_entrance_rays, trace_rays_semi_implicit
from sim.visualization import (
    plot_ray_trajectories_xz,
    plot_ray_trajectories_yz,
    plot_final_ray_positions,
    plot_caustic,
    plot_caustic_log,
)
from models.dynamics_mlp import DynamicsMLP


def learned_trace_rays_semi_implicit(
    model,
    norm,
    field,
    x0,
    y0,
    vx0,
    vy0,
    zmax=1.0,
    num_steps=64,
    device="cpu",
):
    x0 = np.asarray(x0, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    vx0 = np.asarray(vx0, dtype=np.float64)
    vy0 = np.asarray(vy0, dtype=np.float64)

    ny, nx = x0.shape
    dz = zmax / num_steps
    z_hist = np.linspace(0.0, zmax, num_steps + 1, dtype=np.float64)

    x_hist = np.zeros((num_steps + 1, ny, nx), dtype=np.float64)
    y_hist = np.zeros((num_steps + 1, ny, nx), dtype=np.float64)
    vx_hist = np.zeros((num_steps + 1, ny, nx), dtype=np.float64)
    vy_hist = np.zeros((num_steps + 1, ny, nx), dtype=np.float64)
    ax_hist = np.zeros((num_steps, ny, nx), dtype=np.float64)
    ay_hist = np.zeros((num_steps, ny, nx), dtype=np.float64)

    x_hist[0] = x0
    y_hist[0] = y0
    vx_hist[0] = vx0
    vy_hist[0] = vy0

    x_mean = norm["x_mean"].cpu().numpy()
    x_std = norm["x_std"].cpu().numpy()
    y_mean = norm["y_mean"].cpu().numpy()
    y_std = norm["y_std"].cpu().numpy()

    model.eval()

    with torch.no_grad():
        for k in range(num_steps):
            zk = z_hist[k]

            xk = x_hist[k]
            yk = y_hist[k]
            vxk = vx_hist[k]
            vyk = vy_hist[k]

            z_grid = np.full_like(xk, zk)

            n = field.refractive_index(xk, yk, z_grid)
            gx, gy = field.transverse_grad_refractive_index(xk, yk, z_grid)

            inp = np.stack([xk, yk, z_grid, vxk, vyk, n, gx, gy], axis=-1)
            inp = inp.reshape(-1, 8).astype(np.float32)

            inp_n = (inp - x_mean) / x_std

            inp_t = torch.tensor(inp_n, dtype=torch.float32, device=device)
            pred_n = model(inp_t).cpu().numpy()
            pred = pred_n * y_std + y_mean

            axk = pred[:, 0].reshape(ny, nx)
            ayk = pred[:, 1].reshape(ny, nx)

            vx_next = vxk + dz * axk
            vy_next = vyk + dz * ayk
            x_next = xk + dz * vx_next
            y_next = yk + dz * vy_next

            x_hist[k + 1] = x_next
            y_hist[k + 1] = y_next
            vx_hist[k + 1] = vx_next
            vy_hist[k + 1] = vy_next
            ax_hist[k] = axk
            ay_hist[k] = ayk

    return {
        "x": x_hist,
        "y": y_hist,
        "vx": vx_hist,
        "vy": vy_hist,
        "ax": ax_hist,
        "ay": ay_hist,
        "z": z_hist,
        "dz": dz,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    ckpt = torch.load("checkpoints/dynamics_mlp_best.pt", map_location=device)

    model = DynamicsMLP(in_dim=8, out_dim=2, hidden_dim=64, num_hidden_layers=3).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    norm = ckpt["normalization"]
    field_kwargs = ckpt["field_kwargs"]

    seed = 999
    field = RandomGaussianMixtureField(seed=seed, **field_kwargs)

    x0, y0, vx0, vy0 = make_entrance_rays(
        nx=128,
        ny=128,
        xlim=(-1, 1),
        ylim=(-1, 1),
        vx0=0.0,
        vy0=0.0,
    )

    x0 = x0 + np.random.normal(scale=0.005, size=x0.shape)
    y0 = y0 + np.random.normal(scale=0.005, size=y0.shape)

    physics_result = trace_rays_semi_implicit(
        field=field,
        x0=x0,
        y0=y0,
        vx0=vx0,
        vy0=vy0,
        zmax=1.0,
        num_steps=64,
        bend_scale=1.0,
        return_acceleration=True,
    )

    learned_result = learned_trace_rays_semi_implicit(
        model=model,
        norm=norm,
        field=field,
        x0=x0,
        y0=y0,
        vx0=vx0,
        vy0=vy0,
        zmax=1.0,
        num_steps=64,
        device=device,
    )

    out_dir = f"outputs/rollout_eval_seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)

    plot_ray_trajectories_xz(
        physics_result,
        num_rays=25,
        show=False,
        savepath=os.path.join(out_dir, "physics_xz.png"),
    )
    plot_ray_trajectories_yz(
        physics_result,
        num_rays=25,
        show=False,
        savepath=os.path.join(out_dir, "physics_yz.png"),
    )
    plot_final_ray_positions(
        physics_result,
        show=False,
        savepath=os.path.join(out_dir, "physics_final.png"),
    )
    plot_caustic(
        physics_result,
        bins=256,
        show=False,
        savepath=os.path.join(out_dir, "physics_caustic.png"),
    )
    plot_caustic_log(
        physics_result,
        bins=256,
        show=False,
        savepath=os.path.join(out_dir, "physics_caustic_log.png"),
    )

    plot_ray_trajectories_xz(
        learned_result,
        num_rays=25,
        show=False,
        savepath=os.path.join(out_dir, "learned_xz.png"),
    )
    plot_ray_trajectories_yz(
        learned_result,
        num_rays=25,
        show=False,
        savepath=os.path.join(out_dir, "learned_yz.png"),
    )
    plot_final_ray_positions(
        learned_result,
        show=False,
        savepath=os.path.join(out_dir, "learned_final.png"),
    )
    plot_caustic(
        learned_result,
        bins=256,
        show=False,
        savepath=os.path.join(out_dir, "learned_caustic.png"),
    )
    plot_caustic_log(
        learned_result,
        bins=256,
        show=False,
        savepath=os.path.join(out_dir, "learned_caustic_log.png"),
    )

    print(f"Saved rollout comparison plots to: {out_dir}")


if __name__ == "__main__":
    main()