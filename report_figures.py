import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.ndimage import gaussian_filter
import torch

from sim.fields import RandomGaussianMixtureField
from sim.raytrace import make_entrance_rays, trace_rays_semi_implicit
from models.dynamics_mlp import DynamicsMLP


OUTDIR = "outputs/report_figures_v2"
os.makedirs(OUTDIR, exist_ok=True)

FIELD_KWARGS = dict(
    num_blobs=12,
    beta=0.16,
    sinusoid_amplitude=0.06,
    num_sinusoids=3,
    z_mod_amplitude=0.25,
    z_mod_k=3.0,
)

SEED = 496
RAYS_NX = 192
RAYS_NY = 192
NUM_STEPS = 64
ZMAX = 1.0
JITTER_STD = 0.005

CAUSTIC_BINS = 320
CAUSTIC_XLIM = (-1.05, 1.05)
CAUSTIC_YLIM = (-1.05, 1.05)

CHECKPOINT_PATH = "checkpoints/dynamics_mlp_best.pt"

def render_caustic_histogram(trace_result, bins=256, xlim=None, ylim=None, smooth_sigma=0.6):
    x_final = trace_result["x"][-1].ravel()
    y_final = trace_result["y"][-1].ravel()

    if xlim is None:
        xlim = (x_final.min(), x_final.max())
    if ylim is None:
        ylim = (y_final.min(), y_final.max())

    hist, xedges, yedges = np.histogram2d(
        x_final,
        y_final,
        bins=bins,
        range=[xlim, ylim],
    )
    hist = hist.T.astype(np.float64)

    if smooth_sigma is not None and smooth_sigma > 0:
        hist = gaussian_filter(hist, sigma=smooth_sigma)

    return hist, xedges, yedges


def choose_ray_indices(ny, nx, num_rays=36):
    gx = max(1, int(np.sqrt(num_rays)))
    gy = max(1, int(np.ceil(num_rays / gx)))
    xs = np.linspace(0, nx - 1, gx).astype(int)
    ys = np.linspace(0, ny - 1, gy).astype(int)
    ids = [(iy, ix) for iy in ys for ix in xs]
    return ids[:num_rays]


def sample_density_slice(field, nx=220, ny=220, z0=0.5, xlim=(-1, 1), ylim=(-1, 1)):
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = np.full_like(X, z0)

    rho = field.density(X, Y, Z)
    gx, gy = field.transverse_grad_refractive_index(X, Y, Z)
    gmag = np.sqrt(gx**2 + gy**2)

    return x, y, rho, gmag


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

    return {
        "x": x_hist,
        "y": y_hist,
        "vx": vx_hist,
        "vy": vy_hist,
        "z": z_hist,
        "dz": dz,
    }

def save_figure1(field, trace_result, outpath):
    x, y, rho, gmag = sample_density_slice(field, z0=0.5)

    fig = plt.figure(figsize=(18, 4.5))

    ax1 = fig.add_subplot(1, 4, 1)
    vmax = np.max(np.abs(rho))
    im1 = ax1.imshow(
        rho,
        extent=[x[0], x[-1], y[0], y[-1]],
        origin="lower",
        aspect="equal",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
    )
    ax1.set_title(r"Signed density slice $\rho(x,y,z_0)$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

    ax2 = fig.add_subplot(1, 4, 2)
    im2 = ax2.imshow(
        gmag,
        extent=[x[0], x[-1], y[0], y[-1]],
        origin="lower",
        aspect="equal",
        cmap="viridis",
    )
    ax2.set_title(r"Transverse gradient magnitude $\|\nabla_\perp n\|$")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)

    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    ids = choose_ray_indices(trace_result["x"].shape[1], trace_result["x"].shape[2], num_rays=36)
    for iy, ix in ids:
        ax3.plot(
            trace_result["x"][:, iy, ix],
            trace_result["y"][:, iy, ix],
            trace_result["z"],
            linewidth=1.0,
            alpha=0.85,
        )
    ax3.set_title("Representative 3D ray trajectories")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.view_init(elev=22, azim=38)

    ax4 = fig.add_subplot(1, 4, 4)
    x_final = trace_result["x"][-1].ravel()
    y_final = trace_result["y"][-1].ravel()
    ax4.scatter(x_final, y_final, s=1, alpha=0.45)
    ax4.set_title("Final ray landing positions")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_figure2_physics_only(trace_result, outpath):
    hist, xedges, yedges = render_caustic_histogram(
        trace_result,
        bins=CAUSTIC_BINS,
        xlim=CAUSTIC_XLIM,
        ylim=CAUSTIC_YLIM,
        smooth_sigma=0.6,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    im0 = axes[0].imshow(
        hist,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin="lower",
        aspect="equal",
        cmap="inferno",
    )
    axes[0].set_title("Caustic histogram")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)

    im1 = axes[1].imshow(
        hist + 1e-9,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin="lower",
        aspect="equal",
        cmap="inferno",
        vmin=np.percentile(hist, 1),
        vmax=np.percentile(hist, 99), 
    )
    axes[1].set_title("Caustic histogram (log scale)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_figure2_with_learned(physics_result, learned_result, outpath):
    hist_p, xedges, yedges = render_caustic_histogram(
        physics_result,
        bins=CAUSTIC_BINS,
        xlim=CAUSTIC_XLIM,
        ylim=CAUSTIC_YLIM,
        smooth_sigma=0.6,
    )
    hist_l, _, _ = render_caustic_histogram(
        learned_result,
        bins=CAUSTIC_BINS,
        xlim=CAUSTIC_XLIM,
        ylim=CAUSTIC_YLIM,
        smooth_sigma=0.6,
    )

    hist_pn = hist_p / (hist_p.max() + 1e-9)
    hist_ln = hist_l / (hist_l.max() + 1e-9)
    err = np.abs(hist_pn - hist_ln)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    im0 = axes[0].imshow(
        hist_pn + 1e-9,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin="lower",
        aspect="equal",
        cmap="gray",
        vmin=np.percentile(hist_pn, 1),
        vmax=np.percentile(hist_pn, 99),
    )
    axes[0].set_title("Physics caustic")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)

    im1 = axes[1].imshow(
        hist_ln + 1e-9,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin="lower",
        aspect="equal",
        cmap="gray",
        vmin=np.percentile(hist_ln, 1),
        vmax=np.percentile(hist_ln, 99),
    )
    axes[1].set_title("Learned caustic")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)

    im2 = axes[2].imshow(
        err,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin="lower",
        aspect="equal",
        cmap="magma",
    )
    axes[2].set_title(r"Absolute error $|C_{\mathrm{phys}}-C_{\mathrm{learned}}|$")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Generating representative field and physics rollout...")

    field = RandomGaussianMixtureField(seed=SEED, **FIELD_KWARGS)

    x0, y0, vx0, vy0 = make_entrance_rays(
        nx=RAYS_NX,
        ny=RAYS_NY,
        xlim=(-1, 1),
        ylim=(-1, 1),
        vx0=0.0,
        vy0=0.0,
    )

    rng = np.random.default_rng(123)
    x0 = x0 + rng.normal(scale=JITTER_STD, size=x0.shape)
    y0 = y0 + rng.normal(scale=JITTER_STD, size=y0.shape)

    physics_result = trace_rays_semi_implicit(
        field=field,
        x0=x0,
        y0=y0,
        vx0=vx0,
        vy0=vy0,
        zmax=ZMAX,
        num_steps=NUM_STEPS,
        bend_scale=1.0,
        return_acceleration=False,
    )

    save_figure1(
        field,
        physics_result,
        os.path.join(OUTDIR, "figure1_field_rays_positions.png"),
    )

    if os.path.exists(CHECKPOINT_PATH):
        print("Checkpoint found. Generating learned-vs-physics comparison...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

        in_dim = int(ckpt["normalization"]["x_mean"].shape[1])
        model = DynamicsMLP(in_dim=in_dim, out_dim=2, hidden_dim=64, num_hidden_layers=3).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        learned_result = learned_trace_rays_semi_implicit(
            model=model,
            norm=ckpt["normalization"],
            field=field,
            x0=x0,
            y0=y0,
            vx0=vx0,
            vy0=vy0,
            zmax=ZMAX,
            num_steps=NUM_STEPS,
            device=device,
        )

        save_figure2_with_learned(
            physics_result,
            learned_result,
            os.path.join(OUTDIR, "figure2_learned_vs_physics_caustics.png"),
        )
    else:
        print("No checkpoint found. Saving physics-only caustic figure...")
        save_figure2_physics_only(
            physics_result,
            os.path.join(OUTDIR, "figure2_physics_caustics.png"),
        )

    print(f"Saved figures to: {OUTDIR}")


if __name__ == "__main__":
    main()