import numpy as np
import matplotlib.pyplot as plt


def sample_field_on_grid(field, nx=64, ny=64, nz=64,
                         xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 1)):
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    z = np.linspace(zlim[0], zlim[1], nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")
    rho = field.density(X, Y, Z)
    rho = np.transpose(rho, (2, 0, 1))

    return x, y, z, rho


def plot_density_slices(field, nx=96, ny=96, nz=96,
                        xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 1),
                        slice_x=None, slice_y=None, slice_z=None,
                        cmap="inferno", figsize=(12, 4),
                        show=True, savepath=None):
    x, y, z, rho = sample_field_on_grid(
        field, nx=nx, ny=ny, nz=nz, xlim=xlim, ylim=ylim, zlim=zlim
    )

    ix = nx // 2 if slice_x is None else np.argmin(np.abs(x - slice_x))
    iy = ny // 2 if slice_y is None else np.argmin(np.abs(y - slice_y))
    iz = nz // 2 if slice_z is None else np.argmin(np.abs(z - slice_z))

    xy = rho[iz, :, :]
    xz = rho[:, iy, :]
    yz = rho[:, :, ix]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    im0 = axes[0].imshow(
        xy,
        extent=[x[0], x[-1], y[0], y[-1]],
        origin="lower",
        cmap=cmap,
        aspect="auto"
    )
    axes[0].set_title(f"XY slice at z={z[iz]:.3f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        xz,
        extent=[x[0], x[-1], z[0], z[-1]],
        origin="lower",
        cmap=cmap,
        aspect="auto"
    )
    axes[1].set_title(f"XZ slice at y={y[iy]:.3f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(
        yz,
        extent=[y[0], y[-1], z[0], z[-1]],
        origin="lower",
        cmap=cmap,
        aspect="auto"
    )
    axes[2].set_title(f"YZ slice at x={x[ix]:.3f}")
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("z")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Density Field Slices")
    fig.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes


def plot_density_max_projections(field, nx=96, ny=96, nz=96,
                                 xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 1),
                                 cmap="inferno", figsize=(12, 4),
                                 show=True, savepath=None):
    x, y, z, rho = sample_field_on_grid(
        field, nx=nx, ny=ny, nz=nz, xlim=xlim, ylim=ylim, zlim=zlim
    )

    xy = np.max(rho, axis=0)
    xz = np.max(rho, axis=1)
    yz = np.max(rho, axis=2)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    im0 = axes[0].imshow(
        xy,
        extent=[x[0], x[-1], y[0], y[-1]],
        origin="lower",
        cmap=cmap,
        aspect="auto"
    )
    axes[0].set_title("Max Projection XY")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        xz,
        extent=[x[0], x[-1], z[0], z[-1]],
        origin="lower",
        cmap=cmap,
        aspect="auto"
    )
    axes[1].set_title("Max Projection XZ")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(
        yz,
        extent=[y[0], y[-1], z[0], z[-1]],
        origin="lower",
        cmap=cmap,
        aspect="auto"
    )
    axes[2].set_title("Max Projection YZ")
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("z")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Density Field Max Projections")
    fig.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes


def plot_density_point_cloud(field, nx=48, ny=48, nz=48,
                             xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 1),
                             threshold_ratio=0.2, max_points=5000,
                             figsize=(7, 6), elev=20, azim=35,
                             show=True, savepath=None):
    x, y, z, rho = sample_field_on_grid(
        field, nx=nx, ny=ny, nz=nz, xlim=xlim, ylim=ylim, zlim=zlim
    )

    abs_rho = np.abs(rho)
    threshold = 0.35 * np.max(abs_rho)

    mask_pos = rho >= threshold
    mask_neg = rho <= -threshold

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    zz, yy, xx = np.where(mask_pos)
    vals = rho[zz, yy, xx]
    if len(xx) > 3500:
        idx = np.linspace(0, len(xx) - 1, 3500).astype(int)
        xx, yy, zz, vals = xx[idx], yy[idx], zz[idx], vals[idx]
    ax.scatter(x[xx], y[yy], z[zz], c=vals, s=4, alpha=0.5, cmap="inferno")

    zz, yy, xx = np.where(mask_neg)
    vals = -rho[zz, yy, xx]
    if len(xx) > 3500:
        idx = np.linspace(0, len(xx) - 1, 3500).astype(int)
        xx, yy, zz, vals = xx[idx], yy[idx], zz[idx], vals[idx]
    ax.scatter(x[xx], y[yy], z[zz], c=vals, s=4, alpha=0.35, cmap="Blues")

    ax.set_title("Signed 3D density support")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=36)
    plt.tight_layout()
    plt.savefig("signed_density_3d.png", dpi=300, bbox_inches="tight")
    plt.close()
    return fig, ax


def plot_transverse_gradient_magnitude(field, nx=96, ny=96, nz=96,
                                       xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 1),
                                       slice_z=None, cmap="viridis",
                                       figsize=(5, 4), show=True, savepath=None):
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    z = np.linspace(zlim[0], zlim[1], nz)

    iz = nz // 2 if slice_z is None else np.argmin(np.abs(z - slice_z))
    z0 = z[iz]

    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = np.full_like(X, z0)

    gx, gy, _ = field.grad_refractive_index(X, Y, Z)
    mag = np.sqrt(gx**2 + gy**2)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        mag,
        extent=[x[0], x[-1], y[0], y[-1]],
        origin="lower",
        cmap=cmap,
        aspect="auto"
    )
    ax.set_title(f"|∇⊥ n| at z={z0:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_final_ray_positions(trace_result, figsize=(5, 5), s=2,
                             alpha=0.7, show=True, savepath=None,
                             num_rays=None, mode="grid", seed=None):
    x_final = trace_result["x"][-1]
    y_final = trace_result["y"][-1]

    ny, nx = x_final.shape
    if num_rays is None:
        xs = x_final.ravel()
        ys = y_final.ravel()
    else:
        num_rays = int(num_rays)
        num_rays = max(0, min(num_rays, ny * nx))

        if num_rays == 0:
            xs = np.array([])
            ys = np.array([])
        elif mode == "random":
            rng = np.random.default_rng(seed)
            total = ny * nx
            flat_ids = rng.choice(total, size=num_rays, replace=False)
            xs = x_final.ravel()[flat_ids]
            ys = y_final.ravel()[flat_ids]
        elif mode == "grid":
            gx = max(1, int(np.sqrt(num_rays)))
            gy = max(1, int(np.ceil(num_rays / gx)))
            xs_grid = np.linspace(0, nx - 1, gx).astype(int)
            ys_grid = np.linspace(0, ny - 1, gy).astype(int)
            picked = [(iy, ix) for iy in ys_grid for ix in xs_grid]
            picked = picked[:num_rays]
            ix = np.array([p[1] for p in picked], dtype=int)
            iy = np.array([p[0] for p in picked], dtype=int)
            xs = x_final[iy, ix]
            ys = y_final[iy, ix]
        else:
            raise ValueError("mode must be 'random' or 'grid'")

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xs, ys, s=s, alpha=alpha)
    ax.set_title("Final Ray Landing Positions")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")

    fig.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def render_caustic_histogram(trace_result, bins=128,
                             xlim=None, ylim=None,
                             normalize=True):
    x_final = trace_result["x"][-1].ravel()
    y_final = trace_result["y"][-1].ravel()

    if xlim is None:
        x_margin = 0.05 * (x_final.max() - x_final.min() + 1e-12)
        xlim = (x_final.min() - x_margin, x_final.max() + x_margin)

    if ylim is None:
        y_margin = 0.05 * (y_final.max() - y_final.min() + 1e-12)
        ylim = (y_final.min() - y_margin, y_final.max() + y_margin)

    hist, xedges, yedges = np.histogram2d(
        x_final, y_final,
        bins=bins,
        range=[xlim, ylim]
    )

    hist = hist.T

    if normalize and hist.max() > 0:
        hist = hist / hist.max()

    return hist, xedges, yedges


def plot_caustic(trace_result, bins=128, xlim=None, ylim=None,
                 normalize=True, cmap="inferno",
                 figsize=(6, 5), show=True, savepath=None):
    hist, xedges, yedges = render_caustic_histogram(
        trace_result,
        bins=bins,
        xlim=xlim,
        ylim=ylim,
        normalize=normalize,
    )

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        hist,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin="lower",
        cmap=cmap,
        aspect="equal",
    )
    ax.set_title("Caustic Image")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_caustic_log(trace_result, bins=128, xlim=None, ylim=None,
                     cmap="inferno", figsize=(6, 5),
                     show=True, savepath=None,
                     log_transform="log1p",
                     quantile_low=0.01, quantile_high=0.99,
                     eps=1e-12):
    hist, xedges, yedges = render_caustic_histogram(
        trace_result,
        bins=bins,
        xlim=xlim,
        ylim=ylim,
        normalize=False,
    )

    hist = np.maximum(hist, 0.0)

    if log_transform == "log1p":
        log_hist = np.log1p(hist + eps)
    elif log_transform == "log":
        log_hist = np.log(hist + eps)
    else:
        raise ValueError("log_transform must be one of {'log1p', 'log'}")

    flat = log_hist.reshape(-1)
    nz = flat[flat > 0.0]
    if nz.size > 0:
        vmin = float(np.quantile(nz, quantile_low))
        vmax = float(np.quantile(nz, quantile_high))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = float(nz.min()), float(nz.max())
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        log_hist,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin="lower",
        cmap=cmap,
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title("Caustic Image (log-scaled)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log-scaled histogram" + (" (log1p)" if log_transform == "log1p" else ""))

    fig.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def _choose_ray_indices(ny, nx, num_rays=25, mode="grid"):
    if num_rays <= 0:
        return []

    if mode == "grid":
        gx = max(1, int(np.sqrt(num_rays)))
        gy = max(1, int(np.ceil(num_rays / gx)))

        xs = np.linspace(0, nx - 1, gx).astype(int)
        ys = np.linspace(0, ny - 1, gy).astype(int)

        indices = [(iy, ix) for iy in ys for ix in xs]
        return indices[:num_rays]

    elif mode == "random":
        total = ny * nx
        flat_ids = np.random.choice(total, size=min(num_rays, total), replace=False)
        return [(fid // nx, fid % nx) for fid in flat_ids]

    else:
        raise ValueError("mode must be 'grid' or 'random'")


def plot_ray_trajectories_xz(trace_result, num_rays=25, mode="grid",
                             figsize=(6, 5), alpha=0.8, lw=1.0,
                             show=True, savepath=None):
    x = trace_result["x"]
    z = trace_result["z"]
    _, ny, nx = x.shape

    indices = _choose_ray_indices(ny, nx, num_rays=num_rays, mode=mode)

    fig, ax = plt.subplots(figsize=figsize)

    for iy, ix in indices:
        ax.plot(x[:, iy, ix], z, alpha=alpha, lw=lw)

    ax.set_title("Ray Trajectories in XZ")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.invert_yaxis()

    fig.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_ray_trajectories_yz(trace_result, num_rays=25, mode="grid",
                             figsize=(6, 5), alpha=0.8, lw=1.0,
                             show=True, savepath=None):
    y = trace_result["y"]
    z = trace_result["z"]
    _, ny, nx = y.shape

    indices = _choose_ray_indices(ny, nx, num_rays=num_rays, mode=mode)

    fig, ax = plt.subplots(figsize=figsize)

    for iy, ix in indices:
        ax.plot(y[:, iy, ix], z, alpha=alpha, lw=lw)

    ax.set_title("Ray Trajectories in YZ")
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.invert_yaxis()

    fig.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_ray_trajectories_3d(trace_result, num_rays=40, mode="grid",
                             figsize=(7, 6), alpha=0.8, lw=1.0,
                             elev=25, azim=35,
                             show=True, savepath=None):
    x = trace_result["x"]
    y = trace_result["y"]
    z = trace_result["z"]
    _, ny, nx = x.shape

    indices = _choose_ray_indices(ny, nx, num_rays=num_rays, mode=mode)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    for iy, ix in indices:
        ax.plot(x[:, iy, ix], y[:, iy, ix], z, alpha=alpha, lw=lw)

    ax.set_title("3D Ray Trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)

    fig.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_ray_speed(trace_result, figsize=(6, 4),
                   show=True, savepath=None):
    vx = trace_result["vx"]
    vy = trace_result["vy"]
    z = trace_result["z"]

    speed = np.sqrt(vx**2 + vy**2)
    mean_speed = speed.mean(axis=(1, 2))
    max_speed = speed.max(axis=(1, 2))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(z, mean_speed, label="mean transverse speed")
    ax.plot(z, max_speed, label="max transverse speed")
    ax.set_title("Ray Speed vs Depth")
    ax.set_xlabel("z")
    ax.set_ylabel("speed")
    ax.legend()

    fig.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax