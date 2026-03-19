import numpy as np


def make_entrance_rays(nx=64, ny=64, xlim=(-1, 1), ylim=(-1, 1),
                       vx0=0.0, vy0=0.0):
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(x, y, indexing="xy")

    VX = np.full_like(X, float(vx0))
    VY = np.full_like(Y, float(vy0))

    return X, Y, VX, VY


def physics_acceleration(field, x, y, z, bend_scale=1.0):
    gx, gy, _ = field.grad_refractive_index(x, y, z)
    ax = bend_scale * gx
    ay = bend_scale * gy
    return ax, ay


def trace_rays_euler(field,
                     x0, y0, vx0, vy0,
                     zmax=1.0, num_steps=64,
                     bend_scale=1.0,
                     clip_to_bounds=False,
                     xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
                     return_acceleration=False):
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    x0 = np.asarray(x0, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    vx0 = np.asarray(vx0, dtype=np.float64)
    vy0 = np.asarray(vy0, dtype=np.float64)

    if not (x0.shape == y0.shape == vx0.shape == vy0.shape):
        raise ValueError("x0, y0, vx0, vy0 must have the same shape")

    ny, nx = x0.shape
    dz = zmax / num_steps

    x_hist = np.zeros((num_steps + 1, ny, nx), dtype=np.float64)
    y_hist = np.zeros((num_steps + 1, ny, nx), dtype=np.float64)
    vx_hist = np.zeros((num_steps + 1, ny, nx), dtype=np.float64)
    vy_hist = np.zeros((num_steps + 1, ny, nx), dtype=np.float64)
    z_hist = np.linspace(0.0, zmax, num_steps + 1, dtype=np.float64)

    x_hist[0] = x0
    y_hist[0] = y0
    vx_hist[0] = vx0
    vy_hist[0] = vy0

    if return_acceleration:
        ax_hist = np.zeros((num_steps, ny, nx), dtype=np.float64)
        ay_hist = np.zeros((num_steps, ny, nx), dtype=np.float64)

    for k in range(num_steps):
        zk = z_hist[k]

        xk = x_hist[k]
        yk = y_hist[k]
        vxk = vx_hist[k]
        vyk = vy_hist[k]

        axk, ayk = physics_acceleration(
            field=field,
            x=xk,
            y=yk,
            z=zk,
            bend_scale=bend_scale,
        )

        vx_next = vxk + dz * axk
        vy_next = vyk + dz * ayk
        x_next = xk + dz * vxk
        y_next = yk + dz * vyk

        if clip_to_bounds:
            x_next = np.clip(x_next, xlim[0], xlim[1])
            y_next = np.clip(y_next, ylim[0], ylim[1])

        vx_hist[k + 1] = vx_next
        vy_hist[k + 1] = vy_next
        x_hist[k + 1] = x_next
        y_hist[k + 1] = y_next

        if return_acceleration:
            ax_hist[k] = axk
            ay_hist[k] = ayk

    result = {
        "x": x_hist,
        "y": y_hist,
        "vx": vx_hist,
        "vy": vy_hist,
        "z": z_hist,
        "dz": dz,
    }

    if return_acceleration:
        result["ax"] = ax_hist
        result["ay"] = ay_hist

    return result


def trace_rays_semi_implicit(field,
                             x0, y0, vx0, vy0,
                             zmax=1.0, num_steps=64,
                             bend_scale=1.0,
                             clip_to_bounds=False,
                             xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
                             return_acceleration=False):
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    x0 = np.asarray(x0, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    vx0 = np.asarray(vx0, dtype=np.float64)
    vy0 = np.asarray(vy0, dtype=np.float64)

    if not (x0.shape == y0.shape == vx0.shape == vy0.shape):
        raise ValueError("x0, y0, vx0, vy0 must have the same shape")

    ny, nx = x0.shape
    dz = zmax / num_steps

    x_hist = np.zeros((num_steps + 1, ny, nx), dtype=np.float64)
    y_hist = np.zeros((num_steps + 1, ny, nx), dtype=np.float64)
    vx_hist = np.zeros((num_steps + 1, ny, nx), dtype=np.float64)
    vy_hist = np.zeros((num_steps + 1, ny, nx), dtype=np.float64)
    z_hist = np.linspace(0.0, zmax, num_steps + 1, dtype=np.float64)

    x_hist[0] = x0
    y_hist[0] = y0
    vx_hist[0] = vx0
    vy_hist[0] = vy0

    if return_acceleration:
        ax_hist = np.zeros((num_steps, ny, nx), dtype=np.float64)
        ay_hist = np.zeros((num_steps, ny, nx), dtype=np.float64)

    for k in range(num_steps):
        zk = z_hist[k]

        xk = x_hist[k]
        yk = y_hist[k]
        vxk = vx_hist[k]
        vyk = vy_hist[k]

        axk, ayk = physics_acceleration(
            field=field,
            x=xk,
            y=yk,
            z=zk,
            bend_scale=bend_scale,
        )

        vx_next = vxk + dz * axk
        vy_next = vyk + dz * ayk
        x_next = xk + dz * vx_next
        y_next = yk + dz * vy_next

        if clip_to_bounds:
            x_next = np.clip(x_next, xlim[0], xlim[1])
            y_next = np.clip(y_next, ylim[0], ylim[1])

        vx_hist[k + 1] = vx_next
        vy_hist[k + 1] = vy_next
        x_hist[k + 1] = x_next
        y_hist[k + 1] = y_next

        if return_acceleration:
            ax_hist[k] = axk
            ay_hist[k] = ayk

    result = {
        "x": x_hist,
        "y": y_hist,
        "vx": vx_hist,
        "vy": vy_hist,
        "z": z_hist,
        "dz": dz,
    }

    if return_acceleration:
        result["ax"] = ax_hist
        result["ay"] = ay_hist

    return result


def final_ray_positions(trace_result):
    return trace_result["x"][-1], trace_result["y"][-1]


def flatten_rays_over_time(trace_result, include_last_state=False):
    x = trace_result["x"]
    y = trace_result["y"]
    vx = trace_result["vx"]
    vy = trace_result["vy"]
    z = trace_result["z"]

    num_states = x.shape[0]
    num_steps = num_states - 1

    end = num_states if include_last_state else num_steps

    z_grid = np.broadcast_to(
        z[:end].reshape(-1, 1, 1),
        x[:end].shape
    )

    data = {
        "x": x[:end].reshape(-1),
        "y": y[:end].reshape(-1),
        "z": z_grid.reshape(-1),
        "vx": vx[:end].reshape(-1),
        "vy": vy[:end].reshape(-1),
    }

    if "ax" in trace_result and "ay" in trace_result:
        data["ax"] = trace_result["ax"].reshape(-1)
        data["ay"] = trace_result["ay"].reshape(-1)

    return data