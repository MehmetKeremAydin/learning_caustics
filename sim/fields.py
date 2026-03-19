from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


ArrayLike = np.ndarray | float


class DensityField(ABC):
    def __init__(self, n0: float = 1.0, beta: float = 1.0) -> None:
        self.n0 = float(n0)
        self.beta = float(beta)

    @abstractmethod
    def density(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def grad_density(
        self, x: ArrayLike, y: ArrayLike, z: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def refractive_index(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> np.ndarray:
        return self.n0 + self.beta * self.density(x, y, z)

    def grad_refractive_index(
        self, x: ArrayLike, y: ArrayLike, z: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gx, gy, gz = self.grad_density(x, y, z)
        return self.beta * gx, self.beta * gy, self.beta * gz

    def transverse_grad_refractive_index(
        self, x: ArrayLike, y: ArrayLike, z: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        gx, gy, _ = self.grad_refractive_index(x, y, z)
        return gx, gy


@dataclass
class GaussianBlob(DensityField):
    amplitude: float = 1.0
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.5
    sigma: float = 0.2
    n0: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        DensityField.__init__(self, n0=self.n0, beta=self.beta)
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    def density(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        dx = x - self.center_x
        dy = y - self.center_y
        dz = z - self.center_z

        r2 = dx**2 + dy**2 + dz**2
        return self.amplitude * np.exp(-r2 / (2.0 * self.sigma**2))

    def grad_density(
        self, x: ArrayLike, y: ArrayLike, z: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho = self.density(x, y, z)

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        dx = x - self.center_x
        dy = y - self.center_y
        dz = z - self.center_z

        inv_sigma2 = 1.0 / (self.sigma**2)
        gx = -dx * inv_sigma2 * rho
        gy = -dy * inv_sigma2 * rho
        gz = -dz * inv_sigma2 * rho
        return gx, gy, gz


@dataclass
class AnisotropicGaussianBlob(DensityField):
    amplitude: float = 1.0
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.5
    sigma_x: float = 0.2
    sigma_y: float = 0.2
    sigma_z: float = 0.2
    n0: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        DensityField.__init__(self, n0=self.n0, beta=self.beta)
        if self.sigma_x <= 0 or self.sigma_y <= 0 or self.sigma_z <= 0:
            raise ValueError("All sigmas must be positive")

    def density(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        dx = x - self.center_x
        dy = y - self.center_y
        dz = z - self.center_z

        exponent = -0.5 * (
            (dx / self.sigma_x) ** 2
            + (dy / self.sigma_y) ** 2
            + (dz / self.sigma_z) ** 2
        )
        return self.amplitude * np.exp(exponent)

    def grad_density(
        self, x: ArrayLike, y: ArrayLike, z: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho = self.density(x, y, z)

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        dx = x - self.center_x
        dy = y - self.center_y
        dz = z - self.center_z

        gx = -(dx / self.sigma_x**2) * rho
        gy = -(dy / self.sigma_y**2) * rho
        gz = -(dz / self.sigma_z**2) * rho
        return gx, gy, gz


@dataclass
class GaussianMixture(DensityField):
    components: Sequence[DensityField]
    n0: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        DensityField.__init__(self, n0=self.n0, beta=self.beta)
        if len(self.components) == 0:
            raise ValueError("GaussianMixture needs at least one component")

    def density(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> np.ndarray:
        total = None
        for comp in self.components:
            rho = comp.density(x, y, z)
            total = rho if total is None else total + rho
        assert total is not None
        return total

    def grad_density(
        self, x: ArrayLike, y: ArrayLike, z: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gx_total = gy_total = gz_total = None

        for comp in self.components:
            gx, gy, gz = comp.grad_density(x, y, z)
            gx_total = gx if gx_total is None else gx_total + gx
            gy_total = gy if gy_total is None else gy_total + gy
            gz_total = gz if gz_total is None else gz_total + gz

        assert gx_total is not None and gy_total is not None and gz_total is not None
        return gx_total, gy_total, gz_total


@dataclass
class RadialCylinder(DensityField):
    amplitude: float = 1.0
    center_x: float = 0.0
    center_y: float = 0.0
    sigma: float = 0.3
    radius: float = 0.5
    profile: str = "gaussian"
    n0: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        DensityField.__init__(self, n0=self.n0, beta=self.beta)
        if self.profile not in {"gaussian", "quadratic"}:
            raise ValueError("profile must be 'gaussian' or 'quadratic'")
        if self.profile == "gaussian" and self.sigma <= 0:
            raise ValueError("sigma must be positive for gaussian profile")
        if self.profile == "quadratic" and self.radius <= 0:
            raise ValueError("radius must be positive for quadratic profile")

    def density(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        _ = np.asarray(z, dtype=np.float64)

        dx = x - self.center_x
        dy = y - self.center_y
        r2 = dx**2 + dy**2

        if self.profile == "gaussian":
            return self.amplitude * np.exp(-r2 / (2.0 * self.sigma**2))

        q = np.maximum(1.0 - r2 / (self.radius**2), 0.0)
        return self.amplitude * q**2

    def grad_density(
        self, x: ArrayLike, y: ArrayLike, z: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        dx = x - self.center_x
        dy = y - self.center_y

        if self.profile == "gaussian":
            rho = self.density(x, y, z)
            inv_sigma2 = 1.0 / (self.sigma**2)
            gx = -dx * inv_sigma2 * rho
            gy = -dy * inv_sigma2 * rho
            gz = np.zeros_like(gx)
            return gx, gy, gz

        r2 = dx**2 + dy**2
        q = 1.0 - r2 / (self.radius**2)
        inside = q > 0.0

        gx = np.zeros_like(dx)
        gy = np.zeros_like(dy)
        gz = np.zeros_like(dx)

        gx[inside] = self.amplitude * 2.0 * q[inside] * (-2.0 * dx[inside] / self.radius**2)
        gy[inside] = self.amplitude * 2.0 * q[inside] * (-2.0 * dy[inside] / self.radius**2)

        return gx, gy, gz


@dataclass
class SinusoidalField(DensityField):
    amplitude: float = 1.0
    freq_x: float = 1.0
    freq_y: float = 1.0
    freq_z: float = 1.0
    phase_x: float = 0.0
    phase_y: float = 0.0
    phase_z: float = 0.0
    n0: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        DensityField.__init__(self, n0=self.n0, beta=self.beta)

    def density(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        sx = np.sin(2.0 * np.pi * self.freq_x * x + self.phase_x)
        sy = np.sin(2.0 * np.pi * self.freq_y * y + self.phase_y)
        sz = np.sin(2.0 * np.pi * self.freq_z * z + self.phase_z)

        return self.amplitude * sx * sy * sz

    def grad_density(
        self, x: ArrayLike, y: ArrayLike, z: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        ax = 2.0 * np.pi * self.freq_x * x + self.phase_x
        ay = 2.0 * np.pi * self.freq_y * y + self.phase_y
        az = 2.0 * np.pi * self.freq_z * z + self.phase_z

        sx, cx = np.sin(ax), np.cos(ax)
        sy, cy = np.sin(ay), np.cos(ay)
        sz, cz = np.sin(az), np.cos(az)

        gx = self.amplitude * (2.0 * np.pi * self.freq_x) * cx * sy * sz
        gy = self.amplitude * (2.0 * np.pi * self.freq_y) * sx * cy * sz
        gz = self.amplitude * (2.0 * np.pi * self.freq_z) * sx * sy * cz
        return gx, gy, gz


def make_default_field() -> DensityField:
    return GaussianBlob(
        amplitude=1.0,
        center_x=0.0,
        center_y=0.0,
        center_z=0.5,
        sigma=0.22,
        n0=1.0,
        beta=0.08,
    )


import numpy as np


class RandomGaussianMixtureField:
    def __init__(
        self,
        num_blobs=8,
        amplitude_range=(-1.0, 1.0),
        sigma_x_range=(0.05, 0.35),
        sigma_y_range=(0.05, 0.35),
        sigma_z_range=(0.05, 0.30),
        center_x_range=(-0.8, 0.8),
        center_y_range=(-0.8, 0.8),
        center_z_range=(0.10, 0.90),
        n0=1.0,
        beta=0.12,
        sinusoid_amplitude=0.0,
        sinusoid_kx_range=(4.0, 10.0),
        sinusoid_ky_range=(4.0, 10.0),
        sinusoid_kz_range=(3.0, 8.0),
        num_sinusoids=0,
        z_mod_amplitude=0.0,
        z_mod_k=3.0,
        z_mod_phase=0.0,
        seed=None,
    ):
        self.num_blobs = int(num_blobs)
        self.n0 = float(n0)
        self.beta = float(beta)

        self.sinusoid_amplitude = float(sinusoid_amplitude)
        self.num_sinusoids = int(num_sinusoids)

        self.z_mod_amplitude = float(z_mod_amplitude)
        self.z_mod_k = float(z_mod_k)
        self.z_mod_phase = float(z_mod_phase)

        rng = np.random.default_rng(seed)

        self.amplitudes = rng.uniform(
            amplitude_range[0], amplitude_range[1], self.num_blobs
        )

        self.center_x = rng.uniform(
            center_x_range[0], center_x_range[1], self.num_blobs
        )
        self.center_y = rng.uniform(
            center_y_range[0], center_y_range[1], self.num_blobs
        )
        self.center_z = rng.uniform(
            center_z_range[0], center_z_range[1], self.num_blobs
        )

        self.sigma_x = rng.uniform(
            sigma_x_range[0], sigma_x_range[1], self.num_blobs
        )
        self.sigma_y = rng.uniform(
            sigma_y_range[0], sigma_y_range[1], self.num_blobs
        )
        self.sigma_z = rng.uniform(
            sigma_z_range[0], sigma_z_range[1], self.num_blobs
        )

        if self.num_sinusoids > 0 and self.sinusoid_amplitude != 0.0:
            self.sin_amplitudes = rng.uniform(
                -self.sinusoid_amplitude,
                self.sinusoid_amplitude,
                self.num_sinusoids,
            )
            self.sin_kx = rng.uniform(
                sinusoid_kx_range[0], sinusoid_kx_range[1], self.num_sinusoids
            )
            self.sin_ky = rng.uniform(
                sinusoid_ky_range[0], sinusoid_ky_range[1], self.num_sinusoids
            )
            self.sin_kz = rng.uniform(
                sinusoid_kz_range[0], sinusoid_kz_range[1], self.num_sinusoids
            )
            self.sin_phase = rng.uniform(
                0.0, 2.0 * np.pi, self.num_sinusoids
            )
        else:
            self.sin_amplitudes = np.array([], dtype=np.float64)
            self.sin_kx = np.array([], dtype=np.float64)
            self.sin_ky = np.array([], dtype=np.float64)
            self.sin_kz = np.array([], dtype=np.float64)
            self.sin_phase = np.array([], dtype=np.float64)

    def _blob_density_only(self, x, y, z):
        rho = np.zeros_like(x, dtype=np.float64)

        for i in range(self.num_blobs):
            dx = x - self.center_x[i]
            dy = y - self.center_y[i]
            dz = z - self.center_z[i]

            q = (
                (dx / self.sigma_x[i]) ** 2
                + (dy / self.sigma_y[i]) ** 2
                + (dz / self.sigma_z[i]) ** 2
            )

            rho += self.amplitudes[i] * np.exp(-0.5 * q)

        return rho

    def _blob_grad_only(self, x, y, z):
        gx = np.zeros_like(x, dtype=np.float64)
        gy = np.zeros_like(x, dtype=np.float64)
        gz = np.zeros_like(x, dtype=np.float64)

        for i in range(self.num_blobs):
            dx = x - self.center_x[i]
            dy = y - self.center_y[i]
            dz = z - self.center_z[i]

            q = (
                (dx / self.sigma_x[i]) ** 2
                + (dy / self.sigma_y[i]) ** 2
                + (dz / self.sigma_z[i]) ** 2
            )

            blob = self.amplitudes[i] * np.exp(-0.5 * q)

            gx += -(dx / (self.sigma_x[i] ** 2)) * blob
            gy += -(dy / (self.sigma_y[i] ** 2)) * blob
            gz += -(dz / (self.sigma_z[i] ** 2)) * blob

        return gx, gy, gz

    def _sinusoid_density_only(self, x, y, z):
        if self.num_sinusoids == 0 or self.sinusoid_amplitude == 0.0:
            return np.zeros_like(x, dtype=np.float64)

        rho = np.zeros_like(x, dtype=np.float64)

        for i in range(self.num_sinusoids):
            arg = (
                self.sin_kx[i] * x
                + self.sin_ky[i] * y
                + self.sin_kz[i] * z
                + self.sin_phase[i]
            )
            rho += self.sin_amplitudes[i] * np.sin(arg)

        return rho

    def _sinusoid_grad_only(self, x, y, z):
        gx = np.zeros_like(x, dtype=np.float64)
        gy = np.zeros_like(x, dtype=np.float64)
        gz = np.zeros_like(x, dtype=np.float64)

        if self.num_sinusoids == 0 or self.sinusoid_amplitude == 0.0:
            return gx, gy, gz

        for i in range(self.num_sinusoids):
            arg = (
                self.sin_kx[i] * x
                + self.sin_ky[i] * y
                + self.sin_kz[i] * z
                + self.sin_phase[i]
            )
            c = np.cos(arg)

            gx += self.sin_amplitudes[i] * self.sin_kx[i] * c
            gy += self.sin_amplitudes[i] * self.sin_ky[i] * c
            gz += self.sin_amplitudes[i] * self.sin_kz[i] * c

        return gx, gy, gz

    def _z_modulation(self, z):
        return 1.0 + self.z_mod_amplitude * np.sin(self.z_mod_k * z + self.z_mod_phase)

    def _z_modulation_derivative(self, z):
        return self.z_mod_amplitude * self.z_mod_k * np.cos(
            self.z_mod_k * z + self.z_mod_phase
        )

    def density(self, x, y, z):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        rho_base = self._blob_density_only(x, y, z) + self._sinusoid_density_only(x, y, z)
        mod = self._z_modulation(z)

        return rho_base * mod

    def grad_density(self, x, y, z):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        rho_base = self._blob_density_only(x, y, z) + self._sinusoid_density_only(x, y, z)
        gx_base, gy_base, gz_base = self._blob_grad_only(x, y, z)
        gx_s, gy_s, gz_s = self._sinusoid_grad_only(x, y, z)

        gx_base += gx_s
        gy_base += gy_s
        gz_base += gz_s

        mod = self._z_modulation(z)
        dmod_dz = self._z_modulation_derivative(z)

        gx = gx_base * mod
        gy = gy_base * mod
        gz = gz_base * mod + rho_base * dmod_dz

        return gx, gy, gz

    def refractive_index(self, x, y, z):
        return self.n0 + self.beta * self.density(x, y, z)

    def grad_refractive_index(self, x, y, z):
        gx, gy, gz = self.grad_density(x, y, z)
        return self.beta * gx, self.beta * gy, self.beta * gz

    def transverse_grad_refractive_index(self, x, y, z):
        gx, gy, _ = self.grad_refractive_index(x, y, z)
        return gx, gy

    def get_blob_params(self):
        params = []
        for i in range(self.num_blobs):
            params.append(
                {
                    "amplitude": float(self.amplitudes[i]),
                    "center_x": float(self.center_x[i]),
                    "center_y": float(self.center_y[i]),
                    "center_z": float(self.center_z[i]),
                    "sigma_x": float(self.sigma_x[i]),
                    "sigma_y": float(self.sigma_y[i]),
                    "sigma_z": float(self.sigma_z[i]),
                }
            )
        return params

    def get_sinusoid_params(self):
        params = []
        for i in range(self.num_sinusoids):
            params.append(
                {
                    "amplitude": float(self.sin_amplitudes[i]),
                    "kx": float(self.sin_kx[i]),
                    "ky": float(self.sin_ky[i]),
                    "kz": float(self.sin_kz[i]),
                    "phase": float(self.sin_phase[i]),
                }
            )
        return params