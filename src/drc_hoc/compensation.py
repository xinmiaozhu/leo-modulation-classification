"""Doppler-rate compensation utilities."""

from __future__ import annotations

import numpy as np

from src.physics.gamma_metric import compute_gamma


def time_axis(num_samples: int, sample_rate_hz: float, centered: bool = False) -> np.ndarray:
    """Return time samples.

    Args:
        centered: If True, t is centered around zero. Centering is often useful
            for numerical stability in quadratic-phase operations.
    """

    n = np.arange(num_samples, dtype=np.float64)
    t = n / float(sample_rate_hz)
    if centered:
        t = t - np.mean(t)
    return t


def compensate_constant_doppler(
    x: np.ndarray,
    sample_rate_hz: float,
    fd0_hz: float,
    centered: bool = False,
) -> np.ndarray:
    """Remove constant Doppler/CFO phase term exp(j 2*pi*fd0*t)."""

    x = np.asarray(x, dtype=np.complex128)
    t = time_axis(len(x), sample_rate_hz, centered=centered)
    return x * np.exp(-1j * 2.0 * np.pi * fd0_hz * t)


def conjugate_quadratic_compensation(
    x: np.ndarray,
    sample_rate_hz: float,
    mu_hat_hz_per_s: float,
    centered: bool = False,
) -> np.ndarray:
    """Apply exp(-j*pi*mu_hat*t^2) compensation."""

    x = np.asarray(x, dtype=np.complex128)
    t = time_axis(len(x), sample_rate_hz, centered=centered)
    return x * np.exp(-1j * np.pi * mu_hat_hz_per_s * t**2)


def apply_quadratic_phase(
    x: np.ndarray,
    sample_rate_hz: float,
    mu_hz_per_s: float,
    centered: bool = False,
) -> np.ndarray:
    """Apply exp(j*pi*mu*t^2) quadratic phase."""

    x = np.asarray(x, dtype=np.complex128)
    t = time_axis(len(x), sample_rate_hz, centered=centered)
    return x * np.exp(1j * np.pi * mu_hz_per_s * t**2)


def residual_gamma(mu_true_hz_per_s: float, mu_hat_hz_per_s: float, observation_time_s: float) -> float:
    """Compute residual quadratic phase severity after compensation."""

    return float(compute_gamma(mu_true_hz_per_s - mu_hat_hz_per_s, observation_time_s))


def compensate_full_doppler(
    x: np.ndarray,
    sample_rate_hz: float,
    fd0_hat_hz: float = 0.0,
    mu_hat_hz_per_s: float = 0.0,
    centered: bool = False,
) -> np.ndarray:
    """Remove both constant Doppler and Doppler-rate terms."""

    y = compensate_constant_doppler(
        x, sample_rate_hz=sample_rate_hz, fd0_hz=fd0_hat_hz, centered=centered
    )
    y = conjugate_quadratic_compensation(
        y,
        sample_rate_hz=sample_rate_hz,
        mu_hat_hz_per_s=mu_hat_hz_per_s,
        centered=centered,
    )
    return y
