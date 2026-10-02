"""Quadratic phase severity and degradation-boundary utilities."""

from __future__ import annotations

import numpy as np
from scipy.special import fresnel


def compute_gamma(mu_hz_per_s: float | np.ndarray, observation_time_s: float | np.ndarray):
    """Compute quadratic phase severity gamma = pi * |mu| * T^2."""

    return np.pi * np.abs(mu_hz_per_s) * np.asarray(observation_time_s) ** 2


def compute_gamma_from_window(mu_hz_per_s: float, num_samples: int, sample_rate_hz: float) -> float:
    """Compute gamma from sample count and sample rate."""

    observation_time_s = num_samples / float(sample_rate_hz)
    return float(compute_gamma(mu_hz_per_s, observation_time_s))


def oscillatory_attenuation_factor(
    mu_hz_per_s: float,
    observation_time_s: float,
    k: int = 4,
    num_grid: int = 4096,
    centered_time: bool = False,
) -> float:
    """Compute the normalized quadratic-phase coherence magnitude.

    The default interval uses the frame-start convention from zero to ``T``,
    matching signal generation, pilot estimation, and compensation in the
    current LEO pipeline. Set ``centered_time=True`` only for a separately
    defined centered-coordinate analysis.
    """

    T = float(observation_time_s)
    if T <= 0:
        raise ValueError("observation_time_s must be positive.")

    if centered_time:
        t = np.linspace(-0.5 * T, 0.5 * T, num_grid)
    else:
        t = np.linspace(0.0, T, num_grid)
    integrand = np.exp(1j * k * np.pi * mu_hz_per_s * t**2)
    val = np.trapezoid(integrand, t) / T
    return float(np.abs(val))


def discrete_attenuation_factor(
    gamma: float | np.ndarray,
    k: int,
    num_samples: int = 8192,
) -> np.ndarray:
    """Compute finite-sample attenuation on the frame-start coordinate."""

    values = np.atleast_1d(np.asarray(gamma, dtype=np.float64))
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2.")
    u = np.arange(num_samples, dtype=np.float64) / float(num_samples)
    result = np.asarray(
        [np.abs(np.mean(np.exp(1j * float(k) * g * u**2))) for g in values],
        dtype=np.float64,
    )
    return result if np.ndim(gamma) else result[0]


def fresnel_attenuation(
    gamma: float | np.ndarray,
    k: int,
    rate_sign: float | np.ndarray = 1.0,
) -> np.ndarray | complex:
    """Continuous quadratic-phase attenuation in closed Fresnel form.

    This evaluates

        A_k(gamma) = integral_0^1 exp(j sign(mu) k gamma u^2) du.

    SciPy returns the Fresnel pair as ``S(x), C(x)``.  The implementation is
    continuous at ``k * gamma = 0`` and supports scalar or broadcastable array
    inputs.  The magnitude is independent of ``rate_sign``.
    """

    gamma_arr, sign_arr = np.broadcast_arrays(
        np.asarray(gamma, dtype=np.float64),
        np.asarray(rate_sign, dtype=np.float64),
    )
    if np.any(gamma_arr < 0.0):
        raise ValueError("gamma must be nonnegative.")
    a = sign_arr * float(k) * gamma_arr
    abs_a = np.abs(a)
    result = np.ones(a.shape, dtype=np.complex128)
    nonzero = abs_a > 0.0
    if np.any(nonzero):
        x = np.sqrt(2.0 * abs_a[nonzero] / np.pi)
        s_value, c_value = fresnel(x)
        scale = np.sqrt(np.pi / (2.0 * abs_a[nonzero]))
        result[nonzero] = scale * (
            c_value + 1j * np.sign(a[nonzero]) * s_value
        )
    return complex(result.item()) if result.ndim == 0 else result


def fresnel_attenuation_magnitude(
    gamma: float | np.ndarray,
    k: int,
) -> np.ndarray | float:
    """Return ``|A_k(gamma)|`` from the continuous Fresnel expression."""

    value = np.abs(fresnel_attenuation(gamma, k=k, rate_sign=1.0))
    return float(value) if np.ndim(value) == 0 else value


def attenuation_near_zero_lower_bound(
    gamma: float | np.ndarray,
    k: int,
) -> np.ndarray | float:
    """A global nonnegative lower bound tight to second order at zero.

    From ``|E exp(j a U^2)|^2`` and ``cos(x) >= 1-x^2/2`` for
    ``U ~ Uniform(0,1)``, the bound is

        |A_k(gamma)| >= sqrt(max(0, 1 - 4 (k gamma)^2 / 45)).

    The bound becomes vacuous once the expression inside the square root is
    negative, but is useful in the low-severity operating region.
    """

    gamma_arr = np.asarray(gamma, dtype=np.float64)
    if np.any(gamma_arr < 0.0):
        raise ValueError("gamma must be nonnegative.")
    bound = np.sqrt(np.maximum(0.0, 1.0 - 4.0 * (float(k) * gamma_arr) ** 2 / 45.0))
    return float(bound) if bound.ndim == 0 else bound


def finite_sample_attenuation_error_bound(
    gamma: float | np.ndarray,
    k: int,
    num_samples: int,
) -> np.ndarray | float:
    """Left-Riemann error bound between finite and continuous attenuation.

    For ``f(u)=exp(j k gamma u^2)``, ``sup |f'(u)|=2|k|gamma``.  Summing
    the interval-wise left-rule errors gives

        |A_k^(N)(gamma) - A_k(gamma)| <= |k| gamma / N.
    """

    if int(num_samples) < 1:
        raise ValueError("num_samples must be positive.")
    gamma_arr = np.asarray(gamma, dtype=np.float64)
    if np.any(gamma_arr < 0.0):
        raise ValueError("gamma must be nonnegative.")
    bound = abs(int(k)) * gamma_arr / float(num_samples)
    return float(bound) if bound.ndim == 0 else bound


def constant_energy_iq_perturbation(
    gamma: float | np.ndarray,
) -> np.ndarray | float:
    """Normalized I/Q perturbation for a constant-energy continuous frame.

    The normalized squared distance between the undistorted waveform and its
    quadratically rotated version is ``2 * (1 - Re A_1(gamma))``.
    """

    attenuation = fresnel_attenuation(gamma, k=1, rate_sign=1.0)
    distance = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - np.real(attenuation))))
    return float(distance) if np.ndim(distance) == 0 else distance


def hoc_relative_error(C_mu, C_0, eps: float = 1e-12):
    """Relative HOC-feature error |C_mu - C_0| / |C_0|."""

    return np.abs(C_mu - C_0) / (np.abs(C_0) + eps)


def find_gamma_boundary(
    gammas: np.ndarray,
    errors: np.ndarray,
    epsilon: float,
) -> float | None:
    """Find the first gamma where error exceeds epsilon.

    Returns None if no boundary is found.
    """

    gammas = np.asarray(gammas)
    errors = np.asarray(errors)

    order = np.argsort(gammas)
    gammas = gammas[order]
    errors = errors[order]

    idx = np.where(errors >= epsilon)[0]
    if len(idx) == 0:
        return None
    return float(gammas[idx[0]])


def categorize_gamma_region(
    error: float,
    epsilon_safe: float,
    epsilon_fail: float,
) -> str:
    """Categorize degradation region by feature error."""

    if epsilon_safe >= epsilon_fail:
        raise ValueError("epsilon_safe must be smaller than epsilon_fail.")

    if error < epsilon_safe:
        return "quasi_static_safe"
    if error < epsilon_fail:
        return "degradation_transition"
    return "feature_failure"


def bin_by_gamma(
    gammas: np.ndarray,
    values: np.ndarray,
    bins: np.ndarray,
) -> dict[str, np.ndarray]:
    """Aggregate values by gamma bins.

    Returns:
        dict with bin centers, mean, std and count.
    """

    gammas = np.asarray(gammas)
    values = np.asarray(values)
    bins = np.asarray(bins)

    centers = 0.5 * (bins[:-1] + bins[1:])
    means = np.full_like(centers, np.nan, dtype=np.float64)
    stds = np.full_like(centers, np.nan, dtype=np.float64)
    counts = np.zeros_like(centers, dtype=np.int64)

    for i in range(len(centers)):
        mask = (gammas >= bins[i]) & (gammas < bins[i + 1])
        counts[i] = int(np.sum(mask))
        if counts[i] > 0:
            means[i] = float(np.mean(values[mask]))
            stds[i] = float(np.std(values[mask]))

    return {
        "centers": centers,
        "mean": means,
        "std": stds,
        "count": counts,
    }
