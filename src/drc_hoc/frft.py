"""DFRFT and fast chirp-focusing utilities for DRC-HOC.

The paper pipeline describes a DFRFT alpha sweep. For practical alpha_steps
(e.g., 2000), this module uses a fast chirp-focusing implementation: alpha is
mapped to a candidate chirp/Doppler-rate, the branch is dechirped, and FFT
concentration is evaluated. This avoids explicit O(N^2) FRFT integration while
preserving the alpha-search and S_peak winner-takes-all logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def local_peak_sharpness(spectrum: np.ndarray, exclude_bins: int = 5, eta: float = 1e-5) -> float:
    """White-paper S_peak = (Pmax - mean_local)/(mean_local + eta)."""
    power = np.abs(spectrum) ** 2
    flat = power.reshape(-1)
    if flat.size == 0:
        return 0.0
    idx = int(np.argmax(flat))
    pmax = float(flat[idx])
    mask = np.ones_like(flat, dtype=bool)
    left = max(0, idx - exclude_bins)
    right = min(flat.size, idx + exclude_bins + 1)
    mask[left:right] = False
    rest = flat[mask]
    local_mean = float(np.mean(rest)) if rest.size else float(np.mean(flat))
    return float((pmax - local_mean) / (local_mean + eta))


def peak_sharpness(
    spectrum: np.ndarray,
    mode: str = "peak_minus_local",
    eps: float = 1e-12,
    exclude_bins: int = 5,
    eta: float = 1e-5,
) -> float:
    """Compute peak sharpness; default follows the white-paper definition."""
    if mode in {"peak_minus_local", "local"}:
        return local_peak_sharpness(spectrum, exclude_bins=exclude_bins, eta=eta)
    power = np.abs(spectrum) ** 2
    flat = power.reshape(-1)
    if flat.size == 0:
        return 0.0
    idx = int(np.argmax(flat))
    pmax = float(flat[idx])
    if mode == "pmax_over_mean":
        return pmax / (float(np.mean(flat)) + eps)
    if mode == "pmax_over_second":
        mask = np.ones_like(flat, dtype=bool)
        mask[max(0, idx - exclude_bins):min(flat.size, idx + exclude_bins + 1)] = False
        rest = flat[mask]
        second = float(np.max(rest)) if rest.size else float(np.mean(flat))
        return pmax / (second + eps)
    raise ValueError(f"Unsupported sharpness mode: {mode}")


def _centered_grid(N: int) -> np.ndarray:
    n = np.arange(N, dtype=np.float64) - (N - 1) / 2.0
    return n / np.sqrt(N)


def frft_direct(x: np.ndarray, alpha: float, normalize: bool = True) -> np.ndarray:
    """Direct O(N^2) FRFT approximation for small validation tests."""
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)
    if N == 0:
        return x.copy()
    a = ((float(alpha) + np.pi) % (2.0 * np.pi)) - np.pi
    tol = 1e-8
    if abs(a) < tol:
        return x.copy()
    if abs(abs(a) - np.pi) < tol:
        return x[::-1].copy()
    if abs(a - np.pi / 2) < tol:
        y = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))
        return y / np.sqrt(N) if normalize else y
    if abs(a + np.pi / 2) < tol:
        y = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x)))
        return y * np.sqrt(N) if normalize else y
    t = _centered_grid(N)
    u = _centered_grid(N)
    cot_a = 1.0 / np.tan(a)
    csc_a = 1.0 / np.sin(a)
    phase = np.pi * ((u[:, None] ** 2) * cot_a - 2.0 * u[:, None] * t[None, :] * csc_a + (t[None, :] ** 2) * cot_a)
    A = np.sqrt((1.0 - 1j * cot_a) / (2.0 * np.pi))
    y = (A * np.exp(1j * phase)) @ x
    return y / np.sqrt(N) if normalize else y


def make_alpha_grid(alpha_steps: int = 2000, alpha_min: float = 1e-4) -> np.ndarray:
    """Generate alpha grid over [0, pi), excluding singular endpoints."""
    if alpha_steps <= 2:
        raise ValueError("alpha_steps must be greater than 2.")
    grid = np.arange(alpha_steps, dtype=np.float64) * (np.pi / alpha_steps)
    return grid[(grid > alpha_min) & (grid < np.pi - alpha_min)]


def alpha_to_mu_whitepaper(
    alpha: float | np.ndarray,
    k: int,
    sample_rate_hz: float,
    num_samples: int,
    calibration_scale: float | None = None,
    calibration_bias: float = 0.0,
) -> np.ndarray:
    """Map DFRFT angle to Doppler-rate.

    Ideal white-paper formula:
        mu = -cot(alpha)/(k*Ts^2)

    Discrete FRFT implementations use normalized time axes, so a calibration
    coefficient is necessary. If calibration_scale is None, use N^-2, which is
    a practical normalized-time default and should be replaced by chirp-based
    calibration in final experiments.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    Ts = 1.0 / float(sample_rate_hz)
    if calibration_scale is None:
        calibration_scale = 1.0 / (float(num_samples) ** 2)
    return calibration_scale * (-(1.0 / np.tan(alpha)) / (int(k) * Ts**2)) + calibration_bias


def fast_dfrft_chirp_focus(
    y_k: np.ndarray,
    alpha: float,
    k: int,
    sample_rate_hz: float,
    calibration_scale: float | None = None,
    calibration_bias: float = 0.0,
    centered_time: bool = False,
) -> tuple[np.ndarray, float]:
    """Fast alpha-domain DFRFT/chirp focusing.

    Returns focused FFT spectrum and the original-signal mu corresponding to
    the tested alpha.
    """
    y_k = np.asarray(y_k, dtype=np.complex128)
    N = len(y_k)
    mu_hat = float(alpha_to_mu_whitepaper(alpha, k, sample_rate_hz, N, calibration_scale, calibration_bias))
    n = np.arange(N, dtype=np.float64)
    t = n / float(sample_rate_hz)
    if centered_time:
        t = t - np.mean(t)
    # y_k contains k times the original quadratic phase, so compensate k*mu.
    y_comp = y_k * np.exp(-1j * np.pi * (k * mu_hat) * t**2)
    return np.fft.fftshift(np.fft.fft(y_comp)), mu_hat


@dataclass
class DirectFRFT:
    """Wrapper for direct FRFT search on small validation examples."""

    normalize: bool = True
    sharpness_mode: str = "peak_minus_local"
    exclude_bins: int = 5
    eta: float = 1e-5

    def transform(self, x: np.ndarray, alpha: float) -> np.ndarray:
        return frft_direct(x, alpha, normalize=self.normalize)

    def search_peak(self, x: np.ndarray, alpha_grid: np.ndarray) -> dict:
        best = {"alpha": None, "peak": -np.inf, "sharpness": -np.inf, "spectrum": None}
        for alpha in alpha_grid:
            X = self.transform(x, float(alpha))
            sharp = peak_sharpness(X, mode=self.sharpness_mode, exclude_bins=self.exclude_bins, eta=self.eta)
            if sharp > best["sharpness"]:
                best.update({"alpha": float(alpha), "peak": float(np.max(np.abs(X) ** 2)), "sharpness": float(sharp), "spectrum": X})
        return best


def frft_search_peak(x: np.ndarray, alpha_grid: np.ndarray, sharpness_mode: str = "peak_minus_local") -> dict:
    return DirectFRFT(sharpness_mode=sharpness_mode).search_peak(x, alpha_grid)
