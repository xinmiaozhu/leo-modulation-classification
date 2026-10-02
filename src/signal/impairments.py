"""Additional signal impairments for robustness experiments."""

from __future__ import annotations

import numpy as np


def apply_fractional_delay(
    x: np.ndarray,
    delay_samples: float,
    half_length: int = 16,
) -> np.ndarray:
    """Delay a finite complex sequence by a fractional number of samples.

    A windowed-sinc FIR is used with zero extension, avoiding the wrap-around
    implicit in an FFT phase shift. Integer delay is intentionally handled by
    the same implementation so the timing metadata has one sign convention:
    positive values delay the received waveform.
    """

    signal = np.asarray(x, dtype=np.complex128)
    delay = float(delay_samples)
    if not np.isfinite(delay):
        raise ValueError("delay_samples must be finite.")
    if abs(delay) < 1e-12:
        return signal.copy()
    radius = int(half_length)
    if radius < 2:
        raise ValueError("half_length must be at least 2.")
    integer = int(np.floor(delay))
    fractional = delay - integer
    taps_index = np.arange(-radius, radius + 1, dtype=np.float64)
    taps = np.sinc(taps_index - fractional) * np.hamming(2 * radius + 1)
    taps /= np.sum(taps)
    filtered = np.convolve(signal, taps.astype(np.float64), mode="same")
    shifted = np.zeros_like(filtered)
    if integer >= 0:
        if integer < len(signal):
            shifted[integer:] = filtered[: len(signal) - integer]
    else:
        advance = -integer
        if advance < len(signal):
            shifted[: len(signal) - advance] = filtered[advance:]
    return shifted


def apply_cfo(x: np.ndarray, fs: float, freq_offset_hz: float) -> np.ndarray:
    """Apply constant carrier-frequency offset."""

    n = np.arange(len(x), dtype=np.float64)
    t = n / float(fs)
    return x * np.exp(1j * 2.0 * np.pi * freq_offset_hz * t)


def apply_phase_offset(x: np.ndarray, phase_rad: float) -> np.ndarray:
    """Apply a constant phase rotation."""

    return x * np.exp(1j * phase_rad)


def apply_phase_noise(
    x: np.ndarray,
    std_rad: float,
    rng: np.random.Generator | None = None,
    mode: str = "random_walk",
) -> np.ndarray:
    """Apply simple phase noise.

    Args:
        mode:
            "iid": independent phase perturbation for every sample.
            "random_walk": cumulative Gaussian phase perturbation.
    """

    if rng is None:
        rng = np.random.default_rng()

    if std_rad <= 0:
        return x

    if mode == "iid":
        phase = rng.normal(0.0, std_rad, size=len(x))
    elif mode == "random_walk":
        phase = np.cumsum(rng.normal(0.0, std_rad, size=len(x)))
    else:
        raise ValueError("mode must be 'iid' or 'random_walk'.")

    return x * np.exp(1j * phase)


def apply_iq_imbalance(
    x: np.ndarray,
    gain_imbalance_db: float = 0.0,
    phase_imbalance_deg: float = 0.0,
) -> np.ndarray:
    """Apply a simple IQ imbalance model."""

    g = 10 ** (gain_imbalance_db / 20.0)
    phi = np.deg2rad(phase_imbalance_deg)

    I = np.real(x) * g
    Q = np.imag(x) * np.cos(phi) + np.real(x) * np.sin(phi)
    return I + 1j * Q


def normalize_power(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize signal to unit average power."""

    p = np.mean(np.abs(x) ** 2)
    return x / np.sqrt(p + eps)
