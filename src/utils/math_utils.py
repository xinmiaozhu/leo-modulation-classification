"""Common mathematical utilities used by physics, DRC-HOC and training modules."""

from __future__ import annotations

import numpy as np


def to_complex(iq: np.ndarray) -> np.ndarray:
    """Convert IQ array to complex array.

    Supported shapes:
        - [2, N] -> [N]
        - [N, 2] -> [N]
        - already complex -> unchanged
    """

    arr = np.asarray(iq)

    if np.iscomplexobj(arr):
        return arr

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D IQ array, got shape {arr.shape}")

    if arr.shape[0] == 2:
        return arr[0] + 1j * arr[1]

    if arr.shape[1] == 2:
        return arr[:, 0] + 1j * arr[:, 1]

    raise ValueError(f"Cannot infer IQ layout from shape {arr.shape}")


def complex_to_iq(x: np.ndarray, layout: str = "channel_first") -> np.ndarray:
    """Convert complex array to real IQ array.

    Args:
        x: Complex array with shape [N].
        layout: "channel_first" returns [2, N], "channel_last" returns [N, 2].
    """

    x = np.asarray(x)
    iq = np.stack([np.real(x), np.imag(x)], axis=0)

    if layout == "channel_first":
        return iq.astype(np.float32)
    if layout == "channel_last":
        return iq.T.astype(np.float32)

    raise ValueError("layout must be 'channel_first' or 'channel_last'")


def normalize_power(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize complex signal to unit average power."""

    power = np.mean(np.abs(x) ** 2)
    return x / np.sqrt(power + eps)


def db_to_linear(db: float | np.ndarray) -> float | np.ndarray:
    return 10 ** (np.asarray(db) / 10)


def linear_to_db(x: float | np.ndarray, eps: float = 1e-12) -> float | np.ndarray:
    return 10 * np.log10(np.asarray(x) + eps)


def add_awgn_complex(
    x: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add complex AWGN to a complex baseband signal."""

    if rng is None:
        rng = np.random.default_rng()

    signal_power = np.mean(np.abs(x) ** 2)
    snr_linear = db_to_linear(snr_db)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    )
    return x + noise


def safe_standardize(
    x: np.ndarray,
    mean: np.ndarray | float | None = None,
    std: np.ndarray | float | None = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize x and return standardized data, mean and std."""

    x = np.asarray(x, dtype=np.float32)

    if mean is None:
        mean = np.mean(x, axis=0, keepdims=True)
    if std is None:
        std = np.std(x, axis=0, keepdims=True)

    z = (x - mean) / (std + eps)
    return z.astype(np.float32), np.asarray(mean), np.asarray(std)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Compute a simple moving average."""

    if window <= 1:
        return np.asarray(x)

    x = np.asarray(x, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")
