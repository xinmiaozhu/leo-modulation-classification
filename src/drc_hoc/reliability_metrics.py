"""Reliability metrics for DRC-HOC features."""

from __future__ import annotations

import numpy as np

from .cumulants import extract_hoc_features


def compute_feature_stability(features: np.ndarray, eps: float = 1e-12) -> dict[str, float]:
    """Compute stability statistics over subwindow features.

    Args:
        features: Shape [L, D].

    Returns:
        mean_variance: Mean feature variance across dimensions.
        trace_cov: Trace of covariance matrix.
        relative_variance: Variance normalized by squared mean magnitude.
    """

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"features must have shape [L, D], got {features.shape}")

    if features.shape[0] <= 1:
        return {
            "mean_variance": 0.0,
            "trace_cov": 0.0,
            "relative_variance": 0.0,
        }

    var_dim = np.var(features, axis=0)
    mean_variance = float(np.mean(var_dim))
    trace_cov = float(np.sum(var_dim))
    mean_norm2 = float(np.mean(np.mean(features, axis=0) ** 2))
    relative_variance = float(mean_variance / (mean_norm2 + eps))

    return {
        "mean_variance": mean_variance,
        "trace_cov": trace_cov,
        "relative_variance": relative_variance,
    }


def compute_V_hoc(
    x_comp: np.ndarray,
    num_subwindows: int = 4,
    orders: tuple[tuple[int, int], ...] = ((2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (6, 0), (6, 3)),
    representation: str = "real_imag_abs",
) -> float:
    """Compute local HOC variance V_hoc over subwindows."""

    x_comp = np.asarray(x_comp)
    if num_subwindows <= 1:
        return 0.0

    N = len(x_comp)
    seg_len = N // num_subwindows
    if seg_len < 8:
        raise ValueError(
            f"Subwindow too short for HOC extraction: N={N}, L={num_subwindows}."
        )

    h_list = []
    for i in range(num_subwindows):
        seg = x_comp[i * seg_len : (i + 1) * seg_len]
        h = extract_hoc_features(seg, orders=orders, representation=representation)
        h_list.append(h)

    h_arr = np.stack(h_list, axis=0)
    h_mean = np.mean(h_arr, axis=0)
    V_hoc = np.mean(np.sum((h_arr - h_mean) ** 2, axis=1))
    return float(V_hoc)


def normalize_reliability_metadata(
    gamma_hat: float,
    S_peak: float,
    V_hoc: float,
    stats: dict[str, tuple[float, float]],
    eps: float = 1e-8,
) -> np.ndarray:
    """Standardize reliability metadata using train-set statistics.

    Args:
        stats: Dict like {"gamma_hat": (mean, std), ...}
    """

    values = {
        "gamma_hat": gamma_hat,
        "S_peak": S_peak,
        "V_hoc": V_hoc,
    }

    out = []
    for key in ["gamma_hat", "S_peak", "V_hoc"]:
        mean, std = stats[key]
        out.append((values[key] - mean) / (std + eps))

    return np.asarray(out, dtype=np.float32)
