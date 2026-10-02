"""Pulse-shaping utilities."""

from __future__ import annotations

import numpy as np


def upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    """Upsample symbols by inserting zeros."""

    if sps <= 0:
        raise ValueError("sps must be positive.")

    symbols = np.asarray(symbols)
    out = np.zeros(len(symbols) * sps, dtype=np.complex128)
    out[::sps] = symbols
    return out


def rrc_filter(beta: float = 0.35, span: int = 8, sps: int = 8) -> np.ndarray:
    """Generate a root-raised-cosine filter.

    Args:
        beta: Roll-off factor in [0, 1].
        span: Filter span in symbols.
        sps: Samples per symbol.

    Returns:
        Real-valued RRC filter taps normalized to unit energy.
    """

    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1].")
    if span <= 0 or sps <= 0:
        raise ValueError("span and sps must be positive.")

    N = span * sps
    t = np.arange(-N / 2, N / 2 + 1, dtype=np.float64) / sps
    h = np.zeros_like(t)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and abs(abs(4.0 * beta * ti) - 1.0) < 1e-10:
            h[i] = (
                beta / np.sqrt(2.0)
                * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                )
            )
        else:
            numerator = (
                np.sin(np.pi * ti * (1 - beta))
                + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            )
            denominator = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = numerator / denominator

    h = h / np.sqrt(np.sum(h**2) + 1e-12)
    return h.astype(np.float64)


def pulse_shape_symbols(
    symbols: np.ndarray,
    sps: int = 8,
    beta: float = 0.35,
    span: int = 8,
    mode: str = "same",
) -> np.ndarray:
    """Upsample and RRC-filter symbols."""

    up = upsample(symbols, sps)
    taps = rrc_filter(beta=beta, span=span, sps=sps)
    shaped = np.convolve(up, taps, mode=mode)
    return shaped.astype(np.complex128)


def trim_or_pad(x: np.ndarray, length: int) -> np.ndarray:
    """Trim or zero-pad a signal to a target length."""

    x = np.asarray(x)
    if len(x) >= length:
        return x[:length]

    out = np.zeros(length, dtype=x.dtype)
    out[: len(x)] = x
    return out
