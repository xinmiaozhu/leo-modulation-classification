"""Candidate nonlinear transforms for blind Doppler-rate estimation.

Paper-strict DRC-HOC uses a blind universal amplitude normalization first:

    x[n] = r[n] / (|r[n]| + delta)

Then it creates normalized candidate power branches only:

    y_k[n] = x[n]^k,  k in {2, 4, 8}.

Direct-power branches are kept only for optional engineering ablations.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Iterable

import numpy as np


def amplitude_normalize(x: np.ndarray, delta: float = 1e-8) -> np.ndarray:
    """Blind amplitude normalization: x[n] = r[n]/(|r[n]|+delta)."""
    x = np.asarray(x, dtype=np.complex128)
    return x / (np.abs(x) + delta)


def power_transform(x: np.ndarray, k: int) -> np.ndarray:
    """Direct k-th power transform, mainly for ablation."""
    if k <= 0:
        raise ValueError("k must be positive.")
    return np.asarray(x, dtype=np.complex128) ** k


def phase_normalized_power_transform(x: np.ndarray, k: int, delta: float = 1e-8) -> np.ndarray:
    """Blind amplitude normalization followed by k-th power transform."""
    if k <= 0:
        raise ValueError("k must be positive.")
    return amplitude_normalize(x, delta=delta) ** k


def generate_blind_normalized_candidates(
    x: np.ndarray,
    candidate_orders: Iterable[int] = (2, 4, 8),
    delta: float = 1e-8,
) -> "OrderedDict[str, np.ndarray]":
    """Generate normalized power-branch candidates."""
    x_norm = amplitude_normalize(x, delta=delta)
    candidates: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for k in candidate_orders:
        k = int(k)
        candidates[f"norm_power_{k}"] = x_norm ** k
    if not candidates:
        raise ValueError("No normalized candidate branches were generated.")
    return candidates


def generate_candidates(
    x: np.ndarray,
    candidate_orders: Iterable[int] = (2, 4, 8),
    include_direct_power: bool = False,
    include_phase_normalized: bool = True,
    delta: float = 1e-8,
) -> "OrderedDict[str, np.ndarray]":
    """Generate nonlinear-transform candidates.

    Default is paper-strict: normalized power branches only.
    Set include_direct_power=True only for ablations.
    """
    candidates: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for k in candidate_orders:
        k = int(k)
        if include_direct_power:
            candidates[f"power_{k}"] = power_transform(x, k)
        if include_phase_normalized:
            candidates[f"norm_power_{k}"] = phase_normalized_power_transform(x, k, delta=delta)
    if not candidates:
        raise ValueError("No nonlinear-transform candidates were generated.")
    return candidates


def parse_order_from_key(key: str) -> int:
    """Parse candidate order from key such as 'norm_power_4'."""
    match = re.search(r"(\d+)$", key)
    if not match:
        raise ValueError(f"Cannot parse candidate order from key: {key}")
    return int(match.group(1))


def normalize_candidate_energy(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize candidate sequence to unit average power."""
    x = np.asarray(x, dtype=np.complex128)
    p = np.mean(np.abs(x) ** 2)
    return x / np.sqrt(p + eps)
