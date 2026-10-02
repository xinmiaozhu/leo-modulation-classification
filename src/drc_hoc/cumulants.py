"""High-order moments and cumulants for complex baseband signals.

This implementation includes a generic set-partition cumulant calculator for
orders up to 6. It avoids hand-coding fragile sixth-order complex cumulant
formulas and keeps the feature definitions mathematically consistent.

Definition:
    M_{p,q} = E[x^{p-q} (x*)^q]
where p is total order and q is the number of conjugated factors.

C_{p,q} is computed as the joint cumulant of (p-q) copies of x and q copies of
x* using the general partition formula.
"""

from __future__ import annotations

from functools import lru_cache
from math import factorial
from typing import Iterable

import numpy as np


def _to_complex_1d(x: np.ndarray) -> np.ndarray:
    """Convert input to 1D complex array.

    Accepts:
        - complex [N]
        - IQ [2, N]
        - IQ [N, 2]
    """

    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        return arr.reshape(-1).astype(np.complex128)

    if arr.ndim == 2 and arr.shape[0] == 2:
        return (arr[0] + 1j * arr[1]).reshape(-1).astype(np.complex128)

    if arr.ndim == 2 and arr.shape[1] == 2:
        return (arr[:, 0] + 1j * arr[:, 1]).reshape(-1).astype(np.complex128)

    raise ValueError(f"Expected complex vector or IQ array, got shape {arr.shape}")


def center_signal(x: np.ndarray) -> np.ndarray:
    """Remove sample mean."""

    x = _to_complex_1d(x)
    return x - np.mean(x)


def moment_pq(x: np.ndarray, p: int, q: int, center: bool = True) -> complex:
    """Compute M_{p,q} = E[x^(p-q) (x*)^q]."""

    if p < 0 or q < 0 or q > p:
        raise ValueError("Require p >= 0 and 0 <= q <= p.")

    z = center_signal(x) if center else _to_complex_1d(x)

    if p == 0:
        return 1.0 + 0.0j

    return complex(np.mean((z ** (p - q)) * (np.conj(z) ** q)))


@lru_cache(maxsize=None)
def _partitions_tuple(n: int):
    """Return all set partitions of indices 0..n-1 as tuples of tuples."""

    if n == 0:
        return ((),)

    def helper(items):
        if not items:
            yield []
            return
        first = items[0]
        for rest in helper(items[1:]):
            # Put first in an existing block
            for i in range(len(rest)):
                new_rest = [tuple(block) for block in rest]
                new_rest[i] = tuple(sorted((first,) + new_rest[i]))
                # Canonical order by first element of each block
                yield sorted(new_rest, key=lambda b: b[0])
            # Put first in a new block
            yield [(first,)] + [tuple(block) for block in rest]

    # Deduplicate because recursive construction can generate duplicate orderings.
    seen = set()
    out = []
    for part in helper(tuple(range(n))):
        canonical = tuple(sorted([tuple(block) for block in part], key=lambda b: b[0]))
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return tuple(out)


def _block_moment_from_types(
    z: np.ndarray,
    types: tuple[int, ...],
    block: tuple[int, ...],
) -> complex:
    """Moment for one block.

    types[i] = 0 means x, types[i] = 1 means conj(x).
    """

    n_conj = sum(types[i] for i in block)
    p = len(block)
    q = n_conj
    return moment_pq(z, p=p, q=q, center=False)


def cumulant_from_types(x: np.ndarray, types: Iterable[int], center: bool = True) -> complex:
    """Compute joint cumulant for variables specified by types.

    Args:
        x: Complex signal.
        types: Iterable where 0 means x and 1 means x*.
        center: Remove signal mean before computing moments.
    """

    z = center_signal(x) if center else _to_complex_1d(x)
    types = tuple(int(t) for t in types)
    n = len(types)

    if n == 0:
        return 0.0 + 0.0j

    total = 0.0 + 0.0j
    for partition in _partitions_tuple(n):
        m = len(partition)
        coeff = ((-1) ** (m - 1)) * factorial(m - 1)
        prod = 1.0 + 0.0j
        for block in partition:
            prod *= _block_moment_from_types(z, types, block)
        total += coeff * prod

    return complex(total)


def cumulant_pq(x: np.ndarray, p: int, q: int, center: bool = True) -> complex:
    """Compute complex cumulant C_{p,q}.

    C_{p,q} is the cumulant of (p-q) copies of x and q copies of x*.
    """

    if p < 1 or q < 0 or q > p:
        raise ValueError("Require p >= 1 and 0 <= q <= p.")

    types = (0,) * (p - q) + (1,) * q
    return cumulant_from_types(x, types, center=center)


def _feature_from_complex(value: complex, representation: str) -> list[float]:
    """Convert complex cumulant to real-valued features."""

    if representation == "real_imag_abs":
        return [float(np.real(value)), float(np.imag(value)), float(np.abs(value))]
    if representation == "real_imag":
        return [float(np.real(value)), float(np.imag(value))]
    if representation == "abs":
        return [float(np.abs(value))]
    if representation == "real_imag_abs_phase":
        return [
            float(np.real(value)),
            float(np.imag(value)),
            float(np.abs(value)),
            float(np.angle(value)),
        ]
    raise ValueError(f"Unsupported representation: {representation}")


def hoc_feature_names(
    orders: tuple[tuple[int, int], ...] = ((2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (6, 0), (6, 3)),
    representation: str = "real_imag_abs",
) -> list[str]:
    """Return feature names matching extract_hoc_features."""

    suffixes = {
        "real_imag_abs": ["real", "imag", "abs"],
        "real_imag": ["real", "imag"],
        "abs": ["abs"],
        "real_imag_abs_phase": ["real", "imag", "abs", "phase"],
    }[representation]

    names = []
    for p, q in orders:
        for s in suffixes:
            names.append(f"C{p}{q}_{s}")
    return names


def extract_hoc_features(
    x: np.ndarray,
    orders: tuple[tuple[int, int], ...] = ((2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (6, 0), (6, 3)),
    representation: str = "real_imag_abs",
    center: bool = True,
    normalize_power: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Extract real-valued high-order cumulant feature vector.

    Args:
        orders: Tuple of (p, q) cumulant identifiers.
        representation: How each complex cumulant is converted to real features.
        normalize_power: If True, normalize signal by RMS power before cumulants.
    """

    z = _to_complex_1d(x)
    if center:
        z = z - np.mean(z)
    if normalize_power:
        power = np.mean(np.abs(z) ** 2)
        z = z / np.sqrt(power + eps)

    feats: list[float] = []
    for p, q in orders:
        c = cumulant_pq(z, p=p, q=q, center=False)
        feats.extend(_feature_from_complex(c, representation))

    return np.asarray(feats, dtype=np.float32)


def extract_moment_features(
    x: np.ndarray,
    orders: tuple[tuple[int, int], ...] = ((2, 0), (2, 1), (4, 0), (4, 2)),
    representation: str = "real_imag_abs",
    center: bool = True,
    normalize_power: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Extract moment features, useful for debugging HOC attenuation."""

    z = _to_complex_1d(x)
    if center:
        z = z - np.mean(z)
    if normalize_power:
        power = np.mean(np.abs(z) ** 2)
        z = z / np.sqrt(power + eps)

    feats: list[float] = []
    for p, q in orders:
        m = moment_pq(z, p=p, q=q, center=False)
        feats.extend(_feature_from_complex(m, representation))
    return np.asarray(feats, dtype=np.float32)
