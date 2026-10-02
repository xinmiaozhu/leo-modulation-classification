"""Feature adapters required by the paper-aligned AMC baselines."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


NASA_ORDERS: tuple[tuple[int, int], ...] = (
    (2, 0),
    (4, 0),
    (4, 1),
    (4, 2),
    (6, 0),
    (6, 1),
    (6, 2),
    (6, 3),
    (8, 0),
)
NASA_FEATURE_NAMES: tuple[str, ...] = (
    "C20_abs",
    "C40_abs",
    "C41_abs",
    "C42_abs",
    "C60_abs",
    "C61_abs",
    "C62_abs",
    "C63_abs",
    "C80_abs",
    "pilot_esn0_est_db",
)
# Each entry is (coefficient, (((moment_order, conjugations), power), ...)).
# The expressions are the zero-mean joint-cumulant partition formula grouped
# into unique moment monomials.  Grouping avoids evaluating 4,140 partitions
# separately for C80.
_CUMULANT_POLYNOMIALS = {
    (2, 0): (
        (1, (((2, 0), 1),)),
    ),
    (4, 0): (
        (1, (((4, 0), 1),)),
        (-3, (((2, 0), 2),)),
    ),
    (4, 1): (
        (1, (((4, 1), 1),)),
        (-3, (((2, 0), 1), ((2, 1), 1))),
    ),
    (4, 2): (
        (1, (((4, 2), 1),)),
        (-1, (((2, 0), 1), ((2, 2), 1))),
        (-2, (((2, 1), 2),)),
    ),
    (6, 0): (
        (1, (((6, 0), 1),)),
        (-15, (((2, 0), 1), ((4, 0), 1))),
        (-10, (((3, 0), 2),)),
        (30, (((2, 0), 3),)),
    ),
    (6, 1): (
        (1, (((6, 1), 1),)),
        (-10, (((2, 0), 1), ((4, 1), 1))),
        (-10, (((3, 0), 1), ((3, 1), 1))),
        (-5, (((2, 1), 1), ((4, 0), 1))),
        (30, (((2, 0), 2), ((2, 1), 1))),
    ),
    (6, 2): (
        (1, (((6, 2), 1),)),
        (-6, (((2, 0), 1), ((4, 2), 1))),
        (-4, (((3, 0), 1), ((3, 2), 1))),
        (-1, (((2, 2), 1), ((4, 0), 1))),
        (6, (((2, 0), 2), ((2, 2), 1))),
        (-8, (((2, 1), 1), ((4, 1), 1))),
        (-6, (((3, 1), 2),)),
        (24, (((2, 0), 1), ((2, 1), 2))),
    ),
    (6, 3): (
        (1, (((6, 3), 1),)),
        (-3, (((2, 0), 1), ((4, 3), 1))),
        (-1, (((3, 0), 1), ((3, 3), 1))),
        (-3, (((2, 2), 1), ((4, 1), 1))),
        (-9, (((3, 1), 1), ((3, 2), 1))),
        (-9, (((2, 1), 1), ((4, 2), 1))),
        (18, (((2, 0), 1), ((2, 1), 1), ((2, 2), 1))),
        (12, (((2, 1), 3),)),
    ),
    (8, 0): (
        (1, (((8, 0), 1),)),
        (-28, (((2, 0), 1), ((6, 0), 1))),
        (-56, (((3, 0), 1), ((5, 0), 1))),
        (-35, (((4, 0), 2),)),
        (420, (((2, 0), 2), ((4, 0), 1))),
        (560, (((2, 0), 1), ((3, 0), 2))),
        (-630, (((2, 0), 4),)),
    ),
}


def _moment(z: np.ndarray, p: int, q: int) -> complex:
    return complex(np.mean((z ** (p - q)) * (np.conj(z) ** q)))


def mixed_cumulants(
    symbols: np.ndarray,
    orders: Iterable[tuple[int, int]] = NASA_ORDERS,
    *,
    center: bool = True,
    power_normalize: bool = True,
) -> dict[tuple[int, int], complex]:
    """Compute the mixed cumulants used by the NASA satellite classifier."""

    z = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if z.size == 0:
        raise ValueError("At least one symbol is required.")
    if center:
        z = z - np.mean(z)
    if power_normalize:
        power = float(np.mean(np.abs(z) ** 2))
        if not np.isfinite(power) or power <= 1e-12:
            raise ValueError("Cannot power-normalize a zero-energy symbol sequence.")
        z = z / np.sqrt(power)

    requested = tuple((int(p), int(q)) for p, q in orders)
    unknown = [order for order in requested if order not in _CUMULANT_POLYNOMIALS]
    if unknown:
        raise ValueError(f"Unsupported cumulant order(s): {unknown}")

    needed_moments: set[tuple[int, int]] = set()
    for order in requested:
        for _, factors in _CUMULANT_POLYNOMIALS[order]:
            needed_moments.update(key for key, _ in factors)
    moments = {key: _moment(z, *key) for key in needed_moments}

    out: dict[tuple[int, int], complex] = {}
    for order in requested:
        value = 0.0 + 0.0j
        for coefficient, factors in _CUMULANT_POLYNOMIALS[order]:
            monomial = complex(coefficient)
            for key, power in factors:
                monomial *= moments[key] ** power
            value += monomial
        out[order] = complex(value)
    return out


def pilot_esn0_from_evm(
    pilot_evm_mse: float,
    *,
    minimum_db: float = -20.0,
    maximum_db: float = 40.0,
) -> float:
    """Estimate Es/N0 from pilot mean-square EVM without using true SNR."""

    evm = float(pilot_evm_mse)
    if not np.isfinite(evm) or evm <= 0.0:
        return float(minimum_db)
    estimate = -10.0 * np.log10(max(evm, 1e-12))
    return float(np.clip(estimate, minimum_db, maximum_db))


def nasa_feature_vector(symbols: np.ndarray, pilot_evm_mse: float) -> np.ndarray:
    cumulants = mixed_cumulants(symbols, NASA_ORDERS)
    values = [abs(cumulants[order]) for order in NASA_ORDERS]
    values.append(pilot_esn0_from_evm(pilot_evm_mse))
    return np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

