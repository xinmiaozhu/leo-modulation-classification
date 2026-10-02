"""Digital modulation utilities.

Supported modulations:
    BPSK, QPSK, 8PSK, 16QAM, 64QAM, 16APSK, 32APSK

The modulator returns unit-average-power complex baseband symbols.

APSK note:
    The APSK constellations use common DVB-S2-style ring structures:
        16APSK: 4 inner + 12 outer, radius ratio gamma = R2/R1
        32APSK: 4 inner + 12 middle + 16 outer, ratios gamma1=R2/R1, gamma2=R3/R1
    The exact ratios can vary by code rate in standards. Here we use stable
    default ratios that are suitable for synthetic AMC experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def available_modulations() -> list[str]:
    return ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "16APSK", "32APSK"]


def modulation_order(modulation: str) -> int:
    """Return constellation size."""

    modulation = modulation.upper()
    if modulation == "BPSK":
        return 2
    if modulation == "QPSK":
        return 4
    if modulation == "8PSK":
        return 8
    if modulation == "16QAM":
        return 16
    if modulation == "64QAM":
        return 64
    if modulation == "16APSK":
        return 16
    if modulation == "32APSK":
        return 32
    raise ValueError(f"Unsupported modulation: {modulation}")


def bits_per_symbol(modulation: str) -> int:
    return int(np.log2(modulation_order(modulation)))


def _normalize_constellation(points: np.ndarray) -> np.ndarray:
    power = np.mean(np.abs(points) ** 2)
    return points / np.sqrt(power)


def psk_constellation(M: int, phase_offset: float = 0.0) -> np.ndarray:
    """Generate unit-power M-PSK constellation."""

    idx = np.arange(M)
    points = np.exp(1j * (2.0 * np.pi * idx / M + phase_offset))
    return _normalize_constellation(points)


def square_qam_constellation(M: int) -> np.ndarray:
    """Generate square QAM constellation with unit average power."""

    m_side = int(np.sqrt(M))
    if m_side * m_side != M:
        raise ValueError(f"M must be a square number for square QAM, got {M}")

    levels = np.arange(-(m_side - 1), m_side, 2)
    xx, yy = np.meshgrid(levels, levels)
    points = xx.flatten() + 1j * yy.flatten()
    return _normalize_constellation(points.astype(np.complex128))


def apsk_constellation(
    ring_sizes: tuple[int, ...],
    radii: tuple[float, ...],
    phase_offsets: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Generate a generic APSK constellation.

    Args:
        ring_sizes: Number of points on each ring.
        radii: Radius of each ring before unit-power normalization.
        phase_offsets: Optional per-ring phase offsets in radians.

    Returns:
        Unit-average-power complex APSK constellation.
    """

    if len(ring_sizes) != len(radii):
        raise ValueError("ring_sizes and radii must have the same length.")

    if phase_offsets is None:
        phase_offsets = tuple(0.0 for _ in ring_sizes)
    if len(phase_offsets) != len(ring_sizes):
        raise ValueError("phase_offsets must have the same length as ring_sizes.")

    points = []
    for M_ring, radius, offset in zip(ring_sizes, radii, phase_offsets):
        idx = np.arange(M_ring)
        # Half-symbol offsets on outer rings reduce radial alignment and make
        # the APSK geometry less degenerate for synthetic AMC.
        theta = 2.0 * np.pi * idx / M_ring + offset
        points.append(radius * np.exp(1j * theta))

    return _normalize_constellation(np.concatenate(points).astype(np.complex128))


def apsk16_constellation(gamma: float = 2.85) -> np.ndarray:
    """Generate 16APSK constellation.

    Structure:
        4 points on inner ring R1 = 1
        12 points on outer ring R2 = gamma

    Default gamma is a common DVB-S2-like synthetic choice.
    """

    return apsk_constellation(
        ring_sizes=(4, 12),
        radii=(1.0, gamma),
        phase_offsets=(np.pi / 4, np.pi / 12),
    )


def apsk32_constellation(gamma1: float = 2.84, gamma2: float = 5.27) -> np.ndarray:
    """Generate 32APSK constellation.

    Structure:
        4 points on inner ring R1 = 1
        12 points on middle ring R2 = gamma1
        16 points on outer ring R3 = gamma2

    Default ratios are common DVB-S2-like synthetic choices.
    """

    return apsk_constellation(
        ring_sizes=(4, 12, 16),
        radii=(1.0, gamma1, gamma2),
        phase_offsets=(np.pi / 4, np.pi / 12, 0.0),
    )


def get_constellation(modulation: str) -> np.ndarray:
    """Return constellation points for a modulation."""

    modulation = modulation.upper()
    if modulation == "BPSK":
        return psk_constellation(2)
    if modulation == "QPSK":
        return psk_constellation(4, phase_offset=np.pi / 4)
    if modulation == "8PSK":
        return psk_constellation(8)
    if modulation == "16QAM":
        return square_qam_constellation(16)
    if modulation == "64QAM":
        return square_qam_constellation(64)
    if modulation == "16APSK":
        return apsk16_constellation()
    if modulation == "32APSK":
        return apsk32_constellation()
    raise ValueError(f"Unsupported modulation: {modulation}")


@dataclass
class Modulator:
    """Random symbol generator for a given modulation."""

    modulation: str
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        self.modulation = self.modulation.upper()
        if self.modulation not in available_modulations():
            raise ValueError(
                f"Unsupported modulation {self.modulation}. "
                f"Available: {available_modulations()}"
            )
        if self.rng is None:
            self.rng = np.random.default_rng()
        self.constellation = get_constellation(self.modulation)

    @property
    def M(self) -> int:
        return modulation_order(self.modulation)

    @property
    def bits_per_symbol(self) -> int:
        return bits_per_symbol(self.modulation)

    def generate_indices(self, num_symbols: int) -> np.ndarray:
        """Generate random constellation indices."""

        return self.rng.integers(0, self.M, size=num_symbols, endpoint=False)

    def modulate_indices(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        return self.constellation[indices]

    def modulate(self, num_symbols: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate random complex symbols.

        Returns:
            symbols: Complex baseband symbols.
            indices: Integer constellation indices.
        """

        indices = self.generate_indices(num_symbols)
        symbols = self.modulate_indices(indices)
        return symbols.astype(np.complex128), indices
