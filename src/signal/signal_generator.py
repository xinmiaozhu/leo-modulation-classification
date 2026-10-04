"""High-level LEO signal generator.

This module combines modulation, pulse shaping, Doppler-rate distortion and
channel impairments. It is the main entry point for synthetic dataset
generation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from src.signal.modulation import Modulator, available_modulations, get_constellation
from src.signal.pulse_shape import pulse_shape_symbols, trim_or_pad
from src.signal.channel import ChannelConfig, NonStationaryLEOChannel
from src.utils.math_utils import complex_to_iq, normalize_power
from src.signal.impairments import apply_fractional_delay


@dataclass
class SignalSpec:
    """Specification for one generated signal example."""

    modulation: str = "QPSK"
    num_symbols: int = 256
    samples_per_symbol: int = 8
    sample_rate_hz: float = 1.0e6
    carrier_frequency_hz: float = 2.0e9
    snr_db: float = 10.0
    fd0_hz: float = 0.0
    mu_hz_per_s: float = 0.0
    rrc_beta: float = 0.35
    rrc_span: int = 8
    channel_type: str = "awgn"
    compensate_constant_doppler: bool = False
    target_num_samples: int | None = None
    pilot_enabled: bool = False
    pilot_pattern: str = "none"
    pilot_interval_symbols: int = 16
    pilot_guard_symbols: int = 8
    pilot_num_symbols: int | None = None
    pilot_min_spacing_symbols: int = 1
    pilot_block_length_symbols: int = 64
    pilot_block_positions: tuple[str, ...] = ("start", "middle", "end")
    pilot_modulation: str = "QPSK"
    pilot_symbol_mode: str = "pn"
    pilot_symbol_index: int = 0
    pilot_seed: int = 2026
    timing_offset_samples: int = 0
    fractional_timing_offset_samples: float = 0.0
    rician_k_db: float = 10.0
    multipath_delays: tuple[int, ...] = (0, 3, 7)
    multipath_gains_db: tuple[float, ...] = (0.0, -6.0, -10.0)
    multipath_doppler_hz: tuple[float, ...] = ()
    phase_noise_std_rad: float = 0.0
    phase_noise_mode: str = "random_walk"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedSignal:
    """Container for one generated signal sample."""

    iq: np.ndarray
    complex_signal: np.ndarray
    label: int
    modulation: str
    symbol_indices: np.ndarray
    pilot_indices: np.ndarray
    pilot_symbols: np.ndarray
    info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "iq": self.iq,
            "complex_signal": self.complex_signal,
            "label": self.label,
            "modulation": self.modulation,
            "symbol_indices": self.symbol_indices,
            "pilot_indices": self.pilot_indices,
            "pilot_symbols": self.pilot_symbols,
            "info": self.info,
        }


class LEOSignalGenerator:
    """Generate synthetic LEO signal-recognition samples."""

    def __init__(
        self,
        modulations: list[str] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.modulations = modulations or available_modulations()
        self.modulations = [m.upper() for m in self.modulations]

        unknown = sorted(set(self.modulations) - set(available_modulations()))
        if unknown:
            raise ValueError(f"Unsupported modulations: {unknown}")

        self.label_map = {m: i for i, m in enumerate(self.modulations)}
        self.rng = np.random.default_rng() if rng is None else rng
        self._d_optimal_pilot_cache: dict[tuple[int, int, int, int], np.ndarray] = {}

    def label_of(self, modulation: str) -> int:
        modulation = modulation.upper()
        if modulation not in self.label_map:
            raise KeyError(f"Modulation {modulation} not in label map: {self.label_map}")
        return self.label_map[modulation]

    def _pilot_indices(self, spec: SignalSpec) -> np.ndarray:
        if not spec.pilot_enabled:
            return np.asarray([], dtype=np.int64)

        pattern = str(spec.pilot_pattern).lower()
        num_symbols = int(spec.num_symbols)
        guard = max(0, int(spec.pilot_guard_symbols))

        if pattern == "none":
            return np.asarray([], dtype=np.int64)

        if pattern == "comb":
            interval = max(1, int(spec.pilot_interval_symbols))
            start = min(guard, max(num_symbols - 1, 0))
            stop = max(start + 1, num_symbols - guard)
            return np.arange(start, stop, interval, dtype=np.int64)

        if pattern in {"d_optimal", "d-optimal", "doptimal"}:
            start = min(guard, max(num_symbols - 1, 0))
            stop = max(start + 1, num_symbols - guard)
            candidates = np.arange(start, stop, dtype=np.int64)
            if candidates.size == 0:
                return candidates

            if spec.pilot_num_symbols is None:
                count = np.arange(start, stop, max(1, int(spec.pilot_interval_symbols))).size
            else:
                count = int(spec.pilot_num_symbols)
            count = int(np.clip(count, 1, candidates.size))
            spacing = max(1, int(spec.pilot_min_spacing_symbols))
            cache_key = (start, stop, count, spacing)
            if cache_key not in self._d_optimal_pilot_cache:
                self._d_optimal_pilot_cache[cache_key] = self._d_optimal_pilot_indices(
                    candidates, count=count, min_spacing=spacing
                )
            return self._d_optimal_pilot_cache[cache_key].copy()

        if pattern in {"blocks", "preamble"}:
            length = max(1, int(spec.pilot_block_length_symbols))
            positions: list[int] = []
            for pos in spec.pilot_block_positions:
                key = str(pos).lower()
                if key == "start":
                    begin = guard
                elif key == "middle":
                    begin = num_symbols // 2 - length // 2
                elif key == "end":
                    begin = num_symbols - guard - length
                else:
                    raise ValueError(f"Unsupported pilot block position: {pos}")
                begin = int(np.clip(begin, 0, max(num_symbols - 1, 0)))
                end = int(np.clip(begin + length, begin + 1, num_symbols))
                positions.extend(range(begin, end))
            return np.unique(np.asarray(positions, dtype=np.int64))

        raise ValueError(f"Unsupported pilot_pattern: {spec.pilot_pattern}")

    @staticmethod
    def _d_optimal_pilot_indices(
        candidates: np.ndarray,
        count: int,
        min_spacing: int,
    ) -> np.ndarray:
        """Approximate a discrete D-optimal design for phase, CFO, and rate.

        The pilot phase model has regressors proportional to ``1``, ``t`` and
        ``t**2``. Greedy maximization of ``log det(X.T @ X)`` therefore spreads
        the selected pilot times to improve joint constant-frequency and
        Doppler-rate identifiability. The design is deterministic and preserves
        the requested pilot count so it can be compared fairly with a comb.
        """

        positions = np.asarray(candidates, dtype=np.int64).reshape(-1)
        if count >= positions.size:
            return positions.copy()

        center = 0.5 * float(positions[0] + positions[-1])
        scale = max(0.5 * float(positions[-1] - positions[0]), 1.0)
        time = (positions.astype(np.float64) - center) / scale
        rows = np.column_stack((np.ones_like(time), time, time**2))

        selected: list[int] = []
        information = np.eye(3, dtype=np.float64) * 1.0e-9

        # Seed the design with the two endpoints and a centre observation when
        # possible. This avoids an ill-conditioned first greedy step.
        anchors = [0, int(np.argmin(np.abs(time))), len(positions) - 1]
        for idx in anchors:
            if len(selected) >= count:
                break
            candidate = int(positions[idx])
            if all(abs(candidate - prior) >= min_spacing for prior in selected):
                selected.append(candidate)
                information += np.outer(rows[idx], rows[idx])

        while len(selected) < count:
            best_idx = -1
            best_score = -np.inf
            for idx, candidate in enumerate(positions):
                if int(candidate) in selected:
                    continue
                if any(abs(int(candidate) - prior) < min_spacing for prior in selected):
                    continue
                trial = information + np.outer(rows[idx], rows[idx])
                sign, value = np.linalg.slogdet(trial)
                if sign > 0.0 and value > best_score:
                    best_idx = idx
                    best_score = float(value)
            if best_idx < 0:
                # A requested spacing can make the final selection infeasible.
                # Fill deterministically rather than silently changing count.
                for candidate in positions:
                    if int(candidate) not in selected:
                        best_idx = int(np.where(positions == candidate)[0][0])
                        break
            if best_idx < 0:
                break
            selected.append(int(positions[best_idx]))
            information += np.outer(rows[best_idx], rows[best_idx])

        return np.asarray(sorted(selected), dtype=np.int64)

    def _pilot_symbols(self, spec: SignalSpec, count: int) -> np.ndarray:
        if count <= 0:
            return np.asarray([], dtype=np.complex128)

        constellation = get_constellation(spec.pilot_modulation).astype(np.complex128)
        mode = str(spec.pilot_symbol_mode).lower()
        if mode == "constant":
            idx = int(spec.pilot_symbol_index) % len(constellation)
            return np.full(count, constellation[idx], dtype=np.complex128)
        if mode == "pn":
            rng = np.random.default_rng(int(spec.pilot_seed))
            indices = rng.integers(0, len(constellation), size=count, endpoint=False)
            return constellation[indices].astype(np.complex128)
        raise ValueError(f"Unsupported pilot_symbol_mode: {spec.pilot_symbol_mode}")

    def generate_clean_baseband(self, spec: SignalSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate clean pulse-shaped complex baseband signal."""

        mod = Modulator(spec.modulation, rng=self.rng)
        symbols, indices = mod.modulate(spec.num_symbols)
        pilot_indices = self._pilot_indices(spec)
        pilot_symbols = self._pilot_symbols(spec, len(pilot_indices))
        if len(pilot_indices) > 0:
            symbols[pilot_indices] = pilot_symbols
            indices[pilot_indices] = -1

        x = pulse_shape_symbols(
            symbols,
            sps=spec.samples_per_symbol,
            beta=spec.rrc_beta,
            span=spec.rrc_span,
            mode="same",
        )
        x = normalize_power(x)

        if spec.target_num_samples is not None:
            x = trim_or_pad(x, spec.target_num_samples)

        return x.astype(np.complex128), indices, pilot_indices, pilot_symbols

    def generate(self, spec: SignalSpec, *, channel_rng: np.random.Generator | None = None,
                 noise_rng: np.random.Generator | None = None) -> GeneratedSignal:
        """Generate one labeled LEO signal example."""

        spec.modulation = spec.modulation.upper()

        x_clean, symbol_indices, pilot_indices, pilot_symbols = self.generate_clean_baseband(spec)

        channel = NonStationaryLEOChannel(
            sample_rate_hz=spec.sample_rate_hz,
            carrier_frequency_hz=spec.carrier_frequency_hz,
            rng=self.rng if channel_rng is None else channel_rng,
            noise_rng=noise_rng,
        )

        channel_cfg = ChannelConfig(
            sample_rate_hz=spec.sample_rate_hz,
            carrier_frequency_hz=spec.carrier_frequency_hz,
            snr_db=spec.snr_db,
            fd0_hz=spec.fd0_hz,
            mu_hz_per_s=spec.mu_hz_per_s,
            channel_type=spec.channel_type,
            compensate_constant_doppler=spec.compensate_constant_doppler,
            rician_k_db=spec.rician_k_db,
            multipath_delays=spec.multipath_delays,
            multipath_gains_db=spec.multipath_gains_db,
            multipath_doppler_hz=spec.multipath_doppler_hz,
            phase_noise_std_rad=spec.phase_noise_std_rad,
            phase_noise_mode=spec.phase_noise_mode,
        )

        y, channel_info = channel.apply(x_clean, channel_cfg)

        timing_offset = int(spec.timing_offset_samples)
        fractional_offset = float(spec.fractional_timing_offset_samples)
        total_timing_offset = timing_offset + fractional_offset
        if abs(total_timing_offset) >= len(y):
            raise ValueError("timing_offset_samples must be smaller than the frame length.")
        y = apply_fractional_delay(y, total_timing_offset)

        # After an integer shift, the phase law expressed in the received
        # frame-start coordinate has the same quadratic coefficient but a
        # different linear coefficient. This is the CFO reference recoverable
        # by a joint timing/CFO/Doppler estimator on the shifted frame.
        fd0_frame_start = float(
            spec.fd0_hz - spec.mu_hz_per_s * total_timing_offset / float(spec.sample_rate_hz)
        )

        iq = complex_to_iq(y, layout="channel_first")
        label = self.label_of(spec.modulation)

        info = {
            **spec.to_dict(),
            **channel_info,
            "label": int(label),
            "num_pilots": int(len(pilot_indices)),
            "timing_offset_samples": int(timing_offset),
            "fractional_timing_offset_samples": float(fractional_offset),
            "total_timing_offset_samples": float(total_timing_offset),
            "fd0_frame_start_hz": fd0_frame_start,
        }

        return GeneratedSignal(
            iq=iq,
            complex_signal=y.astype(np.complex128),
            label=label,
            modulation=spec.modulation,
            symbol_indices=symbol_indices,
            pilot_indices=pilot_indices.astype(np.int64),
            pilot_symbols=pilot_symbols.astype(np.complex64),
            info=info,
        )

    def random_spec(
        self,
        modulation: str | None = None,
        num_symbols: int = 256,
        samples_per_symbol: int = 8,
        sample_rate_hz: float = 1.0e6,
        carrier_frequency_hz: float = 2.0e9,
        snr_db_range: tuple[float, float] = (-10.0, 20.0),
        fd0_range_hz: tuple[float, float] = (-2.0e5, 2.0e5),
        mu_range_hz_per_s: tuple[float, float] = (-5.0e3, 5.0e3),
        channel_type: str = "awgn",
        target_num_samples: int | None = None,
    ) -> SignalSpec:
        """Sample a random SignalSpec from ranges."""

        if modulation is None:
            modulation = str(self.rng.choice(self.modulations))

        snr_db = float(self.rng.uniform(*snr_db_range))
        fd0_hz = float(self.rng.uniform(*fd0_range_hz))
        mu_hz_per_s = float(self.rng.uniform(*mu_range_hz_per_s))

        return SignalSpec(
            modulation=modulation,
            num_symbols=num_symbols,
            samples_per_symbol=samples_per_symbol,
            sample_rate_hz=sample_rate_hz,
            carrier_frequency_hz=carrier_frequency_hz,
            snr_db=snr_db,
            fd0_hz=fd0_hz,
            mu_hz_per_s=mu_hz_per_s,
            channel_type=channel_type,
            target_num_samples=target_num_samples,
        )

    def generate_batch(self, specs: list[SignalSpec]) -> dict[str, np.ndarray]:
        """Generate a batch and return arrays suitable for HDF5 saving."""

        samples = [self.generate(spec) for spec in specs]

        iq = np.stack([s.iq for s in samples], axis=0).astype(np.float32)
        label = np.array([s.label for s in samples], dtype=np.int64)

        snr_db = np.array([s.info["snr_db"] for s in samples], dtype=np.float32)
        fd0 = np.array([s.info["fd0_hz"] for s in samples], dtype=np.float32)
        mu = np.array([s.info["mu_hz_per_s"] for s in samples], dtype=np.float32)
        gamma = np.array([s.info["gamma"] for s in samples], dtype=np.float32)

        modulation = np.array([s.modulation for s in samples], dtype="S16")

        return {
            "iq": iq,
            "label": label,
            "snr_db": snr_db,
            "fd0": fd0,
            "mu": mu,
            "gamma": gamma,
            "modulation": modulation,
        }
