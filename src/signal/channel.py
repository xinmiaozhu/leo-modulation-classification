"""LEO non-stationary channel and impairment models."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from src.physics.doppler_model import DopplerModel
from src.physics.gamma_metric import compute_gamma
from src.utils.math_utils import add_awgn_complex, normalize_power
from src.signal.impairments import apply_phase_noise


@dataclass
class ChannelConfig:
    """Channel configuration for synthetic LEO signal generation."""

    sample_rate_hz: float = 1.0e6
    carrier_frequency_hz: float = 2.0e9
    snr_db: float = 10.0
    fd0_hz: float = 0.0
    mu_hz_per_s: float = 0.0
    normalize_input_power: bool = True
    compensate_constant_doppler: bool = False

    # Optional multipath/Rician controls
    channel_type: str = "awgn"  # "awgn", "rician", "multipath"
    rician_k_db: float = 10.0
    multipath_delays: tuple[int, ...] = (0, 3, 7)
    multipath_gains_db: tuple[float, ...] = (0.0, -6.0, -10.0)
    multipath_doppler_hz: tuple[float, ...] = ()
    phase_noise_std_rad: float = 0.0
    phase_noise_mode: str = "random_walk"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NonStationaryLEOChannel:
    """Apply Doppler-rate distortion and additive channel effects."""

    def __init__(
        self,
        sample_rate_hz: float,
        carrier_frequency_hz: float,
        rng: np.random.Generator | None = None,
        noise_rng: np.random.Generator | None = None,
    ) -> None:
        self.fs = float(sample_rate_hz)
        self.fc = float(carrier_frequency_hz)
        self.rng = np.random.default_rng() if rng is None else rng
        self.noise_rng = self.rng if noise_rng is None else noise_rng
        self.doppler = DopplerModel(carrier_frequency_hz)

    def apply_doppler_rate(self, x: np.ndarray, fd0_hz: float, mu_hz_per_s: float) -> np.ndarray:
        return self.doppler.apply_to_signal(x, self.fs, fd0_hz, mu_hz_per_s)

    def remove_constant_doppler(self, x: np.ndarray, fd0_hz: float) -> np.ndarray:
        return self.doppler.remove_constant_doppler(x, self.fs, fd0_hz)

    def add_awgn(self, x: np.ndarray, snr_db: float) -> np.ndarray:
        return add_awgn_complex(x, snr_db, self.noise_rng)

    def apply_time_varying_multipath(
        self, x: np.ndarray, delays: tuple[int, ...],
        gains_db: tuple[float, ...], doppler_hz: tuple[float, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sparse specular paths h_l[n] = c_l exp(j 2 pi nu_l n/fs).

        Dopplers are relative to the common carrier law applied before this
        operator. This is a controlled time-varying model, not Jakes fading.
        Returned dense taps describe n=0; frequencies are stored separately.
        """
        if not delays or len(delays) != len(gains_db) or len(delays) != len(doppler_hz):
            raise ValueError("Each path needs a delay, gain and Doppler.")
        if len(set(delays)) != len(delays) or any(int(d) != d or d < 0 for d in delays):
            raise ValueError("Path delays must be distinct non-negative integers.")
        if not np.all(np.isfinite([*gains_db, *doppler_hz])):
            raise ValueError("Path gains and Dopplers must be finite.")
        y = np.zeros(len(x), dtype=np.complex128)
        taps = np.zeros(max(delays) + 1, dtype=np.complex128)
        time = np.arange(len(x)) / self.fs
        for delay, gain, nu in zip(delays, gains_db, doppler_hz):
            coeff = 10 ** (gain / 20) * np.exp(1j * self.rng.uniform(0, 2 * np.pi))
            taps[delay] = coeff
            if delay < len(x):
                y[delay:] += coeff * np.exp(2j * np.pi * nu * time[delay:]) * x[:len(x)-delay]
        return y, taps

    def apply_static_multipath(
        self,
        x: np.ndarray,
        delays: tuple[int, ...] = (0, 3, 7),
        gains_db: tuple[float, ...] = (0.0, -6.0, -10.0),
        random_phase: bool = True,
        return_taps: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Apply a simple discrete-time static multipath channel."""

        if len(delays) != len(gains_db):
            raise ValueError("delays and gains_db must have the same length.")

        max_delay = max(delays)
        y = np.zeros(len(x) + max_delay, dtype=np.complex128)

        taps = np.zeros(max_delay + 1, dtype=np.complex128)
        for delay, gain_db in zip(delays, gains_db):
            amp = 10 ** (gain_db / 20.0)
            phase = self.rng.uniform(0.0, 2.0 * np.pi) if random_phase else 0.0
            coeff = amp * np.exp(1j * phase)
            taps[int(delay)] += coeff
            y[delay : delay + len(x)] += coeff * x

        truncated = y[: len(x)]
        return (truncated, taps) if return_taps else truncated

    def apply_rician(self, x: np.ndarray, k_db: float = 10.0) -> np.ndarray:
        """Apply a flat Rician fading coefficient."""

        K = 10 ** (k_db / 10.0)
        los = np.sqrt(K / (K + 1.0))
        scatter_scale = np.sqrt(1.0 / (K + 1.0))
        scatter = scatter_scale * (
            self.rng.normal(0.0, 1.0 / np.sqrt(2.0))
            + 1j * self.rng.normal(0.0, 1.0 / np.sqrt(2.0))
        )
        h = los + scatter
        return h * x

    def apply(self, x: np.ndarray, config: ChannelConfig) -> tuple[np.ndarray, dict[str, float]]:
        """Apply channel effects according to config.

        Returns:
            y: Received complex signal.
            info: Dictionary containing fd0, mu, gamma and SNR metadata.
        """

        y = np.asarray(x, dtype=np.complex128)

        if config.normalize_input_power:
            y = normalize_power(y)

        y = self.apply_doppler_rate(y, config.fd0_hz, config.mu_hz_per_s)

        if config.compensate_constant_doppler:
            y = self.remove_constant_doppler(y, config.fd0_hz)

        channel_type = config.channel_type.lower()
        channel_taps = np.asarray([1.0 + 0.0j], dtype=np.complex128)
        channel_delays = np.asarray([0], dtype=np.int64)
        if channel_type == "awgn":
            pass
        elif channel_type == "rician":
            y = self.apply_rician(y, k_db=config.rician_k_db)
        elif channel_type == "time_varying_multipath":
            y, channel_taps = self.apply_time_varying_multipath(
                y, config.multipath_delays, config.multipath_gains_db,
                config.multipath_doppler_hz,
            )
            channel_delays = np.asarray(config.multipath_delays, dtype=np.int64)
        elif channel_type == "multipath":
            y, channel_taps = self.apply_static_multipath(
                y,
                delays=config.multipath_delays,
                gains_db=config.multipath_gains_db,
                return_taps=True,
            )
            channel_delays = np.asarray(config.multipath_delays, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported channel_type: {config.channel_type}")

        y = apply_phase_noise(
            y,
            std_rad=float(config.phase_noise_std_rad),
            rng=self.rng,
            mode=str(config.phase_noise_mode),
        )

        y = self.add_awgn(y, config.snr_db)

        T = len(y) / self.fs
        gamma = compute_gamma(config.mu_hz_per_s, T)

        info = {
            "fd0_hz": float(config.fd0_hz),
            "mu_hz_per_s": float(config.mu_hz_per_s),
            "gamma": float(gamma),
            "snr_db": float(config.snr_db),
            "sample_rate_hz": float(self.fs),
            "carrier_frequency_hz": float(self.fc),
            "observation_time_s": float(T),
            "phase_noise_std_rad": float(config.phase_noise_std_rad),
            "phase_noise_mode": str(config.phase_noise_mode),
            "channel_taps": channel_taps,
            "channel_delays": channel_delays,
            "channel_doppler_hz": np.asarray(
                config.multipath_doppler_hz if channel_type == "time_varying_multipath"
                else np.zeros(len(channel_delays)), dtype=float,
            ),
        }

        return y, info
