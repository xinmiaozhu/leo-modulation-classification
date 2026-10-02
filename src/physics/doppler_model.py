"""Doppler and Doppler-rate modeling for LEO signal simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .gamma_metric import compute_gamma


@dataclass
class DopplerState:
    """Doppler parameters for a signal window."""

    fd0_hz: float
    mu_hz_per_s: float
    gamma: float
    observation_time_s: float
    carrier_frequency_hz: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "fd0_hz": float(self.fd0_hz),
            "mu_hz_per_s": float(self.mu_hz_per_s),
            "gamma": float(self.gamma),
            "observation_time_s": float(self.observation_time_s),
            "carrier_frequency_hz": float(self.carrier_frequency_hz),
        }


class DopplerModel:
    """Doppler shift/rate model.

    Sign convention:
        fd = -fc * range_rate / c.
        If range is increasing, received frequency decreases.
    """

    def __init__(self, carrier_frequency_hz: float, c: float = 299_792_458.0) -> None:
        self.fc = float(carrier_frequency_hz)
        self.c = float(c)

    def doppler_shift_from_range_rate(self, range_rate_mps: float | np.ndarray):
        """Compute Doppler shift from slant range rate."""

        return -self.fc * np.asarray(range_rate_mps) / self.c

    def doppler_rate_from_range_acceleration(self, range_acc_mps2: float | np.ndarray):
        """Compute Doppler rate from slant range acceleration."""

        return -self.fc * np.asarray(range_acc_mps2) / self.c

    def phase(self, t_s: np.ndarray, fd0_hz: float, mu_hz_per_s: float) -> np.ndarray:
        """Compute Doppler phase 2*pi*fd0*t + pi*mu*t^2."""

        t_s = np.asarray(t_s, dtype=np.float64)
        return 2.0 * np.pi * fd0_hz * t_s + np.pi * mu_hz_per_s * t_s**2

    def complex_exponential(
        self,
        t_s: np.ndarray,
        fd0_hz: float,
        mu_hz_per_s: float,
    ) -> np.ndarray:
        """Return exp(j * phase)."""

        return np.exp(1j * self.phase(t_s, fd0_hz, mu_hz_per_s))

    def apply_to_signal(
        self,
        x: np.ndarray,
        sample_rate_hz: float,
        fd0_hz: float,
        mu_hz_per_s: float,
    ) -> np.ndarray:
        """Apply Doppler shift and Doppler-rate phase distortion to a signal."""

        x = np.asarray(x)
        n = np.arange(len(x), dtype=np.float64)
        t = n / float(sample_rate_hz)
        return x * self.complex_exponential(t, fd0_hz, mu_hz_per_s)

    def remove_constant_doppler(
        self,
        x: np.ndarray,
        sample_rate_hz: float,
        fd0_hz: float,
    ) -> np.ndarray:
        """Remove only the constant Doppler/CFO term."""

        x = np.asarray(x)
        n = np.arange(len(x), dtype=np.float64)
        t = n / float(sample_rate_hz)
        return x * np.exp(-1j * 2.0 * np.pi * fd0_hz * t)

    def make_state(
        self,
        fd0_hz: float,
        mu_hz_per_s: float,
        observation_time_s: float,
    ) -> DopplerState:
        """Create a DopplerState object."""

        gamma = compute_gamma(mu_hz_per_s, observation_time_s)
        return DopplerState(
            fd0_hz=float(fd0_hz),
            mu_hz_per_s=float(mu_hz_per_s),
            gamma=float(gamma),
            observation_time_s=float(observation_time_s),
            carrier_frequency_hz=float(self.fc),
        )

    def local_linear_fit(
        self,
        times_s: np.ndarray,
        doppler_hz: np.ndarray,
        center_time_s: float | None = None,
        window_s: float | None = None,
    ) -> tuple[float, float]:
        """Fit f_D(t) ~= fd0 + mu * tau over a local time window.

        Args:
            times_s: Time samples.
            doppler_hz: Doppler shift samples.
            center_time_s: Center of local window. If None, use median time.
            window_s: Optional fitting window length.

        Returns:
            fd0_hz, mu_hz_per_s, where tau = t - center_time.
        """

        times_s = np.asarray(times_s, dtype=np.float64)
        doppler_hz = np.asarray(doppler_hz, dtype=np.float64)

        if center_time_s is None:
            center_time_s = float(np.median(times_s))

        if window_s is not None:
            half = window_s / 2.0
            mask = np.abs(times_s - center_time_s) <= half
            if np.sum(mask) >= 2:
                times_fit = times_s[mask]
                doppler_fit = doppler_hz[mask]
            else:
                times_fit = times_s
                doppler_fit = doppler_hz
        else:
            times_fit = times_s
            doppler_fit = doppler_hz

        tau = times_fit - center_time_s
        coeff = np.polyfit(tau, doppler_fit, deg=1)
        mu = coeff[0]
        fd0 = coeff[1]
        return float(fd0), float(mu)


def random_doppler_state(
    carrier_frequency_hz: float,
    observation_time_s: float,
    fd0_range_hz: tuple[float, float] = (-2e5, 2e5),
    mu_range_hz_per_s: tuple[float, float] = (-5e3, 5e3),
    rng: np.random.Generator | None = None,
) -> DopplerState:
    """Sample a random Doppler state from user-specified ranges."""

    if rng is None:
        rng = np.random.default_rng()

    fd0 = rng.uniform(fd0_range_hz[0], fd0_range_hz[1])
    mu = rng.uniform(mu_range_hz_per_s[0], mu_range_hz_per_s[1])

    return DopplerModel(carrier_frequency_hz).make_state(fd0, mu, observation_time_s)
