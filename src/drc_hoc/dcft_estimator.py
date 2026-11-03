"""Pilot-aided DCFT chirp-focus acquisition.

The estimator is intentionally modulation independent.  Known pilots are
removed first, a physically calibrated chirp-focus bank estimates Doppler rate
and the focused FFT peak estimates centered CFO.  No payload label, HOC, EVM,
or constellation candidate is used.

The acquisition peak supplies the final rate and frame-start CFO estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .pilot_estimator import PilotMuEstimator, PilotMuEstimatorConfig, PilotMuResult, make_grid


@dataclass
class DCFTConfig:
    coarse_mu_step_hz_per_s: float = 160.0
    fft_size: int = 2048
    peak_exclude_bins: int = 2
    unwrap_centered_cfo: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DCFTResult:
    estimate: PilotMuResult
    coarse_mu_hz_per_s: float
    coarse_centered_cfo_hz: float
    coarse_fd0_hz: float
    coarse_score: float
    coarse_margin_rel: float
    fft_bin_spacing_hz: float
    num_coarse_mu: int

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["estimate"] = self.estimate.to_dict()
        return payload


class DCFTPilotEstimator:
    """DCFT acquisition on a physically bounded rate/frequency grid.
    """

    def __init__(
        self,
        pilot_config: PilotMuEstimatorConfig,
        dcft_config: DCFTConfig | None = None,
    ) -> None:
        self.pilot_config = pilot_config
        self.dcft_config = dcft_config or DCFTConfig()
        if self.dcft_config.coarse_mu_step_hz_per_s <= 0.0:
            raise ValueError("coarse_mu_step_hz_per_s must be positive.")
        if self.dcft_config.fft_size < 8:
            raise ValueError("fft_size must be at least 8.")
        self.pilot = PilotMuEstimator(pilot_config)

    def _chirp_focus(
        self,
        z: np.ndarray,
        t: np.ndarray,
    ) -> tuple[float, float, float, float, float, float, int]:
        """Return coarse rate, centered CFO, frame-start CFO, score, margin, df."""

        if z.size < self.pilot_config.min_pilots:
            return np.nan, np.nan, np.nan, -np.inf, np.nan, np.nan, 0
        differences = np.diff(t)
        if differences.size == 0:
            return np.nan, np.nan, np.nan, -np.inf, np.nan, np.nan, 0
        dt = float(np.median(differences))
        tolerance = max(1e-12, abs(dt) * 1e-6)
        if dt <= 0.0 or not np.all(np.abs(differences - dt) <= tolerance):
            raise ValueError("Hybrid DFRFT acquisition requires uniformly spaced comb pilots.")

        center = float(np.mean(t))
        tau = t - center
        mu_grid = make_grid(
            self.pilot_config.mu_min,
            self.pilot_config.mu_max,
            self.dcft_config.coarse_mu_step_hz_per_s,
        )
        fft_size = max(int(self.dcft_config.fft_size), int(z.size))
        frequency = np.fft.fftshift(np.fft.fftfreq(fft_size, d=dt))
        dechirped = z[None, :] * np.exp(-1j * np.pi * mu_grid[:, None] * tau[None, :] ** 2)
        spectrum = np.fft.fftshift(np.fft.fft(dechirped, n=fft_size, axis=1), axes=1)
        power = np.abs(spectrum) ** 2 / max(float(z.size**2), 1.0)

        candidates: list[tuple[float, int, int]] = []
        for mu_index, mu in enumerate(mu_grid):
            lower = float(self.pilot_config.fd0_min_hz) + float(mu) * center
            upper = float(self.pilot_config.fd0_max_hz) + float(mu) * center
            lo, hi = min(lower, upper), max(lower, upper)
            if self.dcft_config.unwrap_centered_cfo:
                period = 1.0 / dt
                if hi - lo >= period:
                    raise ValueError("CFO search width must be below the comb-pilot sampling rate.")
                lifted = lo + np.remainder(frequency - lo, period)
                valid_bins = np.flatnonzero(lifted <= hi)
            else:
                valid_bins = np.flatnonzero((frequency >= lo) & (frequency <= hi))
            if valid_bins.size == 0:
                continue
            local_index = int(valid_bins[np.argmax(power[mu_index, valid_bins])])
            candidates.append((float(power[mu_index, local_index]), int(mu_index), local_index))
        if not candidates:
            return np.nan, np.nan, np.nan, -np.inf, np.nan, float(abs(frequency[1] - frequency[0])), int(mu_grid.size)

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_mu_index, best_frequency_index = candidates[0]
        exclusion = max(0, int(self.dcft_config.peak_exclude_bins))
        second_score = -np.inf
        for score, mu_index, frequency_index in candidates[1:]:
            if (
                abs(mu_index - best_mu_index) > 1
                or abs(frequency_index - best_frequency_index) > exclusion
            ):
                second_score = float(score)
                break
        margin = (
            (best_score - second_score) / (abs(second_score) + 1e-12)
            if np.isfinite(second_score)
            else np.inf
        )
        coarse_mu = float(mu_grid[best_mu_index])
        centered_cfo = float(frequency[best_frequency_index])
        if self.dcft_config.unwrap_centered_cfo:
            lower = float(self.pilot_config.fd0_min_hz) + coarse_mu * center
            centered_cfo = lower + float(np.remainder(centered_cfo - lower, 1.0 / dt))
        frame_start_cfo = centered_cfo - coarse_mu * center
        return (
            coarse_mu,
            centered_cfo,
            frame_start_cfo,
            float(best_score),
            float(max(margin, 0.0)),
            float(abs(frequency[1] - frequency[0])),
            int(mu_grid.size),
        )

    def estimate(
        self,
        iq: np.ndarray,
        pilot_indices: np.ndarray,
        pilot_symbols: np.ndarray,
    ) -> DCFTResult:
        best_payload: tuple | None = None
        best_timing = 0
        best_z = np.asarray([], dtype=np.complex128)
        best_t = np.asarray([], dtype=np.float64)
        for timing in self.pilot.timing_phases:
            z, t = self.pilot._extract_pilot_observations(
                iq,
                pilot_indices,
                pilot_symbols,
                int(timing),
            )
            payload = self._chirp_focus(z, t)
            if best_payload is None or float(payload[3]) > float(best_payload[3]):
                best_payload = payload
                best_timing = int(timing)
                best_z = z
                best_t = t

        assert best_payload is not None
        coarse_mu, centered_cfo, coarse_fd0, coarse_score, coarse_margin, bin_spacing, num_mu = best_payload
        acquisition = PilotMuResult(
            float(coarse_mu), float(coarse_fd0), float(coarse_score),
            float(coarse_margin), best_timing, int(best_z.size),
            bool(np.isfinite(coarse_score)),
        )
        return DCFTResult(
            acquisition,
            float(coarse_mu),
            float(centered_cfo),
            float(coarse_fd0),
            float(coarse_score),
            float(coarse_margin),
            float(bin_spacing),
            int(num_mu),
        )


@dataclass
class FrFTCaptureConfig:
    angle_count: int = 65
    output_size: int = 1024


class FrFTPilotEstimator(DCFTPilotEstimator):
    """Physically bounded FrFT angle acquisition.

    Motivated by Luo et al., IEEE Access 2019, DOI 10.1109/ACCESS.2019.2903185.
    Adaptation: known AMC comb pilots replace GNSS correlations; angle sampling
    is validation-selected. This implementation does not include
    the paper's NLS/alternate-search extension. Not an exact paper reproduction.
    """

    def __init__(self, pilot_config, dcft_config=None, frft_config=None):
        super().__init__(pilot_config, dcft_config)
        self.frft_config = frft_config or FrFTCaptureConfig()
        if self.frft_config.angle_count < 2:
            raise ValueError("At least two FrFT angles required")
        self._plans = {}

    def _chirp_focus(self, z, t):
        from .frft import SampledFrFTPlan, rate_to_frft_angle
        z, t = np.asarray(z), np.asarray(t, dtype=float)
        if len(z) < self.pilot_config.min_pilots or t.shape != z.shape:
            raise ValueError("Insufficient or inconsistent pilot observations")
        dt = float(np.median(np.diff(t)))
        if dt <= 0 or not np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-12):
            raise ValueError("FrFT capture requires uniformly spaced pilots")
        n, center = len(z), float(np.mean(t))
        key = (n, round(dt, 14))
        if key not in self._plans:
            bounds = rate_to_frft_angle([self.pilot_config.mu_min, self.pilot_config.mu_max], n, dt)
            angles = np.linspace(bounds[0], bounds[1], self.frft_config.angle_count)
            self._plans[key] = [(a, SampledFrFTPlan(n, float(a), self.frft_config.output_size)) for a in angles]
        candidates = []
        for alpha, plan in self._plans[key]:
            mu = float(-1 / np.tan(alpha) / (n * dt**2))
            # Floating-point endpoint rounding must not escape physical bounds.
            mu = float(np.clip(mu, self.pilot_config.mu_min, self.pilot_config.mu_max))
            frequency = plan.u / (np.sin(alpha) * np.sqrt(n) * dt)
            fd0 = frequency - mu * center
            valid = np.flatnonzero((fd0 >= self.pilot_config.fd0_min_hz) & (fd0 <= self.pilot_config.fd0_max_hz))
            if not valid.size:
                continue
            spectrum = plan.transform(z)
            # Remove the kernel's angle-dependent amplitude scale for fair peaks.
            power = np.abs(spectrum)**2 * abs(np.sin(alpha)) / n
            k = int(valid[np.argmax(power[valid])])
            candidates.append((float(power[k]), mu, float(frequency[k]), float(fd0[k]),
                               float(abs(frequency[1] - frequency[0]))))
        if not candidates:
            raise ValueError("FrFT output grid does not intersect the physical CFO range")
        candidates.sort(reverse=True)
        score, mu, fc, fd0, df = candidates[0]
        margin = (score - candidates[1][0]) / (abs(candidates[1][0]) + 1e-12) if len(candidates) > 1 else np.inf
        return mu, fc, fd0, score, margin, df, len(self._plans[key])
