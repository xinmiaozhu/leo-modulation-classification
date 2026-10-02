"""Pilot-aided DFRFT/chirp-focus acquisition with coherent local refinement.

The estimator is intentionally modulation independent.  Known pilots are
removed first, a physically calibrated chirp-focus bank estimates Doppler rate
and the focused FFT peak estimates centered CFO, and the existing coherent
pilot likelihood refines both parameters locally.  No payload label, HOC, EVM,
or constellation candidate is used.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .pilot_estimator import PilotMuEstimator, PilotMuEstimatorConfig, PilotMuResult, make_grid


@dataclass
class HybridDFRFTConfig:
    coarse_mu_step_hz_per_s: float = 160.0
    fft_size: int = 2048
    fine_mu_radius_hz_per_s: float = 200.0
    fine_mu_step_hz_per_s: float = 5.0
    fine_fd0_radius_hz: float = 30.0
    fine_fd0_step_hz: float = 1.0
    peak_exclude_bins: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HybridDFRFTResult:
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


class HybridDFRFTPilotEstimator:
    """DFRFT-equivalent pilot chirp focusing followed by coherent refinement."""

    def __init__(
        self,
        pilot_config: PilotMuEstimatorConfig,
        hybrid_config: HybridDFRFTConfig | None = None,
    ) -> None:
        self.pilot_config = pilot_config
        self.hybrid_config = hybrid_config or HybridDFRFTConfig()
        if self.hybrid_config.coarse_mu_step_hz_per_s <= 0.0:
            raise ValueError("coarse_mu_step_hz_per_s must be positive.")
        if self.hybrid_config.fft_size < 8:
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
            self.hybrid_config.coarse_mu_step_hz_per_s,
        )
        fft_size = max(int(self.hybrid_config.fft_size), int(z.size))
        frequency = np.fft.fftshift(np.fft.fftfreq(fft_size, d=dt))
        dechirped = z[None, :] * np.exp(-1j * np.pi * mu_grid[:, None] * tau[None, :] ** 2)
        spectrum = np.fft.fftshift(np.fft.fft(dechirped, n=fft_size, axis=1), axes=1)
        power = np.abs(spectrum) ** 2 / max(float(z.size**2), 1.0)

        candidates: list[tuple[float, int, int]] = []
        for mu_index, mu in enumerate(mu_grid):
            lower = float(self.pilot_config.fd0_min_hz) + float(mu) * center
            upper = float(self.pilot_config.fd0_max_hz) + float(mu) * center
            lo, hi = min(lower, upper), max(lower, upper)
            valid_bins = np.flatnonzero((frequency >= lo) & (frequency <= hi))
            if valid_bins.size == 0:
                continue
            local_index = int(valid_bins[np.argmax(power[mu_index, valid_bins])])
            candidates.append((float(power[mu_index, local_index]), int(mu_index), local_index))
        if not candidates:
            return np.nan, np.nan, np.nan, -np.inf, np.nan, float(abs(frequency[1] - frequency[0])), int(mu_grid.size)

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_mu_index, best_frequency_index = candidates[0]
        exclusion = max(0, int(self.hybrid_config.peak_exclude_bins))
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
    ) -> HybridDFRFTResult:
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
        if not np.isfinite(coarse_score):
            invalid = PilotMuResult(np.nan, np.nan, float(coarse_score), np.nan, best_timing, int(best_z.size), False)
            return HybridDFRFTResult(
                invalid,
                float(coarse_mu),
                float(centered_cfo),
                float(coarse_fd0),
                float(coarse_score),
                float(coarse_margin),
                float(bin_spacing),
                int(num_mu),
            )

        mu_radius = max(0.0, float(self.hybrid_config.fine_mu_radius_hz_per_s))
        fine_mu = make_grid(
            max(self.pilot_config.mu_min, coarse_mu - mu_radius),
            min(self.pilot_config.mu_max, coarse_mu + mu_radius),
            self.hybrid_config.fine_mu_step_hz_per_s,
        )
        fd_radius = max(0.0, float(self.hybrid_config.fine_fd0_radius_hz))
        fine_fd0 = make_grid(
            max(self.pilot_config.fd0_min_hz, coarse_fd0 - fd_radius),
            min(self.pilot_config.fd0_max_hz, coarse_fd0 + fd_radius),
            self.hybrid_config.fine_fd0_step_hz,
        )
        mu_hat, fd0_hat, score, margin = self.pilot._score_grid(
            best_z,
            best_t,
            fine_mu,
            fine_fd0,
        )
        valid = bool(np.isfinite(score))
        estimate = PilotMuResult(
            float(mu_hat),
            float(fd0_hat),
            float(score),
            float(margin),
            best_timing,
            int(best_z.size),
            valid,
        )
        return HybridDFRFTResult(
            estimate,
            float(coarse_mu),
            float(centered_cfo),
            float(coarse_fd0),
            float(coarse_score),
            float(coarse_margin),
            float(bin_spacing),
            int(num_mu),
        )
