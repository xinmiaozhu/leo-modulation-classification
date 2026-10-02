"""Pilot-aided Doppler-rate estimation."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from src.signal.pulse_shape import rrc_filter


@dataclass
class PilotMuEstimatorConfig:
    sample_rate_hz: float = 200_000.0
    samples_per_symbol: int = 8
    rrc_beta: float = 0.35
    rrc_span: int = 8
    timing_phases: tuple[int, ...] = (0,)
    mu_min: float = -8160.0
    mu_max: float = -180.0
    coarse_step_hz_per_s: float = 40.0
    fine_radius_hz_per_s: float = 120.0
    fine_step_hz_per_s: float = 5.0
    fd0_min_hz: float = 0.0
    fd0_max_hz: float = 0.0
    fd0_step_hz: float = 100.0
    fine_fd0_radius_hz: float | None = None
    fine_fd0_step_hz: float | None = None
    phase_only: bool = False
    pilot_weighting: str | None = None
    soft_weight_scale: float = 1.0
    amplitude_threshold_rel: float = 0.35
    min_pilots: int = 8
    center_time: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PilotMuResult:
    mu_hat_hz_per_s: float
    fd0_hat_hz: float
    score: float
    score_margin_rel: float
    timing: int
    num_pilots_used: int
    valid: bool

    def to_dict(self) -> dict:
        return asdict(self)


def iq_to_complex(iq: np.ndarray) -> np.ndarray:
    arr = np.asarray(iq)
    if arr.ndim != 2:
        raise ValueError(f"Expected IQ frame with shape [2, N] or [N, 2], got {arr.shape}.")
    if arr.shape[0] == 2:
        return arr[0].astype(np.float64) + 1j * arr[1].astype(np.float64)
    if arr.shape[1] == 2:
        return arr[:, 0].astype(np.float64) + 1j * arr[:, 1].astype(np.float64)
    raise ValueError(f"Expected IQ frame with shape [2, N] or [N, 2], got {arr.shape}.")


def make_grid(min_value: float, max_value: float, step: float) -> np.ndarray:
    lo = float(min(min_value, max_value))
    hi = float(max(min_value, max_value))
    step = abs(float(step))
    if step <= 0.0 or np.isclose(lo, hi):
        return np.asarray([lo], dtype=np.float64)
    grid = np.arange(lo, hi + 0.5 * step, step, dtype=np.float64)
    if grid[-1] < hi:
        grid = np.concatenate([grid, np.asarray([hi], dtype=np.float64)])
    return np.unique(np.clip(grid, lo, hi))


class PilotMuEstimator:
    def __init__(self, config: PilotMuEstimatorConfig) -> None:
        self.config = config
        self.fs = float(config.sample_rate_hz)
        self.sps = max(int(config.samples_per_symbol), 1)
        self.rrc = rrc_filter(beta=config.rrc_beta, span=config.rrc_span, sps=self.sps)
        self.timing_phases = tuple(int(x) for x in config.timing_phases) or (0,)
        mode = config.pilot_weighting
        if mode is None:
            mode = "phase" if config.phase_only else "coherent"
        mode = str(mode).lower()
        if mode not in {"phase", "coherent", "soft", "thresholded_phase"}:
            raise ValueError(
                "pilot_weighting must be 'phase', 'coherent', 'soft', or "
                "'thresholded_phase'."
            )
        if config.soft_weight_scale <= 0.0:
            raise ValueError("soft_weight_scale must be positive.")
        if config.amplitude_threshold_rel < 0.0:
            raise ValueError("amplitude_threshold_rel must be nonnegative.")
        self.pilot_weighting = mode

    def _extract_pilot_observations(
        self,
        iq: np.ndarray,
        pilot_indices: np.ndarray,
        pilot_symbols: np.ndarray,
        timing: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = iq_to_complex(iq)
        matched = np.convolve(x, self.rrc, mode="same")

        sym_idx = np.asarray(pilot_indices, dtype=np.int64).reshape(-1)
        symbols = np.asarray(pilot_symbols, dtype=np.complex128).reshape(-1)
        if len(sym_idx) != len(symbols):
            raise ValueError("pilot_indices and pilot_symbols must have the same length.")

        sample_idx = sym_idx * self.sps + int(timing)
        valid = (sample_idx >= 0) & (sample_idx < len(matched)) & (np.abs(symbols) > 1e-12)
        sample_idx = sample_idx[valid]
        symbols = symbols[valid]
        if sample_idx.size == 0:
            return np.asarray([], dtype=np.complex128), np.asarray([], dtype=np.float64)

        z = matched[sample_idx] * np.conj(symbols) / (np.abs(symbols) ** 2 + 1e-12)
        magnitude = np.abs(z)
        if self.pilot_weighting == "phase":
            z = z / (magnitude + 1e-12)
        elif self.pilot_weighting == "soft":
            # Bounded reliability weights suppress noise-dominated pilot phases
            # without allowing a few large-amplitude samples to dominate.
            reference = max(float(np.median(magnitude)), 1e-12)
            weights = np.minimum(
                magnitude / (float(self.config.soft_weight_scale) * reference),
                1.0,
            )
            z = weights * z / (magnitude + 1e-12)
        elif self.pilot_weighting == "thresholded_phase":
            reference = max(float(np.median(magnitude)), 1e-12)
            keep = magnitude >= float(self.config.amplitude_threshold_rel) * reference
            z = z[keep] / (magnitude[keep] + 1e-12)
            sample_idx = sample_idx[keep]
        t = sample_idx.astype(np.float64) / self.fs
        return z.astype(np.complex128), t

    def _score_grid(
        self,
        z: np.ndarray,
        t: np.ndarray,
        mu_grid: np.ndarray,
        fd0_grid: np.ndarray,
    ) -> tuple[float, float, float, float]:
        if z.size < self.config.min_pilots:
            return np.nan, np.nan, -np.inf, np.nan

        best_score = -np.inf
        second_score = -np.inf
        best_mu = np.nan
        best_fd0 = np.nan
        t2 = t**2
        for fd0 in fd0_grid:
            linear = np.exp(-1j * 2.0 * np.pi * float(fd0) * t)
            phase = np.exp(-1j * np.pi * mu_grid[:, None] * t2[None, :])
            accum = phase @ (z * linear)
            scores = (np.abs(accum) ** 2) / max(float(z.size**2), 1.0)
            order = np.argsort(scores)
            idx = int(order[-1])
            score = float(scores[idx])
            local_second = float(scores[int(order[-2])]) if len(order) > 1 else -np.inf
            if score > best_score:
                second_score = max(second_score, best_score, local_second)
                best_score = score
                best_mu = float(mu_grid[idx])
                best_fd0 = float(fd0)
            else:
                second_score = max(second_score, score, local_second)

        margin = (best_score - second_score) / (abs(second_score) + 1e-12) if np.isfinite(second_score) else np.inf
        return best_mu, best_fd0, best_score, float(max(margin, 0.0))

    def _centered_fd_grid(self, t: np.ndarray, step: float) -> tuple[np.ndarray, float]:
        """Build an instantaneous-frequency grid at the pilot-time centroid.

        Centering makes the linear and quadratic phase coordinates nearly
        orthogonal.  The configured CFO bounds remain frame-start bounds; the
        returned grid covers their image under f_c = f_0 + mu * t_c.
        """
        center = float(np.mean(t)) if t.size else 0.0
        corners = np.asarray(
            [
                fd0 + mu * center
                for fd0 in (self.config.fd0_min_hz, self.config.fd0_max_hz)
                for mu in (self.config.mu_min, self.config.mu_max)
            ],
            dtype=np.float64,
        )
        return make_grid(float(np.min(corners)), float(np.max(corners)), step), center

    def estimate(
        self,
        iq: np.ndarray,
        pilot_indices: np.ndarray,
        pilot_symbols: np.ndarray,
    ) -> PilotMuResult:
        mu_grid = make_grid(
            self.config.mu_min,
            self.config.mu_max,
            self.config.coarse_step_hz_per_s,
        )
        best: PilotMuResult | None = None
        best_center = 0.0
        for timing in self.timing_phases:
            z, t = self._extract_pilot_observations(iq, pilot_indices, pilot_symbols, int(timing))
            if self.config.center_time:
                fd0_grid, center = self._centered_fd_grid(t, self.config.fd0_step_hz)
                score_t = t - center
            else:
                fd0_grid = make_grid(
                    self.config.fd0_min_hz,
                    self.config.fd0_max_hz,
                    self.config.fd0_step_hz,
                )
                center = 0.0
                score_t = t
            mu_hat, fd0_hat, score, margin = self._score_grid(z, score_t, mu_grid, fd0_grid)
            if not np.isfinite(score):
                result = PilotMuResult(np.nan, np.nan, float(score), np.nan, int(timing), int(z.size), False)
            else:
                result = PilotMuResult(float(mu_hat), float(fd0_hat), float(score), float(margin), int(timing), int(z.size), True)
            if best is None or result.score > best.score:
                best = result
                best_center = center

        assert best is not None
        if not best.valid:
            return best

        mu_radius = max(0.0, float(self.config.fine_radius_hz_per_s))
        mu_step = max(0.0, float(self.config.fine_step_hz_per_s))
        if mu_radius > 0.0 and mu_step > 0.0:
            fine_mu = make_grid(
                max(self.config.mu_min, best.mu_hat_hz_per_s - mu_radius),
                min(self.config.mu_max, best.mu_hat_hz_per_s + mu_radius),
                mu_step,
            )
        else:
            fine_mu = np.asarray([best.mu_hat_hz_per_s], dtype=np.float64)

        fd0_radius = self.config.fine_fd0_radius_hz
        fd0_step = self.config.fine_fd0_step_hz
        if self.config.center_time:
            centered_full, _ = self._centered_fd_grid(
                np.asarray([best_center], dtype=np.float64),
                self.config.fd0_step_hz,
            )
            search_fd_min = float(centered_full[0])
            search_fd_max = float(centered_full[-1])
        else:
            search_fd_min = float(self.config.fd0_min_hz)
            search_fd_max = float(self.config.fd0_max_hz)
        if fd0_radius is not None and float(fd0_radius) > 0.0:
            resolved_fd0_step = float(fd0_step) if fd0_step is not None else float(self.config.fd0_step_hz)
            if resolved_fd0_step > 0.0:
                fine_fd0 = make_grid(
                    max(search_fd_min, best.fd0_hat_hz - float(fd0_radius)),
                    min(search_fd_max, best.fd0_hat_hz + float(fd0_radius)),
                    resolved_fd0_step,
                )
            else:
                fine_fd0 = np.asarray([best.fd0_hat_hz], dtype=np.float64)
        else:
            fine_fd0 = np.asarray([best.fd0_hat_hz], dtype=np.float64)

        if fine_mu.size == 1 and fine_fd0.size == 1:
            if self.config.center_time and best.valid:
                return PilotMuResult(
                    best.mu_hat_hz_per_s,
                    best.fd0_hat_hz - best.mu_hat_hz_per_s * best_center,
                    best.score,
                    best.score_margin_rel,
                    best.timing,
                    best.num_pilots_used,
                    True,
                )
            return best
        z, t = self._extract_pilot_observations(iq, pilot_indices, pilot_symbols, best.timing)
        score_t = t - best_center if self.config.center_time else t
        mu_hat, fd0_hat, score, margin = self._score_grid(z, score_t, fine_mu, fine_fd0)
        if np.isfinite(score) and score >= best.score:
            best = PilotMuResult(float(mu_hat), float(fd0_hat), float(score), float(margin), best.timing, int(z.size), True)
        if self.config.center_time and best.valid:
            return PilotMuResult(
                best.mu_hat_hz_per_s,
                best.fd0_hat_hz - best.mu_hat_hz_per_s * best_center,
                best.score,
                best.score_margin_rel,
                best.timing,
                best.num_pilots_used,
                True,
            )
        return best
