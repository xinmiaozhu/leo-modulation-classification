"""Blind Doppler-rate estimators for DRC-HOC.

Main estimator:
    PaperStrictDFRFTEstimator
which follows the white-paper pipeline:
    amplitude normalization -> normalized k-power branches -> alpha search
    -> S_peak winner-takes-all -> calibrated alpha-to-mu mapping.

Legacy DechirpFFTGridEstimator and FRFTGridEstimator are retained as ablation
baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

import numpy as np

from src.signal.modulation import get_constellation
from src.signal.pulse_shape import rrc_filter

from .compensation import conjugate_quadratic_compensation
from .frft import DirectFRFT, fast_dfrft_chirp_focus, make_alpha_grid, peak_sharpness, alpha_to_mu_whitepaper
from .nonlinear_transforms import generate_blind_normalized_candidates, generate_candidates, normalize_candidate_energy, parse_order_from_key
from .reliability_metrics import compute_V_hoc


@dataclass
class EstimationResult:
    mu_hat_hz_per_s: float
    S_peak: float
    selected_key: str
    selected_k: int
    score: float
    alpha_hat: float | None = None
    fd_peak_hz: float | None = None
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["extra"] is None:
            d["extra"] = {}
        return d


class PaperStrictDFRFTEstimator:
    """Paper-strict blind normalized-branch DFRFT estimator.

    It tests configured k branches; each branch y_k=(r/(|r|+delta))^k is scanned over
    alpha. The branch/angle with the largest S_peak wins.
    """

    def __init__(
        self,
        sample_rate_hz: float,
        candidate_orders: Iterable[int] = (2, 4, 8),
        alpha_steps: int = 2000,
        alpha_min: float = 1e-4,
        calibration_scale: float | None = None,
        calibration_bias: float = 0.0,
        delta: float = 1e-8,
        peak_exclude_bins: int = 5,
        peak_eta: float = 1e-5,
        centered_time: bool = False,
        mu_min_hz_per_s: float | None = None,
        mu_max_hz_per_s: float | None = None,
        dc_block: bool = False,
        mu_search_guard_fraction: float = 0.0,
        boundary_penalty_weight: float = 0.5,
        boundary_penalty_fraction: float = 0.03,
        peak_margin_rel_floor: float = 0.05,
        peak_margin_exclude_bins: int | None = None,
        boundary_reject_fraction: float = 0.03,
        boundary_reject_margin_rel: float = 0.10,
        global_reject_margin_rel: float = 0.05,
        enable_fallback: bool = False,
        fallback_mu_hz_per_s: float | None = None,
        hoc_rerank_top_n: int = 5,
        hoc_rerank_pool_size: int = 25,
        hoc_rerank_weight: float = 0.5,
        hoc_rerank_min_margin_rel: float = 0.05,
        hoc_rerank_min_score_ratio: float = 0.70,
        hoc_rerank_num_subwindows: int = 10,
        hoc_rerank_orders: tuple[tuple[int, int], ...] = ((2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (6, 0), (6, 3)),
        hoc_rerank_representation: str = "real_imag_abs",
        evm_rerank_enabled: bool = False,
        evm_rerank_top_n: int = 25,
        evm_rerank_pool_size: int = 50,
        evm_samples_per_symbol: int = 8,
        evm_rrc_beta: float = 0.35,
        evm_rrc_span: int = 8,
        evm_timing_phases: Iterable[int] | None = (0,),
        evm_phase_grid_size: int = 1,
        evm_max_symbols: int = 512,
        evm_symbol_trim: int = 8,
        evm_modulations: Iterable[str] = ("QPSK", "16QAM", "64QAM", "16APSK", "32APSK"),
        evm_refine_all_candidates: bool = True,
        evm_fine_search_radius_hz_per_s: float = 800.0,
        evm_fine_search_step_hz_per_s: float = 80.0,
    ) -> None:
        self.fs = float(sample_rate_hz)
        self.candidate_orders = tuple(int(k) for k in candidate_orders)
        self.alpha_steps = int(alpha_steps)
        self.alpha_min = float(alpha_min)
        self.alpha_grid = make_alpha_grid(alpha_steps=self.alpha_steps, alpha_min=self.alpha_min)
        self.calibration_scale = calibration_scale
        self.calibration_bias = float(calibration_bias)
        self.delta = float(delta)
        self.peak_exclude_bins = int(peak_exclude_bins)
        self.peak_eta = float(peak_eta)
        self.centered_time = bool(centered_time)
        self.mu_min = mu_min_hz_per_s
        self.mu_max = mu_max_hz_per_s
        self.dc_block = bool(dc_block)
        self.mu_search_guard_fraction = float(mu_search_guard_fraction)
        self.boundary_penalty_weight = float(boundary_penalty_weight)
        self.boundary_penalty_fraction = float(boundary_penalty_fraction)
        self.peak_margin_rel_floor = float(peak_margin_rel_floor)
        self.peak_margin_exclude_bins = peak_margin_exclude_bins
        self.boundary_reject_fraction = float(boundary_reject_fraction)
        self.boundary_reject_margin_rel = float(boundary_reject_margin_rel)
        self.global_reject_margin_rel = float(global_reject_margin_rel)
        self.enable_fallback = bool(enable_fallback)
        self.fallback_mu_hz_per_s = fallback_mu_hz_per_s
        self.hoc_rerank_top_n = int(hoc_rerank_top_n)
        self.hoc_rerank_pool_size = int(hoc_rerank_pool_size)
        self.hoc_rerank_weight = float(hoc_rerank_weight)
        self.hoc_rerank_min_margin_rel = float(hoc_rerank_min_margin_rel)
        self.hoc_rerank_min_score_ratio = float(hoc_rerank_min_score_ratio)
        self.hoc_rerank_num_subwindows = int(hoc_rerank_num_subwindows)
        self.hoc_rerank_orders = tuple((int(p), int(q)) for p, q in hoc_rerank_orders)
        self.hoc_rerank_representation = str(hoc_rerank_representation)
        self.evm_rerank_enabled = bool(evm_rerank_enabled)
        self.evm_rerank_top_n = int(evm_rerank_top_n)
        self.evm_rerank_pool_size = int(evm_rerank_pool_size)
        self.evm_samples_per_symbol = int(evm_samples_per_symbol)
        self.evm_rrc_beta = float(evm_rrc_beta)
        self.evm_rrc_span = int(evm_rrc_span)
        if evm_timing_phases is None:
            self.evm_timing_phases = tuple(range(max(self.evm_samples_per_symbol, 1)))
        else:
            self.evm_timing_phases = tuple(int(p) for p in evm_timing_phases)
        self.evm_phase_grid_size = int(evm_phase_grid_size)
        self.evm_max_symbols = int(evm_max_symbols)
        self.evm_symbol_trim = int(evm_symbol_trim)
        self.evm_modulations = tuple(str(m).upper() for m in evm_modulations)
        self.evm_refine_all_candidates = bool(evm_refine_all_candidates)
        self.evm_fine_search_radius = float(evm_fine_search_radius_hz_per_s)
        self.evm_fine_search_step = float(evm_fine_search_step_hz_per_s)
        self._evm_rrc_taps = rrc_filter(beta=self.evm_rrc_beta, span=self.evm_rrc_span, sps=max(self.evm_samples_per_symbol, 1))
        self._evm_constellations = tuple((m, get_constellation(m).astype(np.complex128)) for m in self.evm_modulations)
        self._evm_constellation_map = dict(self._evm_constellations)
        if self.evm_phase_grid_size <= 1:
            self._evm_phase_grid = np.asarray([0.0], dtype=np.float64)
        else:
            self._evm_phase_grid = np.linspace(0.0, 2.0 * np.pi, self.evm_phase_grid_size, endpoint=False, dtype=np.float64)

    def _resolve_evm_constellations(self, evm_modulations: Iterable[str] | None = None) -> tuple[tuple[str, np.ndarray], ...]:
        if evm_modulations is None:
            return self._evm_constellations

        resolved: list[tuple[str, np.ndarray]] = []
        for modulation in evm_modulations:
            key = str(modulation).upper()
            if not key:
                continue
            constellation = self._evm_constellation_map.get(key)
            if constellation is None:
                constellation = get_constellation(key).astype(np.complex128)
            resolved.append((key, constellation))
        return tuple(resolved) if resolved else self._evm_constellations

    def _effective_mu_bounds(self) -> tuple[float | None, float | None]:
        if self.mu_min is None or self.mu_max is None:
            return self.mu_min, self.mu_max

        mu_min = float(self.mu_min)
        mu_max = float(self.mu_max)
        span = mu_max - mu_min
        if span <= 0.0:
            return mu_min, mu_max

        guard = max(0.0, self.mu_search_guard_fraction) * span
        if guard <= 0.0 or 2.0 * guard >= span:
            return mu_min, mu_max
        return mu_min + guard, mu_max - guard

    def _alpha_grid_for_branch(self, k: int, num_samples: int) -> np.ndarray:
        mu_min, mu_max = self._effective_mu_bounds()
        if mu_min is None and mu_max is None:
            return self.alpha_grid

        mu_candidates = alpha_to_mu_whitepaper(
            self.alpha_grid,
            k,
            self.fs,
            num_samples,
            self.calibration_scale,
            self.calibration_bias,
        )
        mask = np.ones_like(mu_candidates, dtype=bool)
        if mu_min is not None:
            mask &= mu_candidates >= mu_min
        if mu_max is not None:
            mask &= mu_candidates <= mu_max

        alpha_grid = self.alpha_grid[mask]
        if alpha_grid.size > 0:
            return alpha_grid

        # Fall back to the configured grid instead of producing an invalid
        # estimate if a very tight physical guard removes every candidate.
        return self.alpha_grid

    def _boundary_distance_fraction(self, mu_hat: float, mu_min: float | None, mu_max: float | None) -> float | None:
        if mu_min is None or mu_max is None:
            return None

        span = float(mu_max) - float(mu_min)
        if span <= 0.0:
            return None

        distance = min(abs(float(mu_hat) - float(mu_min)), abs(float(mu_max) - float(mu_hat)))
        return float(max(0.0, distance / span))

    def _boundary_factor(self, mu_hat: float, mu_min: float | None, mu_max: float | None) -> float:
        distance_fraction = self._boundary_distance_fraction(mu_hat, mu_min, mu_max)
        if distance_fraction is None:
            return 1.0

        scale = max(self.boundary_penalty_fraction, 1e-12)
        weight = min(max(self.boundary_penalty_weight, 0.0), 1.0)
        return float(1.0 - weight * np.exp(-distance_fraction / scale))

    def _is_boundary_low_margin(self, mu_hat: float, margin_rel: float, mu_min: float | None, mu_max: float | None) -> bool:
        distance_fraction = self._boundary_distance_fraction(mu_hat, mu_min, mu_max)
        if distance_fraction is None:
            return False

        reject_fraction = max(0.0, self.boundary_reject_fraction)
        if reject_fraction <= 0.0:
            return False

        return bool(distance_fraction <= reject_fraction and float(margin_rel) < self.boundary_reject_margin_rel)

    def _fallback_mu(self, mu_min: float | None, mu_max: float | None) -> float:
        if self.fallback_mu_hz_per_s is not None:
            return float(self.fallback_mu_hz_per_s)
        if mu_min is not None and mu_max is not None:
            return float(0.5 * (float(mu_min) + float(mu_max)))
        return 0.0

    def _margin_factor(self, best_score: float, second_score: float | None) -> tuple[float, float]:
        if second_score is None or not np.isfinite(second_score):
            return 1.0, np.inf

        margin_rel = max(0.0, (float(best_score) - float(second_score)) / (abs(float(second_score)) + 1e-12))
        floor = max(self.peak_margin_rel_floor, 1e-12)
        return float(margin_rel / (margin_rel + floor)), float(margin_rel)

    def _second_score(self, branch_scores: list[dict[str, Any]], best_idx: int) -> float | None:
        if len(branch_scores) <= 1:
            return None

        exclude_bins = self.peak_margin_exclude_bins
        if exclude_bins is None:
            exclude_bins = self.peak_exclude_bins
        exclude_bins = max(0, int(exclude_bins))

        for candidate in sorted(branch_scores, key=lambda item: item["raw_score"], reverse=True):
            if abs(int(candidate["alpha_index"]) - int(best_idx)) > exclude_bins:
                return float(candidate["raw_score"])
        return None

    @staticmethod
    def _minmax_normalize(values: np.ndarray, invert: bool = False) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        out = np.zeros_like(values, dtype=np.float64)
        if not np.any(finite):
            return out

        finite_values = values[finite]
        lo = float(np.min(finite_values))
        hi = float(np.max(finite_values))
        if hi - lo <= 1e-12:
            out[finite] = 1.0 if not invert else 0.0
            return out

        norm = (finite_values - lo) / (hi - lo)
        out[finite] = 1.0 - norm if invert else norm
        return out

    def _hoc_rerank_candidates(self, x: np.ndarray, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        if not candidates:
            return None, []
        candidates = [c for c in candidates if np.isfinite(float(c["effective_score"]))]
        if not candidates:
            return None, []
        best = max(candidates, key=lambda item: item["effective_score"])
        if self.hoc_rerank_top_n <= 0 or self.hoc_rerank_weight <= 0.0:
            best["hoc_rerank_used"] = False
            best["hoc_rerank_reason"] = "disabled"
            best["hoc_rerank_eligible_count"] = 0
            return best, []

        best_score = float(best["effective_score"])
        if not np.isfinite(best_score) or best_score <= 0.0:
            best["hoc_rerank_used"] = False
            best["hoc_rerank_reason"] = "nonpositive_best_score"
            best["hoc_rerank_eligible_count"] = 0
            return best, []

        min_margin = max(0.0, self.hoc_rerank_min_margin_rel)
        min_score_ratio = min(max(self.hoc_rerank_min_score_ratio, 0.0), 1.0)
        score_floor = best_score * min_score_ratio
        eligible = [
            c
            for c in candidates
            if float(c["effective_score"]) >= score_floor and float(c["margin_rel"]) >= min_margin
        ]
        eligible_count = len(eligible)

        # HOC stability is a tie-breaker only. If the primary winner itself is
        # not eligible, do not allow weaker candidates to overrule it.
        if not any(c is best for c in eligible):
            best["hoc_rerank_used"] = False
            best["hoc_rerank_reason"] = "best_not_eligible"
            best["hoc_rerank_eligible_count"] = int(eligible_count)
            return best, []

        top_n = max(1, min(self.hoc_rerank_top_n, len(eligible)))
        rerank_candidates = sorted(eligible, key=lambda item: item["effective_score"], reverse=True)[:top_n]
        if len(rerank_candidates) <= 1:
            best["hoc_rerank_used"] = False
            best["hoc_rerank_reason"] = "single_eligible"
            best["hoc_rerank_eligible_count"] = int(eligible_count)
            return best, []

        for candidate in rerank_candidates:
            try:
                r_comp = conjugate_quadratic_compensation(
                    x,
                    sample_rate_hz=self.fs,
                    mu_hat_hz_per_s=float(candidate["mu"]),
                    centered=self.centered_time,
                )
                candidate["hoc_v"] = float(
                    compute_V_hoc(
                        r_comp,
                        num_subwindows=self.hoc_rerank_num_subwindows,
                        orders=self.hoc_rerank_orders,
                        representation=self.hoc_rerank_representation,
                    )
                )
            except Exception:
                candidate["hoc_v"] = np.inf

        effective_scores = np.asarray([c["effective_score"] for c in rerank_candidates], dtype=np.float64)
        hoc_values = np.asarray([c["hoc_v"] for c in rerank_candidates], dtype=np.float64)
        effective_norm = self._minmax_normalize(effective_scores, invert=False)
        hoc_stability_norm = self._minmax_normalize(hoc_values, invert=True)
        hoc_penalty_norm = 1.0 - hoc_stability_norm
        weight = max(0.0, self.hoc_rerank_weight)

        for idx, candidate in enumerate(rerank_candidates):
            candidate["hoc_score_norm"] = float(effective_norm[idx])
            candidate["hoc_stability_norm"] = float(hoc_stability_norm[idx])
            candidate["hoc_penalty_norm"] = float(hoc_penalty_norm[idx])
            candidate["hoc_rerank_score"] = float(float(candidate["effective_score"]) * (1.0 + weight * hoc_stability_norm[idx]))
            candidate["hoc_rerank_used"] = True
            candidate["hoc_rerank_reason"] = "eligible"
            candidate["hoc_rerank_eligible_count"] = int(eligible_count)

        return max(rerank_candidates, key=lambda item: item["hoc_rerank_score"]), rerank_candidates

    def _evm_score_symbols(self, symbols: np.ndarray, evm_constellations: tuple[tuple[str, np.ndarray], ...]) -> tuple[float, str, float]:
        symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
        trim = max(0, self.evm_symbol_trim)
        if trim > 0 and symbols.size > 2 * trim:
            symbols = symbols[trim:-trim]
        if self.evm_max_symbols > 0 and symbols.size > self.evm_max_symbols:
            idx = np.linspace(0, symbols.size - 1, self.evm_max_symbols, dtype=np.int64)
            symbols = symbols[idx]
        if symbols.size < 16:
            return np.inf, "", 0.0

        symbols = symbols - np.mean(symbols)
        symbols = symbols / (np.sqrt(np.mean(np.abs(symbols) ** 2)) + 1e-12)

        best_score = np.inf
        best_modulation = ""
        best_phase = 0.0
        for phase in self._evm_phase_grid:
            rotated = symbols * np.exp(-1j * phase)
            for modulation, constellation in evm_constellations:
                distances = np.abs(rotated[:, None] - constellation[None, :]) ** 2
                score = float(np.mean(np.min(distances, axis=1)))
                if score < best_score:
                    best_score = score
                    best_modulation = modulation
                    best_phase = float(phase)
        return best_score, best_modulation, best_phase

    def _evm_score_mu(self, x: np.ndarray, mu_hat: float, evm_constellations: tuple[tuple[str, np.ndarray], ...]) -> tuple[float, str, int, float]:
        r_comp = conjugate_quadratic_compensation(
            x,
            sample_rate_hz=self.fs,
            mu_hat_hz_per_s=float(mu_hat),
            centered=self.centered_time,
        )
        matched = np.convolve(r_comp, self._evm_rrc_taps, mode="same")

        best_score = np.inf
        best_modulation = ""
        best_timing = -1
        best_phase = 0.0
        sps = max(self.evm_samples_per_symbol, 1)
        timing_phases = self.evm_timing_phases or tuple(range(sps))
        for timing in timing_phases:
            timing = int(timing) % sps
            symbols = matched[timing::sps]
            score, modulation, phase = self._evm_score_symbols(symbols, evm_constellations)
            if score < best_score:
                best_score = float(score)
                best_modulation = modulation
                best_timing = int(timing)
                best_phase = float(phase)
        return best_score, best_modulation, best_timing, best_phase

    def _evm_refine_candidate(
        self,
        x: np.ndarray,
        candidate: dict[str, Any],
        mu_min: float | None,
        mu_max: float | None,
        evm_constellations: tuple[tuple[str, np.ndarray], ...],
    ) -> dict[str, Any]:
        radius = max(0.0, self.evm_fine_search_radius)
        step = max(0.0, self.evm_fine_search_step)
        if radius <= 0.0 or step <= 0.0:
            return candidate

        center_mu = float(candidate["mu"])
        offsets = np.arange(-radius, radius + 0.5 * step, step, dtype=np.float64)
        best = candidate
        for offset in offsets:
            mu = center_mu + float(offset)
            if mu_min is not None and mu < mu_min:
                continue
            if mu_max is not None and mu > mu_max:
                continue
            score, modulation, timing, phase = self._evm_score_mu(x, mu, evm_constellations)
            if score < float(best.get("evm_score", np.inf)):
                best = {
                    **candidate,
                    "mu": float(mu),
                    "evm_score": float(score),
                    "evm_best_modulation": modulation,
                    "evm_best_timing": int(timing),
                    "evm_best_phase": float(phase),
                    "evm_fine_used": True,
                }
        return best

    def _evm_rerank_candidates(
        self,
        x: np.ndarray,
        candidates: list[dict[str, Any]],
        mu_min: float | None,
        mu_max: float | None,
        evm_constellations: tuple[tuple[str, np.ndarray], ...],
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        if not self.evm_rerank_enabled or self.evm_rerank_top_n <= 0:
            return None, []

        candidates = [c for c in candidates if np.isfinite(float(c["raw_score"]))]
        if not candidates:
            return None, []

        pool_size = max(self.evm_rerank_pool_size, self.evm_rerank_top_n, 1)
        ranked = sorted(candidates, key=lambda item: item["raw_score"], reverse=True)[:pool_size]
        best_effective = max(candidates, key=lambda item: item["effective_score"])
        if not any(c is best_effective for c in ranked):
            ranked.append(best_effective)

        top_n = max(1, min(self.evm_rerank_top_n, len(ranked)))
        rerank_candidates = sorted(ranked, key=lambda item: item["raw_score"], reverse=True)[:top_n]
        if not any(c is best_effective for c in rerank_candidates):
            rerank_candidates.append(best_effective)

        scored: list[dict[str, Any]] = []
        for candidate in rerank_candidates:
            score, modulation, timing, phase = self._evm_score_mu(x, float(candidate["mu"]), evm_constellations)
            scored_candidate = {
                **candidate,
                "pre_evm_mu": float(candidate["mu"]),
                "evm_score": float(score),
                "evm_best_modulation": modulation,
                "evm_best_timing": int(timing),
                "evm_best_phase": float(phase),
                "evm_rerank_used": True,
                "evm_fine_used": False,
            }
            if self.evm_refine_all_candidates:
                scored_candidate = self._evm_refine_candidate(x, scored_candidate, mu_min, mu_max, evm_constellations)
            scored.append(scored_candidate)

        if not scored:
            return None, []

        best = min(scored, key=lambda item: item["evm_score"])
        if not self.evm_refine_all_candidates:
            best = self._evm_refine_candidate(x, best, mu_min, mu_max, evm_constellations)
        best["evm_rerank_used"] = True
        best["evm_candidate_count"] = int(len(scored))
        return best, scored

    def estimate(self, x: np.ndarray, evm_modulations: Iterable[str] | None = None) -> EstimationResult:
        candidates = generate_blind_normalized_candidates(x, self.candidate_orders, delta=self.delta)
        evm_constellations = self._resolve_evm_constellations(evm_modulations)
        evm_allowed_modulations = tuple(modulation for modulation, _ in evm_constellations)
        best: dict[str, Any] = {
            "effective_score": -np.inf,
            "raw_score": 0.0,
            "mu": 0.0,
            "alpha": None,
            "key": "none",
            "k": 1,
            "peak": 0.0,
            "second_score": None,
            "margin_rel": 0.0,
            "margin_factor": 0.0,
            "boundary_factor": 1.0,
            "fallback_used": False,
            "fallback_reason": "",
            "pre_fallback_mu": np.nan,
            "pre_fallback_selected_k": -1,
            "pre_fallback_margin_rel": np.nan,
            "rejected_boundary_low_margin": False,
        }
        mu_min_eff, mu_max_eff = self._effective_mu_bounds()
        branch_summaries: dict[str, dict[str, Any]] = {}
        rejected_boundary_low_margin_count = 0
        candidate_pool: list[dict[str, Any]] = []

        for key, y in candidates.items():
            if self.dc_block:
                y = y - np.mean(y)
            y = normalize_candidate_energy(y)
            k = parse_order_from_key(key)
            alpha_grid = self._alpha_grid_for_branch(k, len(y))
            branch_scores: list[dict[str, Any]] = []
            for alpha_index, alpha in enumerate(alpha_grid):
                spectrum, mu_hat = fast_dfrft_chirp_focus(
                    y,
                    alpha=float(alpha),
                    k=k,
                    sample_rate_hz=self.fs,
                    calibration_scale=self.calibration_scale,
                    calibration_bias=self.calibration_bias,
                    centered_time=self.centered_time,
                )
                if mu_min_eff is not None and mu_hat < mu_min_eff:
                    continue
                if mu_max_eff is not None and mu_hat > mu_max_eff:
                    continue
                score = peak_sharpness(spectrum, mode="peak_minus_local", exclude_bins=self.peak_exclude_bins, eta=self.peak_eta)
                branch_scores.append(
                    {
                        "raw_score": float(score),
                        "mu": float(mu_hat),
                        "alpha": float(alpha),
                        "alpha_index": int(alpha_index),
                        "peak": float(np.max(np.abs(spectrum) ** 2)),
                    }
                )

            if not branch_scores:
                continue

            branch_accepted: list[dict[str, Any]] = []
            rejected_in_branch = 0
            pool_size = max(self.hoc_rerank_pool_size, self.hoc_rerank_top_n, self.evm_rerank_pool_size, self.evm_rerank_top_n, 1)

            for candidate in sorted(branch_scores, key=lambda item: item["raw_score"], reverse=True):
                candidate_second_score = self._second_score(branch_scores, int(candidate["alpha_index"]))
                candidate_margin_factor, candidate_margin_rel = self._margin_factor(float(candidate["raw_score"]), candidate_second_score)
                if self._is_boundary_low_margin(float(candidate["mu"]), candidate_margin_rel, mu_min_eff, mu_max_eff):
                    rejected_in_branch += 1
                    rejected_boundary_low_margin_count += 1
                    continue
                boundary_factor = self._boundary_factor(float(candidate["mu"]), mu_min_eff, mu_max_eff)
                candidate_eff = float(candidate["raw_score"]) * candidate_margin_factor * boundary_factor
                branch_accepted.append(
                    {
                        **candidate,
                        "key": key,
                        "k": int(k),
                        "second_score": None if candidate_second_score is None else float(candidate_second_score),
                        "margin_rel": float(candidate_margin_rel),
                        "margin_factor": float(candidate_margin_factor),
                        "boundary_factor": float(boundary_factor),
                        "effective_score": float(candidate_eff),
                        "hoc_v": np.nan,
                        "hoc_score_norm": np.nan,
                        "hoc_stability_norm": np.nan,
                        "hoc_penalty_norm": np.nan,
                        "hoc_rerank_score": np.nan,
                        "hoc_rerank_used": False,
                    }
                )
                if len(branch_accepted) >= pool_size:
                    break

            if not branch_accepted:
                top_raw = max(branch_scores, key=lambda item: item["raw_score"])
                branch_summaries[key] = {
                    "k": int(k),
                    "mu": float(top_raw["mu"]),
                    "alpha": float(top_raw["alpha"]),
                    "raw_score": float(top_raw["raw_score"]),
                    "second_score": None,
                    "margin_rel": 0.0,
                    "margin_factor": 0.0,
                    "boundary_factor": self._boundary_factor(float(top_raw["mu"]), mu_min_eff, mu_max_eff),
                    "effective_score": -np.inf,
                    "num_alpha_candidates": int(len(alpha_grid)),
                    "rejected_boundary_low_margin": True,
                    "rejected_boundary_low_margin_count": int(rejected_in_branch),
                }
                continue

            branch_best = max(branch_accepted, key=lambda item: item["effective_score"])
            candidate_pool.extend(branch_accepted)
            branch_summaries[key] = {
                "k": int(k),
                "mu": float(branch_best["mu"]),
                "alpha": float(branch_best["alpha"]),
                "raw_score": float(branch_best["raw_score"]),
                "second_score": None if branch_best["second_score"] is None else float(branch_best["second_score"]),
                "margin_rel": float(branch_best["margin_rel"]),
                "margin_factor": float(branch_best["margin_factor"]),
                "boundary_factor": float(branch_best["boundary_factor"]),
                "effective_score": float(branch_best["effective_score"]),
                "num_alpha_candidates": int(len(alpha_grid)),
                "num_accepted_candidates": int(len(branch_accepted)),
                "rejected_boundary_low_margin": bool(rejected_in_branch > 0),
                "rejected_boundary_low_margin_count": int(rejected_in_branch),
            }

        evm_rerank_candidates: list[dict[str, Any]] = []
        rerank_best, evm_rerank_candidates = self._evm_rerank_candidates(x, candidate_pool, mu_min_eff, mu_max_eff, evm_constellations)
        rerank_candidates: list[dict[str, Any]] = []
        if rerank_best is None:
            rerank_best, rerank_candidates = self._hoc_rerank_candidates(x, candidate_pool)
        if rerank_best is not None:
            best.update(
                {
                    "effective_score": float(rerank_best["effective_score"]),
                    "raw_score": float(rerank_best["raw_score"]),
                    "mu": float(rerank_best["mu"]),
                    "alpha": float(rerank_best["alpha"]),
                    "key": str(rerank_best["key"]),
                    "k": int(rerank_best["k"]),
                    "peak": float(rerank_best["peak"]),
                    "second_score": None if rerank_best["second_score"] is None else float(rerank_best["second_score"]),
                    "margin_rel": float(rerank_best["margin_rel"]),
                    "margin_factor": float(rerank_best["margin_factor"]),
                    "boundary_factor": float(rerank_best["boundary_factor"]),
                    "fallback_used": False,
                    "rejected_boundary_low_margin": False,
                    "hoc_v": float(rerank_best.get("hoc_v", np.nan)),
                    "hoc_score_norm": float(rerank_best.get("hoc_score_norm", np.nan)),
                    "hoc_stability_norm": float(rerank_best.get("hoc_stability_norm", np.nan)),
                    "hoc_penalty_norm": float(rerank_best.get("hoc_penalty_norm", np.nan)),
                    "hoc_rerank_score": float(rerank_best.get("hoc_rerank_score", np.nan)),
                    "hoc_rerank_used": bool(rerank_best.get("hoc_rerank_used", False)),
                    "hoc_rerank_reason": str(rerank_best.get("hoc_rerank_reason", "")),
                    "hoc_rerank_eligible_count": int(rerank_best.get("hoc_rerank_eligible_count", 0)),
                    "pre_evm_mu": float(rerank_best.get("pre_evm_mu", np.nan)),
                    "evm_score": float(rerank_best.get("evm_score", np.nan)),
                    "evm_best_modulation": str(rerank_best.get("evm_best_modulation", "")),
                    "evm_best_timing": int(rerank_best.get("evm_best_timing", -1)),
                    "evm_best_phase": float(rerank_best.get("evm_best_phase", np.nan)),
                    "evm_rerank_used": bool(rerank_best.get("evm_rerank_used", False)),
                    "evm_fine_used": bool(rerank_best.get("evm_fine_used", False)),
                    "evm_candidate_count": int(rerank_best.get("evm_candidate_count", 0)),
                }
            )

        if best["alpha"] is None and self.enable_fallback:
            fallback_mu = self._fallback_mu(mu_min_eff, mu_max_eff)
            best.update(
                {
                    "effective_score": 0.0,
                    "raw_score": 0.0,
                    "mu": float(fallback_mu),
                    "alpha": np.nan,
                    "key": "fallback_midpoint",
                    "k": 0,
                    "fallback_used": True,
                    "fallback_reason": "no_valid_candidate",
                    "pre_fallback_mu": np.nan,
                    "pre_fallback_selected_k": -1,
                    "pre_fallback_margin_rel": np.nan,
                    "rejected_boundary_low_margin": bool(rejected_boundary_low_margin_count > 0),
                }
            )
        elif best["alpha"] is None:
            best.update({"effective_score": 0.0, "raw_score": 0.0, "mu": 0.0, "alpha": np.pi / 2, "key": "none", "k": 1})
        elif self.enable_fallback and float(best["margin_rel"]) < self.global_reject_margin_rel:
            fallback_mu = self._fallback_mu(mu_min_eff, mu_max_eff)
            best.update(
                {
                    "mu": float(fallback_mu),
                    "key": "fallback_low_margin",
                    "k": 0,
                    "fallback_used": True,
                    "fallback_reason": "global_low_margin",
                    "pre_fallback_mu": float(best["mu"]),
                    "pre_fallback_selected_k": int(best["k"]),
                    "pre_fallback_margin_rel": float(best["margin_rel"]),
                }
            )

        return EstimationResult(
            mu_hat_hz_per_s=float(best["mu"]),
            S_peak=float(best["raw_score"]),
            selected_key=str(best["key"]),
            selected_k=int(best["k"]),
            score=float(best["effective_score"]),
            alpha_hat=float(best["alpha"]),
            fd_peak_hz=None,
            extra={
                "estimator": "paper_strict_dfrft",
                "alpha_steps": int(self.alpha_steps),
                "num_alpha_grid": int(len(self.alpha_grid)),
                "alpha_min": float(self.alpha_min),
                "calibration_scale": None if self.calibration_scale is None else float(self.calibration_scale),
                "calibration_bias": float(self.calibration_bias),
                "normalized_branches_only": True,
                "peak_exclude_bins": int(self.peak_exclude_bins),
                "peak_eta": float(self.peak_eta),
                "dc_block": bool(self.dc_block),
                "mu_search_guard_fraction": float(self.mu_search_guard_fraction),
                "boundary_penalty_weight": float(self.boundary_penalty_weight),
                "boundary_penalty_fraction": float(self.boundary_penalty_fraction),
                "peak_margin_rel_floor": float(self.peak_margin_rel_floor),
                "peak_margin_exclude_bins": int(self.peak_margin_exclude_bins if self.peak_margin_exclude_bins is not None else self.peak_exclude_bins),
                "boundary_reject_fraction": float(self.boundary_reject_fraction),
                "boundary_reject_margin_rel": float(self.boundary_reject_margin_rel),
                "global_reject_margin_rel": float(self.global_reject_margin_rel),
                "enable_fallback": bool(self.enable_fallback),
                "fallback_mu_hz_per_s": None if self.fallback_mu_hz_per_s is None else float(self.fallback_mu_hz_per_s),
                "hoc_rerank_top_n": int(self.hoc_rerank_top_n),
                "hoc_rerank_pool_size": int(self.hoc_rerank_pool_size),
                "hoc_rerank_weight": float(self.hoc_rerank_weight),
                "hoc_rerank_min_margin_rel": float(self.hoc_rerank_min_margin_rel),
                "hoc_rerank_min_score_ratio": float(self.hoc_rerank_min_score_ratio),
                "hoc_rerank_num_subwindows": int(self.hoc_rerank_num_subwindows),
                "effective_mu_min_hz_per_s": None if mu_min_eff is None else float(mu_min_eff),
                "effective_mu_max_hz_per_s": None if mu_max_eff is None else float(mu_max_eff),
                "selected_second_score": None if best["second_score"] is None else float(best["second_score"]),
                "selected_margin_rel": float(best["margin_rel"]),
                "selected_margin_factor": float(best["margin_factor"]),
                "selected_boundary_factor": float(best["boundary_factor"]),
                "fallback_used": bool(best["fallback_used"]),
                "fallback_reason": str(best["fallback_reason"]),
                "pre_fallback_mu": float(best["pre_fallback_mu"]),
                "pre_fallback_selected_k": int(best["pre_fallback_selected_k"]),
                "pre_fallback_margin_rel": float(best["pre_fallback_margin_rel"]),
                "selected_rejected_boundary_low_margin": bool(best["rejected_boundary_low_margin"]),
                "rejected_boundary_low_margin_count": int(rejected_boundary_low_margin_count),
                "selected_hoc_v": float(best.get("hoc_v", np.nan)),
                "selected_hoc_score_norm": float(best.get("hoc_score_norm", np.nan)),
                "selected_hoc_stability_norm": float(best.get("hoc_stability_norm", np.nan)),
                "selected_hoc_penalty_norm": float(best.get("hoc_penalty_norm", np.nan)),
                "selected_hoc_rerank_score": float(best.get("hoc_rerank_score", np.nan)),
                "hoc_rerank_used": bool(best.get("hoc_rerank_used", False)),
                "hoc_rerank_reason": str(best.get("hoc_rerank_reason", "")),
                "hoc_rerank_eligible_count": int(best.get("hoc_rerank_eligible_count", 0)),
                "evm_rerank_enabled": bool(self.evm_rerank_enabled),
                "evm_rerank_top_n": int(self.evm_rerank_top_n),
                "evm_rerank_pool_size": int(self.evm_rerank_pool_size),
                "evm_samples_per_symbol": int(self.evm_samples_per_symbol),
                "evm_rrc_beta": float(self.evm_rrc_beta),
                "evm_rrc_span": int(self.evm_rrc_span),
                "evm_timing_phases": tuple(int(p) for p in self.evm_timing_phases),
                "evm_phase_grid_size": int(self.evm_phase_grid_size),
                "evm_max_symbols": int(self.evm_max_symbols),
                "evm_symbol_trim": int(self.evm_symbol_trim),
                "evm_modulations": tuple(self.evm_modulations),
                "evm_allowed_modulations": evm_allowed_modulations,
                "evm_label_aware_used": bool(evm_modulations is not None),
                "evm_refine_all_candidates": bool(self.evm_refine_all_candidates),
                "evm_fine_search_radius_hz_per_s": float(self.evm_fine_search_radius),
                "evm_fine_search_step_hz_per_s": float(self.evm_fine_search_step),
                "pre_evm_mu": float(best.get("pre_evm_mu", np.nan)),
                "selected_evm_score": float(best.get("evm_score", np.nan)),
                "selected_evm_modulation": str(best.get("evm_best_modulation", "")),
                "selected_evm_timing": int(best.get("evm_best_timing", -1)),
                "selected_evm_phase": float(best.get("evm_best_phase", np.nan)),
                "evm_rerank_used": bool(best.get("evm_rerank_used", False)),
                "evm_fine_used": bool(best.get("evm_fine_used", False)),
                "evm_candidate_count": int(best.get("evm_candidate_count", 0)),
                "hoc_rerank_candidates": [
                    {
                        "k": int(c["k"]),
                        "mu": float(c["mu"]),
                        "raw_score": float(c["raw_score"]),
                        "effective_score": float(c["effective_score"]),
                        "hoc_v": float(c.get("hoc_v", np.nan)),
                        "hoc_rerank_score": float(c.get("hoc_rerank_score", np.nan)),
                    }
                    for c in rerank_candidates
                ],
                "evm_rerank_candidates": [
                    {
                        "k": int(c["k"]),
                        "mu": float(c["mu"]),
                        "raw_score": float(c["raw_score"]),
                        "effective_score": float(c["effective_score"]),
                        "evm_score": float(c.get("evm_score", np.nan)),
                        "evm_best_modulation": str(c.get("evm_best_modulation", "")),
                        "evm_best_timing": int(c.get("evm_best_timing", -1)),
                    }
                    for c in evm_rerank_candidates
                ],
                "branch_summaries": branch_summaries,
            },
        )


class DechirpFFTGridEstimator:
    """Legacy dechirp-FFT mu-grid estimator for ablation/debugging."""

    def __init__(
        self,
        sample_rate_hz: float,
        mu_grid_hz_per_s: np.ndarray | None = None,
        candidate_orders: Iterable[int] = (2, 4, 8),
        include_direct_power: bool = False,
        include_phase_normalized: bool = True,
        sharpness_mode: str = "peak_minus_local",
        centered_time: bool = False,
    ) -> None:
        self.fs = float(sample_rate_hz)
        self.mu_grid = np.asarray(mu_grid_hz_per_s if mu_grid_hz_per_s is not None else np.linspace(-5.0e3, 5.0e3, 201), dtype=np.float64)
        self.candidate_orders = tuple(int(k) for k in candidate_orders)
        self.include_direct_power = include_direct_power
        self.include_phase_normalized = include_phase_normalized
        self.sharpness_mode = sharpness_mode
        self.centered_time = centered_time

    def _score_for_mu(self, y: np.ndarray, k: int, mu: float) -> tuple[float, float, np.ndarray]:
        y_comp = conjugate_quadratic_compensation(y, sample_rate_hz=self.fs, mu_hat_hz_per_s=k * mu, centered=self.centered_time)
        Y = np.fft.fftshift(np.fft.fft(y_comp))
        score = peak_sharpness(Y, mode=self.sharpness_mode)
        freqs = np.fft.fftshift(np.fft.fftfreq(len(y_comp), d=1.0 / self.fs))
        fd_peak = float(freqs[int(np.argmax(np.abs(Y) ** 2))])
        return float(score), fd_peak, Y

    def estimate(self, x: np.ndarray) -> EstimationResult:
        candidates = generate_candidates(x, self.candidate_orders, self.include_direct_power, self.include_phase_normalized)
        best: dict[str, Any] = {"score": -np.inf, "mu": 0.0, "key": "", "k": 1, "fd_peak": None}
        for key, y in candidates.items():
            y = normalize_candidate_energy(y)
            k = parse_order_from_key(key)
            for mu in self.mu_grid:
                score, fd_peak, _ = self._score_for_mu(y, k=k, mu=float(mu))
                if score > best["score"]:
                    best.update({"score": score, "mu": float(mu), "key": key, "k": k, "fd_peak": fd_peak})
        return EstimationResult(float(best["mu"]), float(best["score"]), str(best["key"]), int(best["k"]), float(best["score"]), None, best["fd_peak"], {"estimator": "dechirp_fft_grid", "num_mu_grid": int(len(self.mu_grid))})


class FRFTGridEstimator:
    """Legacy direct-FRFT estimator for small validation examples."""

    def __init__(
        self,
        sample_rate_hz: float,
        alpha_grid: np.ndarray | None = None,
        candidate_orders: Iterable[int] = (2, 4, 8),
        alpha_to_mu_scale: float | None = None,
        include_direct_power: bool = False,
        include_phase_normalized: bool = True,
        sharpness_mode: str = "peak_minus_local",
    ) -> None:
        self.fs = float(sample_rate_hz)
        self.alpha_grid = np.asarray(alpha_grid if alpha_grid is not None else make_alpha_grid(101, 0.05), dtype=np.float64)
        self.candidate_orders = tuple(int(k) for k in candidate_orders)
        self.alpha_to_mu_scale = alpha_to_mu_scale
        self.include_direct_power = include_direct_power
        self.include_phase_normalized = include_phase_normalized
        self.frft = DirectFRFT(sharpness_mode=sharpness_mode)

    def alpha_to_mu(self, alpha: float, selected_k: int, num_samples: int) -> float:
        return float(alpha_to_mu_whitepaper(alpha, selected_k, self.fs, num_samples, self.alpha_to_mu_scale))

    def estimate(self, x: np.ndarray) -> EstimationResult:
        candidates = generate_candidates(x, self.candidate_orders, self.include_direct_power, self.include_phase_normalized)
        best: dict[str, Any] = {"score": -np.inf, "alpha": None, "mu": 0.0, "key": "", "k": 1}
        for key, y in candidates.items():
            y = normalize_candidate_energy(y)
            k = parse_order_from_key(key)
            res = self.frft.search_peak(y, self.alpha_grid)
            score = float(res["sharpness"])
            if score > best["score"]:
                alpha = float(res["alpha"])
                best.update({"score": score, "alpha": alpha, "mu": self.alpha_to_mu(alpha, k, len(y)), "key": key, "k": k})
        return EstimationResult(float(best["mu"]), float(best["score"]), str(best["key"]), int(best["k"]), float(best["score"]), float(best["alpha"]) if best["alpha"] is not None else None, None, {"estimator": "direct_frft_grid", "num_alpha_grid": int(len(self.alpha_grid))})
