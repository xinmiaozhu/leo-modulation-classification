"""High-level DRC-HOC feature extractor.

Default pipeline exactly follows the engineering white paper:
    r_in -> blind amplitude-normalized branches -> DFRFT alpha search
    -> S_peak winner-takes-all -> alpha-to-mu mapping
    -> compensate original non-normalized r_in -> HOC + V_hoc
    -> meta = [gamma_hat, S_peak, V_hoc]
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from src.physics.gamma_metric import compute_gamma
from src.utils.math_utils import to_complex

from .compensation import compensate_constant_doppler, conjugate_quadratic_compensation, residual_gamma
from .cumulants import extract_hoc_features, hoc_feature_names
from .doppler_rate_estimator import DechirpFFTGridEstimator, EstimationResult, FRFTGridEstimator, PaperStrictDFRFTEstimator
from .reliability_metrics import compute_V_hoc


@dataclass
class DRCHOCConfig:
    sample_rate_hz: float = 1.0e6
    observation_time_s: float | None = None
    candidate_orders: tuple[int, ...] = (2, 4, 8)

    # Main paper-strict estimator.
    estimator_type: str = "paper_strict_dfrft"  # paper_strict_dfrft, dechirp_fft, frft

    # Physical range also used as clipping range for alpha-to-mu candidates.
    mu_grid_min: float = -5.0e3
    mu_grid_max: float = 5.0e3
    mu_grid_size: int = 201

    # DFRFT alpha-search parameters.
    alpha_steps: int = 2000
    alpha_min: float = 1.0e-4
    calibration_scale: float | None = None
    calibration_bias: float = 0.0
    peak_exclude_bins: int = 5
    peak_eta: float = 1.0e-5
    delta: float = 1.0e-8
    dc_block: bool = False
    mu_search_guard_fraction: float = 0.0
    boundary_penalty_weight: float = 0.5
    boundary_penalty_fraction: float = 0.03
    peak_margin_rel_floor: float = 0.05
    peak_margin_exclude_bins: int | None = None
    boundary_reject_fraction: float = 0.03
    boundary_reject_margin_rel: float = 0.10
    global_reject_margin_rel: float = 0.05
    enable_fallback: bool = False
    fallback_mu_hz_per_s: float | None = None
    hoc_rerank_top_n: int = 5
    hoc_rerank_pool_size: int = 25
    hoc_rerank_weight: float = 0.5
    hoc_rerank_min_margin_rel: float = 0.05
    hoc_rerank_min_score_ratio: float = 0.70
    hoc_rerank_num_subwindows: int | None = None
    evm_rerank_enabled: bool = False
    evm_label_aware: bool = False
    evm_rerank_top_n: int = 25
    evm_rerank_pool_size: int = 50
    evm_samples_per_symbol: int = 8
    evm_rrc_beta: float = 0.35
    evm_rrc_span: int = 8
    evm_timing_phases: tuple[int, ...] | None = (0,)
    evm_phase_grid_size: int = 1
    evm_max_symbols: int = 512
    evm_symbol_trim: int = 8
    evm_modulations: tuple[str, ...] = ("QPSK", "16QAM", "64QAM", "16APSK", "32APSK")
    evm_refine_all_candidates: bool = True
    evm_fine_search_radius_hz_per_s: float = 800.0
    evm_fine_search_step_hz_per_s: float = 80.0

    compensate_fd0: bool = False
    centered_time: bool = False
    num_subwindows: int = 10

    hoc_orders: tuple[tuple[int, int], ...] = ((2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (6, 0), (6, 3))
    hoc_representation: str = "real_imag_abs"

    # Direct FRFT validation parameters.
    alpha_grid_size: int = 101
    alpha_to_mu_scale: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DRCHOCExtractor:
    """End-to-end Doppler-rate-compensated HOC extractor."""

    def __init__(self, config: DRCHOCConfig | None = None, estimator: Any | None = None) -> None:
        self.config = config or DRCHOCConfig()
        self.estimator = estimator or self._build_estimator(self.config)

    def _build_estimator(self, cfg: DRCHOCConfig):
        name = cfg.estimator_type.lower()
        if name in {"paper_strict_dfrft", "paper_strict_frft", "strict_dfrft"}:
            return PaperStrictDFRFTEstimator(
                sample_rate_hz=cfg.sample_rate_hz,
                candidate_orders=cfg.candidate_orders,
                alpha_steps=cfg.alpha_steps,
                alpha_min=cfg.alpha_min,
                calibration_scale=cfg.calibration_scale,
                calibration_bias=cfg.calibration_bias,
                delta=cfg.delta,
                peak_exclude_bins=cfg.peak_exclude_bins,
                peak_eta=cfg.peak_eta,
                centered_time=cfg.centered_time,
                mu_min_hz_per_s=cfg.mu_grid_min,
                mu_max_hz_per_s=cfg.mu_grid_max,
                dc_block=cfg.dc_block,
                mu_search_guard_fraction=cfg.mu_search_guard_fraction,
                boundary_penalty_weight=cfg.boundary_penalty_weight,
                boundary_penalty_fraction=cfg.boundary_penalty_fraction,
                peak_margin_rel_floor=cfg.peak_margin_rel_floor,
                peak_margin_exclude_bins=cfg.peak_margin_exclude_bins,
                boundary_reject_fraction=cfg.boundary_reject_fraction,
                boundary_reject_margin_rel=cfg.boundary_reject_margin_rel,
                global_reject_margin_rel=cfg.global_reject_margin_rel,
                enable_fallback=cfg.enable_fallback,
                fallback_mu_hz_per_s=cfg.fallback_mu_hz_per_s,
                hoc_rerank_top_n=cfg.hoc_rerank_top_n,
                hoc_rerank_pool_size=cfg.hoc_rerank_pool_size,
                hoc_rerank_weight=cfg.hoc_rerank_weight,
                hoc_rerank_min_margin_rel=cfg.hoc_rerank_min_margin_rel,
                hoc_rerank_min_score_ratio=cfg.hoc_rerank_min_score_ratio,
                hoc_rerank_num_subwindows=cfg.hoc_rerank_num_subwindows or cfg.num_subwindows,
                hoc_rerank_orders=cfg.hoc_orders,
                hoc_rerank_representation=cfg.hoc_representation,
                evm_rerank_enabled=cfg.evm_rerank_enabled,
                evm_rerank_top_n=cfg.evm_rerank_top_n,
                evm_rerank_pool_size=cfg.evm_rerank_pool_size,
                evm_samples_per_symbol=cfg.evm_samples_per_symbol,
                evm_rrc_beta=cfg.evm_rrc_beta,
                evm_rrc_span=cfg.evm_rrc_span,
                evm_timing_phases=cfg.evm_timing_phases,
                evm_phase_grid_size=cfg.evm_phase_grid_size,
                evm_max_symbols=cfg.evm_max_symbols,
                evm_symbol_trim=cfg.evm_symbol_trim,
                evm_modulations=cfg.evm_modulations,
                evm_refine_all_candidates=cfg.evm_refine_all_candidates,
                evm_fine_search_radius_hz_per_s=cfg.evm_fine_search_radius_hz_per_s,
                evm_fine_search_step_hz_per_s=cfg.evm_fine_search_step_hz_per_s,
            )
        if name == "dechirp_fft":
            mu_grid = np.linspace(cfg.mu_grid_min, cfg.mu_grid_max, cfg.mu_grid_size)
            return DechirpFFTGridEstimator(
                sample_rate_hz=cfg.sample_rate_hz,
                mu_grid_hz_per_s=mu_grid,
                candidate_orders=cfg.candidate_orders,
                centered_time=cfg.centered_time,
                include_direct_power=False,
                include_phase_normalized=True,
            )
        if name == "frft":
            alpha_grid = np.linspace(0.05, np.pi - 0.05, cfg.alpha_grid_size)
            return FRFTGridEstimator(
                sample_rate_hz=cfg.sample_rate_hz,
                alpha_grid=alpha_grid,
                candidate_orders=cfg.candidate_orders,
                alpha_to_mu_scale=cfg.alpha_to_mu_scale,
                include_direct_power=False,
                include_phase_normalized=True,
            )
        raise ValueError(f"Unsupported estimator_type: {cfg.estimator_type}")

    @property
    def feature_names(self) -> list[str]:
        return hoc_feature_names(self.config.hoc_orders, self.config.hoc_representation)

    def extract(
        self,
        x: np.ndarray,
        fd0_hat_hz: float = 0.0,
        mu_true_hz_per_s: float | None = None,
        modulation_hint: str | None = None,
        return_compensated_signal: bool = False,
    ) -> dict[str, Any]:
        """Extract h_drc and metadata.

        Estimation may use normalized power branches, but the final quadratic
        phase compensation is applied to the original non-normalized signal so
        that HOC features keep amplitude information.
        """
        r_in = to_complex(x)

        # White paper assumes coarse synchronization has removed most constant
        # frequency offset. If fd0_hat is provided, we can optionally remove it.
        if self.config.compensate_fd0 and abs(fd0_hat_hz) > 0:
            r_base = compensate_constant_doppler(r_in, self.config.sample_rate_hz, fd0_hat_hz, self.config.centered_time)
        else:
            r_base = r_in

        evm_modulations = None
        if self.config.evm_label_aware and modulation_hint:
            evm_modulations = (str(modulation_hint).upper(),)

        if isinstance(self.estimator, PaperStrictDFRFTEstimator):
            est: EstimationResult = self.estimator.estimate(r_base, evm_modulations=evm_modulations)
        else:
            est = self.estimator.estimate(r_base)

        # Critical point: compensate original non-normalized base signal, not y_k.
        r_comp = conjugate_quadratic_compensation(r_base, self.config.sample_rate_hz, est.mu_hat_hz_per_s, self.config.centered_time)

        h_drc = extract_hoc_features(
            r_comp,
            orders=self.config.hoc_orders,
            representation=self.config.hoc_representation,
            normalize_power=True,
        )
        V_hoc = compute_V_hoc(
            r_comp,
            num_subwindows=self.config.num_subwindows,
            orders=self.config.hoc_orders,
            representation=self.config.hoc_representation,
        )

        T = self.config.observation_time_s if self.config.observation_time_s is not None else len(r_comp) / self.config.sample_rate_hz
        gamma_hat = float(compute_gamma(est.mu_hat_hz_per_s, T))

        out: dict[str, Any] = {
            "h_drc": h_drc.astype(np.float32),
            "mu_hat": float(est.mu_hat_hz_per_s),
            "gamma_hat": gamma_hat,
            "S_peak": float(est.S_peak),
            "V_hoc": float(V_hoc),
            "selected_key": est.selected_key,
            "selected_k": int(est.selected_k),
            "alpha_hat": est.alpha_hat,
            "fd_peak_hz": est.fd_peak_hz,
            "estimator_score": float(est.score),
            "feature_names": self.feature_names,
            "valid_flag": True,
            "estimator_extra": est.extra or {},
        }
        if mu_true_hz_per_s is not None:
            out["gamma_res"] = residual_gamma(mu_true_hz_per_s, est.mu_hat_hz_per_s, T)
            out["mu_error"] = float(est.mu_hat_hz_per_s - mu_true_hz_per_s)
        if return_compensated_signal:
            out["x_comp"] = r_comp
        return out

    def extract_batch(self, signals: np.ndarray, fd0_hat_hz: np.ndarray | float | None = None, mu_true_hz_per_s: np.ndarray | float | None = None) -> dict[str, np.ndarray]:
        signals = np.asarray(signals)
        B = signals.shape[0]

        def value(v, i, default=0.0):
            if v is None:
                return default
            if np.isscalar(v):
                return float(v)
            return float(np.asarray(v)[i])

        records = []
        for i in range(B):
            records.append(self.extract(signals[i], fd0_hat_hz=value(fd0_hat_hz, i), mu_true_hz_per_s=None if mu_true_hz_per_s is None else value(mu_true_hz_per_s, i)))

        out = {
            "h_drc": np.stack([r["h_drc"] for r in records]).astype(np.float32),
            "mu_hat": np.array([r["mu_hat"] for r in records], dtype=np.float32),
            "gamma_hat": np.array([r["gamma_hat"] for r in records], dtype=np.float32),
            "S_peak": np.array([r["S_peak"] for r in records], dtype=np.float32),
            "V_hoc": np.array([r["V_hoc"] for r in records], dtype=np.float32),
            "selected_k": np.array([r["selected_k"] for r in records], dtype=np.int64),
            "valid_flag": np.array([r["valid_flag"] for r in records], dtype=bool),
        }
        if "gamma_res" in records[0]:
            out["gamma_res"] = np.array([r["gamma_res"] for r in records], dtype=np.float32)
            out["mu_error"] = np.array([r["mu_error"] for r in records], dtype=np.float32)
        return out
