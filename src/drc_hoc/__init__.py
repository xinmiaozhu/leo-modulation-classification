"""DRC-HOC modules."""

from .nonlinear_transforms import (
    amplitude_normalize,
    power_transform,
    phase_normalized_power_transform,
    generate_blind_normalized_candidates,
    generate_candidates,
    parse_order_from_key,
)
from .frft import (
    DirectFRFT,
    frft_direct,
    frft_search_peak,
    peak_sharpness,
    local_peak_sharpness,
    make_alpha_grid,
    alpha_to_mu_whitepaper,
    fast_dfrft_chirp_focus,
)
from .doppler_rate_estimator import (
    EstimationResult,
    PaperStrictDFRFTEstimator,
    DechirpFFTGridEstimator,
    FRFTGridEstimator,
)
from .compensation import (
    conjugate_quadratic_compensation,
    compensate_constant_doppler,
    residual_gamma,
)
from .cumulants import (
    moment_pq,
    cumulant_pq,
    extract_hoc_features,
    hoc_feature_names,
)
from .reliability_metrics import compute_V_hoc, compute_feature_stability
from .extractor import DRCHOCConfig, DRCHOCExtractor
from .pilot_estimator import PilotMuEstimatorConfig, PilotMuEstimator, PilotMuResult
from .hybrid_dfrft_estimator import (
    HybridDFRFTConfig,
    HybridDFRFTPilotEstimator,
    HybridDFRFTResult,
)

__all__ = [
    "amplitude_normalize",
    "power_transform",
    "phase_normalized_power_transform",
    "generate_blind_normalized_candidates",
    "generate_candidates",
    "parse_order_from_key",
    "DirectFRFT",
    "frft_direct",
    "frft_search_peak",
    "peak_sharpness",
    "local_peak_sharpness",
    "make_alpha_grid",
    "alpha_to_mu_whitepaper",
    "fast_dfrft_chirp_focus",
    "EstimationResult",
    "PaperStrictDFRFTEstimator",
    "DechirpFFTGridEstimator",
    "FRFTGridEstimator",
    "conjugate_quadratic_compensation",
    "compensate_constant_doppler",
    "residual_gamma",
    "moment_pq",
    "cumulant_pq",
    "extract_hoc_features",
    "hoc_feature_names",
    "compute_V_hoc",
    "compute_feature_stability",
    "DRCHOCConfig",
    "DRCHOCExtractor",
    "PilotMuEstimatorConfig",
    "PilotMuEstimator",
    "PilotMuResult",
    "HybridDFRFTConfig",
    "HybridDFRFTPilotEstimator",
    "HybridDFRFTResult",
]
