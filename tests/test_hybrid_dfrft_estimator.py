import numpy as np

from src.drc_hoc.hybrid_dfrft_estimator import HybridDFRFTConfig, HybridDFRFTPilotEstimator
from src.drc_hoc.pilot_estimator import PilotMuEstimatorConfig


def test_pilot_chirp_focus_recovers_rate_and_frame_start_cfo():
    config = PilotMuEstimatorConfig(
        sample_rate_hz=200_000.0,
        samples_per_symbol=8,
        mu_min=-8160.0,
        mu_max=-180.0,
        fd0_min_hz=-550.0,
        fd0_max_hz=550.0,
        pilot_weighting="coherent",
    )
    hybrid = HybridDFRFTPilotEstimator(
        config,
        HybridDFRFTConfig(coarse_mu_step_hz_per_s=160.0, fft_size=2048),
    )
    t = np.arange(64, dtype=np.float64) * (16 * 8 / config.sample_rate_hz)
    mu = -4321.0
    fd0 = -321.0
    z = np.exp(1j * (0.37 + 2.0 * np.pi * fd0 * t + np.pi * mu * t**2))
    coarse_mu, _, coarse_fd0, score, _, bin_spacing, count = hybrid._chirp_focus(z, t)

    assert np.isfinite(score)
    assert count > 10
    assert bin_spacing < 1.0
    assert abs(coarse_mu - mu) <= 0.5 * 160.0 + 1e-9
    assert abs(coarse_fd0 - fd0) <= 3.0
