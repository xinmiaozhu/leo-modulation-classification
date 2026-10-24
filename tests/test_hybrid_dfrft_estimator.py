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


def test_frft_czt_matches_independent_dense_kernel():
    from src.drc_hoc.frft import SampledFrFTPlan
    rng = np.random.default_rng(73)
    z = rng.normal(size=64) + 1j * rng.normal(size=64)
    for angle in (1.37, np.pi / 2, 1.73):
        plan = SampledFrFTPlan(64, angle, 256)
        np.testing.assert_allclose(plan.transform(z), plan.direct_reference(z), atol=2e-11)


def test_frft_physical_units_and_off_grid_refinement():
    from src.drc_hoc.frft import rate_to_frft_angle
    from src.drc_hoc.hybrid_dfrft_estimator import FrFTPilotEstimator, FrFTCaptureConfig
    from src.drc_hoc.pilot_estimator import make_grid
    config = PilotMuEstimatorConfig(mu_min=-8160, mu_max=-180, fd0_min_hz=-550, fd0_max_hz=550)
    estimator = FrFTPilotEstimator(config, HybridDFRFTConfig(), FrFTCaptureConfig(65, 2048))
    dt = 16 * 8 / 200000
    for start in (0., .003):
        t = start + np.arange(64) * dt
        for mu, f0 in ((-4321.3, -321.7), (-8100.3, 499.1), (-180.1, -499.2)):
            alpha = rate_to_frft_angle(mu, 64, dt)
            np.testing.assert_allclose(-1 / np.tan(alpha) / (64 * dt**2), mu, atol=1e-9)
            z = np.exp(1j * (.37 + 2 * np.pi * f0 * t + np.pi * mu * t**2))
            cm, fc, cf, *_ = estimator._chirp_focus(z, t)
            np.testing.assert_allclose(cf, fc - cm * t.mean(), atol=1e-12)
            assert abs(cm - mu) < 150
            assert abs(cf - f0) < 5
            mg = make_grid(max(-8160, cm - 200), min(-180, cm + 200), 5)
            fg = make_grid(max(-550, cf - 30), min(550, cf + 30), 1)
            rm, rf, *_ = estimator.pilot._score_grid(z, t, mg, fg)
            # CFO/rate errors are coupled on the finite observation interval.
            assert abs(rm - mu) < 40
            assert abs(rf - f0) < 1


def test_frft_rejects_nonuniform_pilot_times():
    import pytest
    from src.drc_hoc.hybrid_dfrft_estimator import FrFTPilotEstimator
    t = np.arange(64) * .00064
    t[20] += .00001
    with pytest.raises(ValueError, match="uniformly"):
        FrFTPilotEstimator(PilotMuEstimatorConfig())._chirp_focus(np.ones(64), t)


def test_stratified_paired_bootstrap_preserves_stratum_weights_and_pairing():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'scripts/28_summarize_frontend_attribution.py'
    spec = importlib.util.spec_from_file_location('comparison_report', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Unequal strata and deterministic within-stratum differences: the correct
    # stratified interval collapses, while pooled resampling would add variance.
    a = np.array([1,1,1,0,0])
    b = 1-a
    labels = np.array([0,0,0,1,1])
    snr = np.array([-5,-5,-5,10,10])
    result = module.stratified_paired_accuracy_interval(a,b,labels,snr)
    assert result['gain_pp'] == result['ci95_low'] == result['ci95_high'] == 20.
    zero = module.stratified_paired_accuracy_interval(a,a,labels,snr)
    assert zero['ci95_low'] == zero['ci95_high'] == 0.
    reverse = module.stratified_paired_accuracy_interval(b,a,labels,snr)
    assert reverse['ci95_low'] == reverse['ci95_high'] == -20.
