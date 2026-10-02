"""Tests for the paper-strict DRC-HOC pipeline."""

import numpy as np

from src.drc_hoc.nonlinear_transforms import generate_blind_normalized_candidates
from src.drc_hoc.doppler_rate_estimator import PaperStrictDFRFTEstimator
from src.drc_hoc.extractor import DRCHOCConfig, DRCHOCExtractor
from src.signal.signal_generator import LEOSignalGenerator, SignalSpec


def test_blind_normalized_candidates_only():
    rng = np.random.default_rng(0)
    x = rng.normal(size=128) + 1j * rng.normal(size=128)
    cands = generate_blind_normalized_candidates(x, candidate_orders=(2, 4, 8))
    assert list(cands.keys()) == ["norm_power_2", "norm_power_4", "norm_power_8"]
    for y in cands.values():
        assert np.all(np.isfinite(y))


def test_paper_strict_estimator_returns_alpha_and_k():
    t = np.arange(256) / 1e6
    x = np.exp(1j * np.pi * 1000.0 * t**2)
    est = PaperStrictDFRFTEstimator(
        sample_rate_hz=1e6,
        candidate_orders=(2, 4),
        alpha_steps=101,
        mu_min_hz_per_s=-2000,
        mu_max_hz_per_s=2000,
    )
    out = est.estimate(x)
    assert np.isfinite(out.mu_hat_hz_per_s)
    assert out.alpha_hat is not None
    assert out.selected_k in [1, 2, 4]


def test_extractor_paper_strict_small():
    rng = np.random.default_rng(0)
    gen = LEOSignalGenerator(modulations=["QPSK"], rng=rng)
    spec = SignalSpec(
        modulation="QPSK",
        num_symbols=64,
        samples_per_symbol=4,
        sample_rate_hz=1e6,
        snr_db=20,
        fd0_hz=0,
        mu_hz_per_s=1000,
        target_num_samples=256,
    )
    s = gen.generate(spec)
    cfg = DRCHOCConfig(
        sample_rate_hz=1e6,
        estimator_type="paper_strict_dfrft",
        alpha_steps=101,
        mu_grid_min=-2000,
        mu_grid_max=2000,
        candidate_orders=(2, 4, 8),
        num_subwindows=4,
    )
    ext = DRCHOCExtractor(cfg)
    out = ext.extract(s.iq, mu_true_hz_per_s=1000)
    assert out["h_drc"].ndim == 1
    assert out["selected_key"].startswith("norm_power") or out["selected_key"] == "none"
    assert np.isfinite(out["S_peak"])
    assert np.isfinite(out["gamma_hat"])
