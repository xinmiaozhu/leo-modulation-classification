"""Smoke tests for src/drc_hoc."""

import numpy as np

from src.signal.signal_generator import LEOSignalGenerator, SignalSpec
from src.drc_hoc.cumulants import extract_hoc_features
from src.drc_hoc.extractor import DRCHOCConfig, DRCHOCExtractor
from src.drc_hoc.frft import frft_direct


def test_cumulant_features_shape():
    rng = np.random.default_rng(0)
    x = rng.normal(size=256) + 1j * rng.normal(size=256)
    h = extract_hoc_features(x)
    assert h.ndim == 1
    assert h.size > 0
    assert np.all(np.isfinite(h))


def test_frft_direct_shape():
    rng = np.random.default_rng(0)
    x = rng.normal(size=64) + 1j * rng.normal(size=64)
    y = frft_direct(x, alpha=0.7)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y.real))


def test_drc_hoc_extractor():
    rng = np.random.default_rng(0)
    gen = LEOSignalGenerator(modulations=["QPSK"], rng=rng)
    spec = SignalSpec(
        modulation="QPSK",
        num_symbols=64,
        samples_per_symbol=4,
        sample_rate_hz=1e6,
        snr_db=20.0,
        fd0_hz=0.0,
        mu_hz_per_s=1000.0,
        target_num_samples=256,
    )
    sample = gen.generate(spec)

    cfg = DRCHOCConfig(
        sample_rate_hz=1e6,
        mu_grid_min=-2000,
        mu_grid_max=2000,
        mu_grid_size=41,
        alpha_steps=41,
        estimator_type="paper_strict_dfrft",
        candidate_orders=(2, 4),
        num_subwindows=4,
    )
    extractor = DRCHOCExtractor(cfg)
    out = extractor.extract(sample.iq, mu_true_hz_per_s=1000.0)

    assert out["h_drc"].ndim == 1
    assert np.isfinite(out["mu_hat"])
    assert np.isfinite(out["gamma_hat"])
    assert np.isfinite(out["S_peak"])
    assert np.isfinite(out["V_hoc"])
