"""Tests for APSK modulation support."""

import numpy as np

from src.signal.modulation import (
    available_modulations,
    get_constellation,
    Modulator,
)
from src.signal.signal_generator import LEOSignalGenerator, SignalSpec


def test_available_modulations_include_apsk():
    mods = available_modulations()
    assert "16APSK" in mods
    assert "32APSK" in mods


def test_apsk_constellation_sizes_and_power():
    for mod, M in [("16APSK", 16), ("32APSK", 32)]:
        const = get_constellation(mod)
        assert const.shape == (M,)
        power = np.mean(np.abs(const) ** 2)
        assert np.isclose(power, 1.0, atol=1e-6)


def test_apsk_signal_generation():
    gen = LEOSignalGenerator(modulations=["16APSK", "32APSK"], rng=np.random.default_rng(0))
    for mod in ["16APSK", "32APSK"]:
        spec = SignalSpec(
            modulation=mod,
            num_symbols=32,
            samples_per_symbol=4,
            sample_rate_hz=1e6,
            snr_db=10.0,
            fd0_hz=0.0,
            mu_hz_per_s=1000.0,
            target_num_samples=128,
        )
        sample = gen.generate(spec)
        assert sample.iq.shape == (2, 128)
        assert sample.modulation == mod
