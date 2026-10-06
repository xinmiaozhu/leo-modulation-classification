"""Smoke test for physics and signal modules.

Run from the project root:
    python -m pytest tests/test_physics_signal_smoke.py
"""

import numpy as np
import pytest

from src.physics.gamma_metric import (
    attenuation_near_zero_lower_bound,
    compute_gamma,
    constant_energy_iq_perturbation,
    discrete_attenuation_factor,
    finite_sample_attenuation_error_bound,
    fresnel_attenuation,
    fresnel_attenuation_magnitude,
)
from src.physics.leo_orbit import circular_pass_profile
from src.signal.signal_generator import LEOSignalGenerator, SignalSpec


def test_compute_gamma():
    gamma = compute_gamma(2000.0, 0.02)
    assert gamma > 0


def test_discrete_attenuation_and_orbit_profile():
    attenuation = discrete_attenuation_factor(
        np.asarray([0.0, 1.0, 4.0]),
        k=4,
        num_samples=512,
    )
    assert attenuation.shape == (3,)
    assert attenuation[0] == pytest.approx(1.0)
    assert attenuation[-1] < attenuation[1]

    profile = circular_pass_profile(
        carrier_hz=30.0e9,
        altitude_m=600.0e3,
        max_elevation_deg=60.0,
        min_elevation_deg=10.0,
        time_step_s=2.0,
    )
    assert profile.time_s.size > 10
    assert np.all(np.isfinite(profile.doppler_rate_hz_per_s))
    assert float(np.min(profile.doppler_rate_hz_per_s)) < 0.0


def test_fresnel_attenuation_matches_discrete_sum_and_bounds():
    gamma = np.asarray([0.0, 0.1, 0.4, 0.8, 2.0])
    num_samples = 8192
    for k in (1, 2, 4, 6):
        continuous = fresnel_attenuation(gamma, k=k)
        finite_complex = np.asarray(
            [
                np.mean(
                    np.exp(
                        1j
                        * k
                        * value
                        * (np.arange(num_samples, dtype=np.float64) / num_samples) ** 2
                    )
                )
                for value in gamma
            ]
        )
        error = np.abs(finite_complex - continuous)
        bound = finite_sample_attenuation_error_bound(gamma, k=k, num_samples=num_samples)
        assert np.all(error <= bound + 1e-12)
        assert np.all(
            fresnel_attenuation_magnitude(gamma, k=k)
            + 1e-12
            >= attenuation_near_zero_lower_bound(gamma, k=k)
        )

    positive = fresnel_attenuation(gamma, k=4, rate_sign=1.0)
    negative = fresnel_attenuation(gamma, k=4, rate_sign=-1.0)
    assert np.allclose(negative, np.conj(positive))


def test_constant_energy_iq_perturbation_identity():
    gamma = np.asarray([0.0, 0.1, 0.4, 0.8])
    expected = np.sqrt(
        np.maximum(0.0, 2.0 * (1.0 - np.real(fresnel_attenuation(gamma, k=1))))
    )
    assert np.allclose(constant_energy_iq_perturbation(gamma), expected)


def test_generate_signal():
    gen = LEOSignalGenerator(modulations=["BPSK", "QPSK"], rng=np.random.default_rng(0))
    spec = SignalSpec(
        modulation="QPSK",
        num_symbols=64,
        samples_per_symbol=4,
        sample_rate_hz=1e6,
        snr_db=10.0,
        fd0_hz=1000.0,
        mu_hz_per_s=2000.0,
        target_num_samples=256,
    )
    sample = gen.generate(spec)
    assert sample.iq.shape == (2, 256)
    assert sample.label == 1
    assert "gamma" in sample.info
