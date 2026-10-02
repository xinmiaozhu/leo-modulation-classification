import numpy as np

from src.signal.impairments import apply_fractional_delay
from src.signal.signal_generator import LEOSignalGenerator, SignalSpec


def test_fractional_delay_preserves_shape_and_has_expected_centroid_shift():
    x = np.zeros(129, dtype=np.complex128)
    x[64] = 1.0
    y = apply_fractional_delay(x, 0.5)
    coordinate = np.arange(len(y), dtype=np.float64)
    centroid = float(np.sum(coordinate * np.abs(y) ** 2) / np.sum(np.abs(y) ** 2))
    assert y.shape == x.shape
    assert 64.35 < centroid < 64.65


def test_joint_impairments_are_recorded_and_finite():
    generator = LEOSignalGenerator(modulations=["QPSK"], rng=np.random.default_rng(7))
    sample = generator.generate(
        SignalSpec(
            modulation="QPSK",
            num_symbols=64,
            samples_per_symbol=4,
            sample_rate_hz=20_000.0,
            carrier_frequency_hz=2.0e9,
            target_num_samples=256,
            fd0_hz=125.0,
            mu_hz_per_s=-500.0,
            channel_type="rician",
            rician_k_db=8.0,
            fractional_timing_offset_samples=0.375,
            phase_noise_std_rad=1e-3,
        )
    )
    assert np.all(np.isfinite(sample.iq))
    assert sample.info["fractional_timing_offset_samples"] == 0.375
    assert sample.info["phase_noise_std_rad"] == 1e-3
    assert np.isclose(
        sample.info["fd0_frame_start_hz"],
        125.0 - (-500.0) * 0.375 / 20_000.0,
    )
