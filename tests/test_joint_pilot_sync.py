import numpy as np

from src.drc_hoc.pilot_estimator import PilotMuEstimator, PilotMuEstimatorConfig
from src.signal.signal_generator import LEOSignalGenerator, SignalSpec


def _generator(seed: int = 19) -> LEOSignalGenerator:
    return LEOSignalGenerator(modulations=["QPSK"], rng=np.random.default_rng(seed))


def test_doptimal_pilots_keep_requested_count_and_aperture():
    sample = _generator().generate(
        SignalSpec(
            modulation="QPSK",
            num_symbols=1024,
            samples_per_symbol=8,
            sample_rate_hz=200_000.0,
            target_num_samples=8192,
            pilot_enabled=True,
            pilot_pattern="d_optimal",
            pilot_num_symbols=64,
            pilot_guard_symbols=0,
        )
    )

    assert len(sample.pilot_indices) == 64
    assert int(sample.pilot_indices[0]) == 0
    assert int(sample.pilot_indices[-1]) == 1023
    assert np.all(np.diff(sample.pilot_indices) > 0)


def test_joint_cfo_doppler_search_recovers_high_snr_frame():
    fd0_true = 300.0
    mu_true = -4000.0
    sample = _generator().generate(
        SignalSpec(
            modulation="QPSK",
            num_symbols=1024,
            samples_per_symbol=8,
            sample_rate_hz=200_000.0,
            target_num_samples=8192,
            snr_db=30.0,
            fd0_hz=fd0_true,
            mu_hz_per_s=mu_true,
            pilot_enabled=True,
            pilot_pattern="d_optimal",
            pilot_num_symbols=64,
            pilot_guard_symbols=0,
        )
    )
    estimator = PilotMuEstimator(
        PilotMuEstimatorConfig(
            sample_rate_hz=200_000.0,
            samples_per_symbol=8,
            timing_phases=(0,),
            mu_min=-8160.0,
            mu_max=-180.0,
            coarse_step_hz_per_s=160.0,
            fine_radius_hz_per_s=200.0,
            fine_step_hz_per_s=5.0,
            fd0_min_hz=-500.0,
            fd0_max_hz=500.0,
            fd0_step_hz=100.0,
            fine_fd0_radius_hz=120.0,
            fine_fd0_step_hz=5.0,
            pilot_weighting="coherent",
        )
    )
    result = estimator.estimate(sample.iq, sample.pilot_indices, sample.pilot_symbols)

    assert result.valid
    assert abs(result.mu_hat_hz_per_s - mu_true) <= 10.0
    assert abs(result.fd0_hat_hz - fd0_true) <= 10.0


def test_integer_timing_shift_updates_frame_start_cfo_reference():
    fs = 200_000.0
    timing = 3
    mu = -4000.0
    fd0 = 300.0
    sample = _generator().generate(
        SignalSpec(
            modulation="QPSK",
            num_symbols=64,
            samples_per_symbol=8,
            sample_rate_hz=fs,
            target_num_samples=512,
            snr_db=20.0,
            fd0_hz=fd0,
            mu_hz_per_s=mu,
            pilot_enabled=True,
            pilot_pattern="comb",
            pilot_interval_symbols=8,
            timing_offset_samples=timing,
        )
    )
    expected = fd0 - mu * timing / fs
    assert np.isclose(sample.info["fd0_frame_start_hz"], expected)
