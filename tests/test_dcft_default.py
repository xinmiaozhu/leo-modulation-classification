"""Regression checks for the acquisition-only receiver."""

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.drc_hoc.dcft_estimator import DCFTConfig, DCFTPilotEstimator
from src.drc_hoc.pilot_estimator import PilotMuEstimator, PilotMuEstimatorConfig
from src.signal.pulse_shape import rrc_filter
from src.signal.signal_generator import LEOSignalGenerator, SignalSpec


def precompute_module():
    path = Path(__file__).resolve().parents[1] / "scripts/06_precompute_pilot_mu.py"
    spec = importlib.util.spec_from_file_location("precompute_dcft_default", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sample():
    return LEOSignalGenerator(rng=np.random.default_rng(72)).generate(
        SignalSpec(modulation="QPSK", num_symbols=1024, samples_per_symbol=8,
                   sample_rate_hz=200000, target_num_samples=8192, snr_db=60,
                   fd0_hz=-321, mu_hz_per_s=-4321, pilot_enabled=True,
                   pilot_pattern="comb", pilot_interval_symbols=16, pilot_guard_symbols=0)
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cpu_and_batch_are_acquisition_only(sample, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA parity test")
    config = PilotMuEstimatorConfig(fd0_min_hz=-550, fd0_max_hz=550,
                                    pilot_weighting="coherent")
    hybrid = DCFTConfig()
    estimator = DCFTPilotEstimator(config, hybrid)
    def score_grid(*args):
        pytest.fail("Acquisition must never execute a coherent grid")

    monkeypatch.setattr(estimator.pilot, "_score_grid", score_grid)
    result = estimator.estimate(sample.iq, sample.pilot_indices, sample.pilot_symbols)
    assert result.estimate.valid
    assert result.estimate.mu_hat_hz_per_s == result.coarse_mu_hz_per_s
    assert result.estimate.fd0_hat_hz == result.coarse_fd0_hz
    # Exercise the same tensor code even on test hosts without CUDA.
    batch = precompute_module()._estimate_dcft_cuda_batch(
        sample.iq[None], sample.pilot_indices[None], sample.pilot_symbols[None],
        config=config, dcft_config=hybrid, device=torch.device(device),
        rrc_taps=rrc_filter(.35, 8, 8),
    )
    np.testing.assert_allclose(batch[0][0], result.estimate.mu_hat_hz_per_s, atol=5)
    np.testing.assert_allclose(batch[1][0], result.estimate.fd0_hat_hz, atol=1)


def test_default_cli_skips_refinement_and_records_acquisition(sample, tmp_path, monkeypatch):
    raw, output = tmp_path / "raw.h5", tmp_path / "pilot.h5"
    with h5py.File(raw, "w") as handle:
        handle["iq"] = sample.iq[None]
        handle["pilot_indices"] = sample.pilot_indices[None]
        handle["pilot_symbols"] = sample.pilot_symbols[None]
        handle["fs"] = [200000.]

    def unexpected_refinement(*args):
        pytest.fail("Default CPU CLI executed coherent refinement")

    monkeypatch.setattr(PilotMuEstimator, "_score_grid", unexpected_refinement)
    monkeypatch.setattr(sys, "argv", ["precompute", "--raw-data", str(raw),
                                      "--output", str(output), "--no-progress"])
    precompute_module().main()
    with h5py.File(output) as handle:
        assert handle.attrs["pilot_estimator_type"] == "dcft_acquisition"
        assert "fine_mu_step_hz_per_s" not in json.loads(handle.attrs["dcft_config_json"])
        assert handle["pilot_valid"][0]
        assert abs(handle["pilot_mu_hat"][0] + 4321) < 160
        assert abs(handle["pilot_fd0_hat"][0] + 321) < 5
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["estimator_type"] == "dcft_acquisition"
    assert "fine_mu_step_hz_per_s" not in summary["dcft_config"]


def test_explicit_legacy_modes_keep_their_cfo_defaults(monkeypatch):
    module = precompute_module()
    for mode, bounds in [("coherent_grid", (0., 0.)), ("dcft_acquisition", (-550., 550.)), ("dfrft_only", (-550., 550.))]:
        monkeypatch.setattr(sys, "argv", ["precompute", "--raw-data", "raw.h5",
                                          "--output", "pilot.h5", "--estimator-type", mode])
        args = module.parse_args()
        assert (args.fd0_min, args.fd0_max) == bounds
