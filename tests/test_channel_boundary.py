import numpy as np
import pytest

from src.signal.channel import NonStationaryLEOChannel
from src.signal.multipath_equalization import (
    channel_operator, compensated_path_coefficients, oracle_equalize,
)
from src.signal.signal_generator import LEOSignalGenerator, SignalSpec


def test_zero_path_doppler_recovers_existing_static_model():
    x = np.random.default_rng(2).normal(size=100) + 1j
    a = NonStationaryLEOChannel(200000, 30e9, np.random.default_rng(3))
    b = NonStationaryLEOChannel(200000, 30e9, np.random.default_rng(3))
    static, taps = a.apply_static_multipath(x, (0, 3, 7), return_taps=True)
    varying, initial = b.apply_time_varying_multipath(x, (0, 3, 7), (0, -6, -10), (0, 0, 0))
    np.testing.assert_allclose(static, varying, atol=1e-14)
    np.testing.assert_allclose(taps, initial)


def test_operator_adjoint_compensation_and_inverse():
    rng = np.random.default_rng(11)
    n, fs = 256, 2000
    x = rng.normal(size=n) + 1j * rng.normal(size=n)
    y = rng.normal(size=n) + 1j * rng.normal(size=n)
    delays = [0, 3]
    taps = np.array([1, 0, 0, .2j])
    c = compensated_path_coefficients(taps, delays, [0, 10], n, fs, -500, 30)
    a = channel_operator(c, delays)
    np.testing.assert_allclose(np.vdot(a @ x, y), np.vdot(x, a.H @ y), atol=1e-12)
    t = np.arange(n) / fs
    phase = np.exp(2j * np.pi * (30 * t - 250 * t**2))
    h = channel_operator(compensated_path_coefficients(taps, delays, [0, 10], n, fs), delays)
    np.testing.assert_allclose(a @ x, phase.conj() * (h @ (phase * x)), atol=1e-12)
    restored, diagnostic = oracle_equalize(a @ x, c, delays, regularization=0)
    np.testing.assert_allclose(restored, x, atol=1e-5)
    assert diagnostic["stop_code"] != 7


def test_pairing_preserves_symbols_and_noise_despite_path_count():
    def generate(delays, gains, snr):
        generator = LEOSignalGenerator(["QPSK"], np.random.default_rng(31))
        return generator.generate(SignalSpec(
            modulation="QPSK", channel_type="multipath", multipath_delays=delays,
            multipath_gains_db=gains, snr_db=snr,
        ), channel_rng=np.random.default_rng(32), noise_rng=np.random.default_rng(33))
    a, a_clean = generate((0,), (0,), 10), generate((0,), (0,), np.inf)
    b, b_clean = generate((0, 3), (0, -6), 10), generate((0, 3), (0, -6), np.inf)
    np.testing.assert_array_equal(a.symbol_indices, b.symbol_indices)
    def residual(noisy, clean):
        return (noisy.complex_signal-clean.complex_signal) / np.sqrt(np.mean(abs(clean.complex_signal)**2))
    np.testing.assert_allclose(residual(a, a_clean), residual(b, b_clean), atol=1e-14)


def test_invalid_varying_paths_rejected():
    ch = NonStationaryLEOChannel(200000, 30e9)
    with pytest.raises(ValueError):
        ch.apply_time_varying_multipath(np.ones(10), (0, 0), (0, -6), (0, 1))


def test_runner_invalidates_cache_for_upstream_changes_and_missing_output(tmp_path, monkeypatch):
    import runpy
    from pathlib import Path
    from types import SimpleNamespace

    runner = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/51_run_channel_boundary.py"))
    run = runner["run_step"]
    monkeypatch.setitem(run.__globals__, "ROOT", tmp_path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/step.py").write_text("# fake preprocessing step")
    (tmp_path / "src").mkdir()
    (tmp_path / "logs").mkdir()
    source, output = tmp_path / "input.bin", tmp_path / "output.bin"
    source.write_bytes(b"first")
    calls = []

    def execute(*args, **kwargs):
        calls.append(args)
        output.write_bytes(source.read_bytes())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run.__globals__["subprocess"], "run", execute)
    arguments = ["--raw-data", source, "--output", output]
    run("task", "step.py", arguments, tmp_path, 1)
    run("task", "step.py", arguments, tmp_path, 1)
    assert len(calls) == 1
    source.write_bytes(b"upstream changed")
    run("task", "step.py", arguments, tmp_path, 1)
    assert output.read_bytes() == b"upstream changed"
    assert len(calls) == 2
    output.unlink()
    run("task", "step.py", arguments, tmp_path, 1)
    assert len(calls) == 3


def test_training_resume_uses_only_matching_pending_run(tmp_path, monkeypatch):
    import json
    import runpy
    from pathlib import Path
    from types import SimpleNamespace

    runner = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/51_run_channel_boundary.py"))
    run = runner["run_step"]
    monkeypatch.setitem(run.__globals__, "ROOT", tmp_path)
    for folder in ("scripts", "src", "logs", "checkpoint"):
        (tmp_path / folder).mkdir()
    (tmp_path / "scripts/13_train_model.py").write_text("# training stub")
    checkpoint = tmp_path / "checkpoint"
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        (checkpoint / "last.pt").write_bytes(b"checkpoint")
        if len(calls) == 1:
            return SimpleNamespace(returncode=1)
        (checkpoint / "best.pt").write_bytes(b"best")
        (checkpoint / "train_result.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run.__globals__["subprocess"], "run", execute)
    arguments = ["--output-dir", checkpoint]
    with pytest.raises(RuntimeError):
        run("train", "13_train_model.py", arguments, tmp_path, 1)
    run("train", "13_train_model.py", arguments, tmp_path, 1)
    assert "--resume" not in calls[0] and "--resume" in calls[1]
    assert json.loads((tmp_path / "run_progress.json").read_text())["state"] == "step_completed"


def test_evaluation_only_generation_preserves_paired_test_samples(tmp_path):
    import runpy
    import subprocess
    import sys
    from pathlib import Path
    import h5py

    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(str(root / "scripts/51_run_channel_boundary.py"))
    config = {"evaluation_only_nontraining_conditions": True, "training_conditions": ["flat"],
              "train_per_class_snr": 1, "val_per_class_snr": 1, "test_per_class_snr": 2}
    for condition in ("flat", "short"):
        args = runner["generation_split_arguments"](config, condition)
        subprocess.run([sys.executable, str(root / "scripts/03_generate_snr_balanced_eval_dataset.py"),
                        "--output", str(tmp_path / f"{condition}.h5"),
                        "--splits-output", str(tmp_path / f"{condition}.npz"),
                        "--modulations", "QPSK", "--snr-db", "5", "--num-symbols", "64",
                        "--target-num-samples", "512", "--paired-seeds", "--no-progress", *map(str, args)],
                       cwd=root, check=True, capture_output=True)
    with h5py.File(tmp_path / "flat.h5") as full, h5py.File(tmp_path / "short.h5") as only:
        indices = np.load(tmp_path / "flat.npz")["test"]
        for key in ("pair_key", "iq", "mu", "label", "pilot_symbols"):
            np.testing.assert_array_equal(full[key][indices], only[key][:])
        assert len(np.load(tmp_path / "short.npz")["train"]) == 0
