"""Protocol-B orchestration and comparison integrity checks."""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_baselines_use_protocol_b_and_native_preprocessing(tmp_path, monkeypatch):
    runner = load_script("17_run_hybrid_dfrft_seeds.py")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    for path in runner.DATA.values():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    monkeypatch.setattr(sys, "argv", ["runner", "--methods", "cnn2", "cnn_lstm_dual",
                                     "satellite_cnn", "nasa_hoc_nn", "--seeds", "41", "--dry-run"])
    calls = []
    monkeypatch.setattr(runner, "run", lambda command, dry_run: calls.append(command))
    runner.main()
    assert len(calls) == 9
    assert calls[0][1].endswith("12_precompute_paper_baseline_features.py")
    for train, evaluate in zip(calls[1::2], calls[2::2]):
        def value(command, flag):
            return command[command.index(flag) + 1]
        assert value(train, "--raw-data") == runner.DATA["raw"]
        assert value(evaluate, "--raw-data") == runner.DATA["raw"]
        assert value(train, "--splits") == value(evaluate, "--splits") == runner.DATA["splits"]
        assert value(evaluate, "--split") == "test"
        assert value(train, "--feature-data") == value(evaluate, "--feature-data")
        model = value(train, "--model")
        if model == "paper_nasa_hoc_nn":
            assert value(train, "--feature-data") == runner.PAPER_FEATURE
            assert value(train, "--hoc-transform") == "none"
            assert "--no-hoc-standardize" not in train
        else:
            assert value(train, "--feature-data") == runner.DATA["feature"]
            expected = "power" if model == "paper_cnn_lstm_dual" else "zscore"
            assert value(train, "--iq-normalize") == expected
            assert value(train, "--iq-source") == "comp"
    assert not (tmp_path / "outputs").exists()


def test_summary_retains_existing_methods_and_repairs_missing_csv(tmp_path, monkeypatch):
    runner = load_script("17_run_hybrid_dfrft_seeds.py")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "DATA", {})
    result = tmp_path / "outputs/results/hybrid_dfrft/protocol_b"
    for method in ("proposed", "cnn2"):
        directory = result / method
        directory.mkdir(parents=True)
        (directory / "seed_41_test.summary.json").write_text("{}")
    pd.DataFrame({"correct": [1, 1]}).to_csv(result / "proposed/seed_41_test.csv", index=False)
    # A summary alone must not cause evaluation to be skipped.
    def evaluate(command, dry_run):
        assert command[1].endswith("14_evaluate_model.py")
        output = tmp_path / command[command.index("--output") + 1]
        pd.DataFrame({"correct": [1, 0]}).to_csv(output, index=False)
    monkeypatch.setattr(runner, "run", evaluate)
    runner.DATA.update(raw="raw", feature="feature", splits="splits")
    for path in runner.DATA.values():
        (tmp_path / path).touch()
    monkeypatch.setattr(sys, "argv", ["runner", "--methods", "cnn2", "--seeds", "41",
                                     "--stages", "eval", "summary"])
    runner.main()
    table = pd.read_csv(result / "seed_runs.csv")
    assert set(table.method) == {"proposed", "cnn2"}
    payload = json.loads((result / "summary.json").read_text())
    assert payload["paired"]["proposed_minus_cnn2"]["mean_delta_pp"] == 50


@pytest.mark.parametrize("mismatch", [False, True])
def test_seven_method_plot_checks_sample_alignment(tmp_path, monkeypatch, mismatch):
    plot = load_script("48_plot_protocol_b_accuracy_vs_snr.py")
    monkeypatch.setattr(plot, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["plot", "--seed", "41"])
    source = tmp_path / "outputs/results/hybrid_dfrft/protocol_b"
    for i, (method, *_) in enumerate(plot.METHODS):
        folder = source / method
        folder.mkdir(parents=True)
        pd.DataFrame({"index": [0, 1 if not (mismatch and i == 6) else 2],
                      "label": [0, 1], "snr_db": [-4, 8], "correct": [0, 1]}).to_csv(
                          folder / "seed_41_test.csv", index=False)
        (folder / "seed_41_test.summary.json").write_text(json.dumps(
            {"split": "test", "accuracy": 0.5, "checkpoint": "test-fixture"}))
    if mismatch:
        with pytest.raises(AssertionError):
            plot.main()
        plot.plt.close("all")
    else:
        plot.main()
        name = "paper_baseline_protocol_b_seed41_accuracy_vs_snr"
        assert (tmp_path / f"outputs/figures/paper/{name}.pdf").read_bytes().startswith(b"%PDF")
        assert pd.read_csv(source / f"{name}.csv").method.nunique() == 7
        assert json.loads((source / f"{name}.json").read_text())["missing_reference_methods"] == []
