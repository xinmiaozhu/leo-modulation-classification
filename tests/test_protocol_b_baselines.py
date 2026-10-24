"""Protocol-B orchestration and comparison integrity checks."""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("code,complete,accepted", [
    (0xC0000409, True, True), (0xC0000409, False, False), (1, True, False),
])
def test_training_exit_recovery_requires_known_code_and_verified_outputs(monkeypatch, code, complete, accepted):
    runner = load_script("16_run_hybrid_dfrft_seeds.py")
    command = ["python", "scripts/12_train_model.py"]
    def fail(*args, **kwargs):
        raise runner.subprocess.CalledProcessError(code, command)
    monkeypatch.setattr(runner.subprocess, "run", fail)
    monkeypatch.setattr(runner, "completed_training_artifacts", lambda *args: complete)
    if accepted:
        runner.run(command, False)
    else:
        with pytest.raises(runner.subprocess.CalledProcessError):
            runner.run(command, False)


def test_frozen_hoc_keeps_iq_and_matches_reference(tmp_path):
    from src.drc_hoc.cumulants import extract_hoc_features
    module = load_script("08_precompute_external_mu_hoc.py")
    rng = np.random.default_rng(15)
    iq = rng.normal(size=(3, 2, 128)).astype(np.float32)
    source, raw, output = [tmp_path / name for name in ("source.h5", "raw.h5", "hoc.h5")]
    for path in (source, raw):
        with h5py.File(path, "w") as f:
            f["iq_comp" if path == source else "iq"] = iq
            f["label"] = [0, 1, 0]
            f["snr_db"] = [0., 5., 10.]
            if path == source:
                f["h_drc"] = np.zeros((3, 21), dtype=np.float32)
    module.prepare_frozen_iq_hoc(source, raw, output, progress=False)
    with h5py.File(output, "r") as f:
        assert f.attrs["hoc_complete"]
        assert isinstance(f.get("iq_comp", getlink=True), h5py.ExternalLink)
        np.testing.assert_array_equal(f["iq_comp"][:], iq)
        reference = np.stack([extract_hoc_features(row) for row in iq])
        np.testing.assert_allclose(f["h_drc"][:], reference, rtol=1e-5, atol=1e-6)
    with h5py.File(source, "r") as f:
        assert not np.any(f["h_drc"][:])
    module.prepare_frozen_iq_hoc(source, raw, output, progress=False)
    with h5py.File(output, "r+") as f:
        f.attrs["hoc_complete"] = False
    with pytest.raises(ValueError, match="Incomplete or stale"):
        module.prepare_frozen_iq_hoc(source, raw, output, progress=False)
    with pytest.raises(ValueError, match="must not overwrite"):
        module.prepare_frozen_iq_hoc(source, raw, source, progress=False)


@pytest.mark.parametrize("protocol", ["A", "B"])
@pytest.mark.parametrize("method", ["fusion_hoc", "iq_hoc"])
def test_final_hoc_runner_uses_frozen_inputs_and_checkpoint_config(tmp_path, monkeypatch, protocol, method):
    runner = load_script("17_run_representation_ablation_seeds.py")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    cfg = runner.PROTOCOLS[protocol]
    for path in cfg.values():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    calls = []
    monkeypatch.setattr(runner, "run", lambda command, log_path, dry_run=False: calls.append(command))
    runner.main(["--protocols", protocol, "--methods", method, "--dry-run"])
    train = [c for c in calls if c[1].endswith("12_train_model.py")]
    evaluate = [c for c in calls if c[1].endswith("13_evaluate_model.py")]
    assert len(train) == len(evaluate) == 5
    assert [int(c[c.index("--seed") + 1]) for c in train] == runner.SEEDS
    for t, e in zip(train, evaluate):
        assert "--use-hoc-stream" in t
        if method == "iq_hoc":
            assert t[t.index("--model-config") + 1].endswith("dualnet_iq_only.yaml")
            assert t[t.index("--hoc-hidden-dims") + 1:t.index("--hoc-hidden-dims") + 3] == ["64", "64"]
        else:
            assert "--hoc-hidden-dims" not in t
        assert "--model-config" not in e
        assert e[e.index("--split") + 1] == "test"
        assert t[t.index("--feature-data") + 1] == runner.hoc_path(cfg["train_feature"])
        assert e[e.index("--feature-data") + 1] == runner.hoc_path(cfg["eval_feature"])
    if protocol == "B":
        assert "hybrid_dfrft" in cfg["train_feature"]
        assert "hybrid_dfrft" in cfg["train_symbol"]
    assert not (tmp_path / "outputs").exists()


def test_hoc_paired_summary_retains_existing_methods_and_rejects_bad_indices(tmp_path, monkeypatch):
    runner = load_script("17_run_representation_ablation_seeds.py")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    cfg = dict(runner.PROTOCOLS["B"])
    cfg.update(eval_raw="raw.h5", eval_splits="split.npz")
    monkeypatch.setitem(runner.PROTOCOLS, "B", cfg)
    with h5py.File(tmp_path / "raw.h5", "w") as f:
        f["label"] = [0, 1, 0, 1]
        f["snr_db"] = [0., 0., 5., 5.]
    np.savez(tmp_path / "split.npz", test=np.arange(4))
    for seed in runner.SEEDS:
        for method in runner.DISPLAY:
            output = (runner.fusion_paths("B", seed)[1] if method == "fusion"
                      else runner.branch_paths("B", method, seed)[1])
            output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"index": range(4), "label": [0, 1, 0, 1], "snr_db": [0., 0., 5., 5.],
                          "pred": ([0, 1, 0, 1] if method == "fusion_hoc" else
                                   [1, 1, 1, 1] if method == "iq_hoc" else [0, 1, 1, 1])}).to_csv(output, index=False)
    runner.summarize("B", ["fusion_hoc"], runner.SEEDS)
    root = runner.result_root("B")
    assert len(pd.read_csv(root / "seed_runs.csv")) == 25
    summary = json.loads((root / "summary.json").read_text())
    assert summary["formal_five_seed_complete"]
    assert summary["paired_hoc_gain"]["mean_pp"] == 25
    assert summary["paired_hoc_gain"]["sample_sd_pp"] == 0
    assert summary["paired_descriptor_vs_hoc"]["mean_pp"] == 25
    assert summary["paired_descriptor_vs_hoc"]["positive_seeds"] == 5
    assert summary["paired_iq_hoc_gain"]["mean_pp"] == -25
    comparison = pd.read_csv(root / "paired_descriptor_vs_hoc.csv")
    assert comparison.seed.tolist() == runner.SEEDS
    assert (comparison.descriptor_gain_over_hoc_pp == 25).all()
    bad = runner.branch_paths("B", "fusion_hoc", 41)[1]
    table = pd.read_csv(bad)
    table.loc[0, "index"] = 1
    table.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="indices mismatch"):
        runner.summarize("B", ["fusion_hoc"], runner.SEEDS)


def test_missing_baselines_use_protocol_b_and_native_preprocessing(tmp_path, monkeypatch):
    runner = load_script("16_run_hybrid_dfrft_seeds.py")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    for path in runner.DATA.values():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    monkeypatch.setattr(sys, "argv", ["runner", "--methods", "cnn2", "cnn_lstm_dual",
                                     "satellite_cnn", "iqformer", "--seeds", "41", "--dry-run"])
    calls = []
    monkeypatch.setattr(runner, "run", lambda command, dry_run: calls.append(command))
    runner.main()
    assert len(calls) == 8
    for train, evaluate in zip(calls[::2], calls[1::2]):
        def value(command, flag):
            return command[command.index(flag) + 1]
        assert value(train, "--raw-data") == runner.DATA["raw"]
        assert value(evaluate, "--raw-data") == runner.DATA["raw"]
        assert value(train, "--splits") == value(evaluate, "--splits") == runner.DATA["splits"]
        assert value(evaluate, "--split") == "test"
        assert value(train, "--feature-data") == value(evaluate, "--feature-data")
        model = value(train, "--model")
        assert value(train, "--feature-data") == runner.DATA["feature"]
        expected = "power" if model == "paper_cnn_lstm_dual" else "zscore"
        assert value(train, "--iq-normalize") == expected
        assert value(train, "--iq-source") == "comp"
        assert "--model-config" not in evaluate
    assert not (tmp_path / "outputs").exists()


def test_summary_retains_existing_methods_and_repairs_missing_csv(tmp_path, monkeypatch):
    runner = load_script("16_run_hybrid_dfrft_seeds.py")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "validate_test_predictions", lambda *args: None)
    monkeypatch.setattr(runner, "DATA", {})
    result = tmp_path / "outputs/results/hybrid_dfrft/protocol_b"
    for method in ("proposed", "cnn2"):
        directory = result / method
        directory.mkdir(parents=True)
        (directory / "seed_41_test.summary.json").write_text("{}")
    pd.DataFrame({"correct": [1, 1]}).to_csv(result / "proposed/seed_41_test.csv", index=False)
    # A summary alone must not cause evaluation to be skipped.
    def evaluate(command, dry_run):
        assert command[1].endswith("13_evaluate_model.py")
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


@pytest.mark.parametrize("methods", [["iqformer"], ["cnn2", "cnn_lstm_dual", "satellite_cnn"]])
def test_iqformer_final_ab_runner_uses_matched_seeds_and_checkpoint_config(monkeypatch, methods):
    runner = load_script("16_run_hybrid_dfrft_seeds.py")
    calls = []
    monkeypatch.setattr(runner, "run", lambda command, dry_run: calls.append(command))
    monkeypatch.setattr(sys, "argv", ["runner", "--protocols", "A", "B",
                                     "--methods", *methods, "--dry-run", "--overwrite"])
    runner.main()
    assert len(calls) == 20 * len(methods)
    train, evaluate = calls[::2], calls[1::2]
    assert [int(c[c.index("--seed") + 1]) for c in train] == [s for s in runner.SEEDS for _ in range(2 * len(methods))]
    for i, (t, e) in enumerate(zip(train, evaluate)):
        cfg = runner.protocol_inputs("A" if (i // len(methods)) % 2 == 0 else "B")
        assert t[t.index("--feature-data") + 1] == cfg["train_feature"]
        assert e[e.index("--feature-data") + 1] == cfg["eval_feature"]
        assert e[e.index("--split") + 1] == "test"
        assert "--symbol-feature-data" not in t
        assert "--model-config" not in e


def test_five_seed_curve_uses_sample_sd_and_rejects_missing_or_duplicate_seed():
    plot = load_script("47_plot_protocol_b_accuracy_vs_snr.py")
    seeds = [41, 73, 107, 149, 211]
    table = pd.DataFrame(dict(method=["model"] * 5, display_label=["Model"] * 5,
                              snr_db=[0.] * 5, seed=seeds, sample_count=[700] * 5,
                              accuracy_percent=[70., 75., 80., 85., 90.]))
    result = plot.aggregate_seed_curves(table, seeds).iloc[0]
    assert result.mean_percent == 80
    assert result.sample_sd_percent == pytest.approx(np.sqrt(62.5))
    with pytest.raises(ValueError, match="Missing matched seed"):
        plot.aggregate_seed_curves(table.iloc[:-1], seeds)
    with pytest.raises(ValueError, match="Duplicate"):
        plot.aggregate_seed_curves(pd.concat([table, table.iloc[:1]]), seeds)


def test_partial_campaign_does_not_publish_final_five_seed_means(tmp_path, monkeypatch):
    runner = load_script("16_run_hybrid_dfrft_seeds.py")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "protocol_inputs", lambda p: {})
    runner.five_seed_report(require_complete=False)
    output = tmp_path / "outputs/results/baseline_comparison"
    assert not (output / "seed_summary.csv").exists()
    assert len(json.loads((output / "completeness.json").read_text())["missing"]) == 70
    with pytest.raises(ValueError, match="incomplete"):
        runner.five_seed_report(require_complete=True)


def test_complete_campaign_publishes_percent_accuracy_and_paired_sample_sd(tmp_path, monkeypatch):
    runner = load_script("16_run_hybrid_dfrft_seeds.py")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    cfg = dict(eval_raw="raw.h5", eval_splits="splits.npz")
    monkeypatch.setattr(runner, "protocol_inputs", lambda p: cfg)
    with h5py.File(tmp_path / "raw.h5", "w") as f:
        f["label"] = [0, 1]
        f["snr_db"] = [-4., 8.]
    np.savez(tmp_path / "splits.npz", test=[0, 1])
    for protocol in ("A", "B"):
        for method in runner.METHODS:
            for seed in runner.SEEDS:
                path = runner.prediction_path(protocol, method, seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                perfect = method == "proposed" or seed == 41
                pd.DataFrame(dict(index=[0, 1], label=[0, 1], snr_db=[-4., 8.],
                                  pred=[0, int(perfect)], correct=[1, int(perfect)])).to_csv(path, index=False)
                path.with_suffix(".summary.json").write_text(json.dumps(dict(
                    split="test", accuracy=1.0 if perfect else 0.5)))
    costs = tmp_path / "outputs/results/baseline_comparison/protocol_a_accuracy_complexity.csv"
    costs.parent.mkdir(parents=True)
    pd.DataFrame(dict(method_id=["C1", "C2", "C3", "C4", "C5", "C6", "Proposed"],
                      method=list(range(7)), accuracy=[0.9] * 7, accuracy_percent=[90.] * 7,
                      parameter_count=[100] * 7, estimated_mmacs=[1.] * 7,
                      network_cpu_median_ms=[2.] * 7, end_to_end_cpu_median_ms=[3.] * 7)).to_csv(costs, index=False)
    runner.five_seed_report()
    output = tmp_path / "outputs/results/baseline_comparison"
    summary = pd.read_csv(output / "seed_summary.csv")
    assert len(summary) == 14 and (summary["count"] == 5).all()
    cnn = summary[(summary.protocol == "A") & (summary.method == "cnn2")].iloc[0]
    assert cnn.mean_percent == 60
    assert cnn.sample_sd_percent == pytest.approx(np.sqrt(500))
    gains = pd.read_csv(output / "paired_gains.csv")
    assert (gains.mean_gain_pp == 40).all() and (gains.positive_seeds == 4).all()
    cost = pd.read_csv(output / "protocol_a_accuracy_complexity.csv")
    assert "accuracy_percent" not in cost
    assert (cost.parameter_count == 100).all() and (cost["count"] == 5).all()
    paper = pd.read_csv(output / "table_iv.csv")
    assert (paper.training_seed_count == 5).all()
    assert set(paper.accuracy_display) == {"60.00 ± 22.36", "100.00 ± 0.00"}
    assert (paper.accuracy_seeds == "41,73,107,149,211").all()
    metadata = json.loads((output / "table_iv_metadata.json").read_text())
    assert "not training-seed" in metadata["latency_statistic"]
    assert json.loads((output / "completeness.json").read_text())["complete"]


def test_formal_plot_entries_default_to_all_five_seeds(monkeypatch):
    plot = load_script("47_plot_protocol_b_accuracy_vs_snr.py")
    calls = []
    monkeypatch.setattr(plot, "plot_five_seed", lambda protocol, output_dir=None: calls.append(protocol))
    monkeypatch.setattr(sys, "argv", ["plot"])
    plot.main()
    monkeypatch.setattr(sys, "argv", ["plot", "--protocol", "A"])
    plot.main()
    assert calls == ["B", "A"]
    monkeypatch.setattr(sys, "argv", ["plot", "--seed", "41"])
    with pytest.raises(SystemExit):
        plot.main()
    plot_a = load_script("46_plot_paper_baseline_comparison.py")
    monkeypatch.setattr(sys, "argv", ["plot", "--single-seed"])
    with pytest.raises(SystemExit):
        plot_a.main()


def test_table_iv_default_cannot_accept_single_seed_accuracy_override(monkeypatch):
    benchmark = load_script("40_benchmark_paper_models.py")
    monkeypatch.setattr(sys, "argv", ["benchmark"])
    assert benchmark.parse_args().table_iv
    monkeypatch.setattr(sys, "argv", ["benchmark", "--table-iv", "--summary", "seed41.csv"])
    with pytest.raises(SystemExit):
        benchmark.parse_args()
    monkeypatch.setattr(sys, "argv", ["benchmark", "--table-iv", "--accuracy-override", "proposed=0.9"])
    with pytest.raises(SystemExit):
        benchmark.parse_args()


def test_formal_figure_exports_mean_sd_and_publication_names(tmp_path, monkeypatch):
    from types import SimpleNamespace
    plot = load_script("47_plot_protocol_b_accuracy_vs_snr.py")
    monkeypatch.setattr(plot, "ROOT", tmp_path)
    verified = []
    monkeypatch.setattr(plot, "load_runner", lambda: SimpleNamespace(
        five_seed_report=lambda require_complete: verified.append(require_complete)))
    rows = [dict(method=method, display_label=label, seed=seed, snr_db=snr,
                 sample_count=700, accuracy_percent=70. + i)
            for method, label, *_ in plot.METHODS
            for i, seed in enumerate([41, 73, 107, 149, 211]) for snr in [-4., 8.]]
    individual = pd.DataFrame(rows)
    aggregate = plot.aggregate_seed_curves(individual, [41, 73, 107, 149, 211])
    monkeypatch.setattr(plot, "matched_seed_curves", lambda protocol: (individual, aggregate, []))
    plot.plot_five_seed("A")
    assert verified == [True]
    out = tmp_path / "outputs/figures/paper"
    assert (out / "paper_baseline_protocol_a_accuracy_vs_snr.pdf").read_bytes().startswith(b"%PDF")
    assert not (out / "paper_baseline_protocol_a_seed41_accuracy_vs_snr.pdf").exists()
    summary = pd.read_csv(tmp_path / "outputs/results/baseline_comparison/protocol_a_snr_summary.csv")
    assert (summary.seed_count == 5).all() and (summary.mean_percent == 72).all()


def test_iqformer_predictions_reject_duplicate_test_rows(tmp_path, monkeypatch):
    runner = load_script("16_run_hybrid_dfrft_seeds.py")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with h5py.File(tmp_path / "raw.h5", "w") as f:
        f["label"] = [0, 1]
        f["snr_db"] = [-4., 8.]
    np.savez(tmp_path / "split.npz", test=[0, 1])
    path = tmp_path / "prediction.csv"
    table = pd.DataFrame(dict(index=[0, 1], label=[0, 1], pred=[0, 0], snr_db=[-4., 8.], correct=[1, 0]))
    table.to_csv(path, index=False)
    path.with_suffix(".summary.json").write_text(json.dumps(dict(split="test", accuracy=0.5)))
    cfg = dict(eval_raw="raw.h5", eval_splits="split.npz")
    runner.validate_test_predictions(path, cfg)
    table.loc[1, "index"] = 0
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="indices mismatch"):
        runner.validate_test_predictions(path, cfg)


