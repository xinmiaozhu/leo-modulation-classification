#!/usr/bin/env python
"""Matched-seed paper baselines on final Protocol A/B compensated frames.

Protocol B uses DCFT acquisition plus coherent refinement; hybrid_dfrft is
only the historical artifact path. IQFormer retains its official STFT branch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [41, 73, 107, 149, 211]
DATA = {
    "raw": "data/processed/leo_7mods_joint_practical.h5",
    "feature": "data/features/hybrid_dfrft/protocol_b_iqcomp.h5",
    "symbol": "data/features/hybrid_dfrft/protocol_b_symbol_exact.h5",
    "splits": "data/splits/leo_7mods_joint_practical_splits.npz",
}
METHODS = {
    "cnn2": ("paper_cnn2", "configs/model/paper_cnn2.yaml", False),
    "cnn_lstm_dual": ("paper_cnn_lstm_dual", "configs/model/paper_cnn_lstm_dual.yaml", False),
    "satellite_cnn": ("paper_satellite_cnn", "configs/model/paper_satellite_cnn.yaml", False),
    "iqformer": ("paper_iqformer", "configs/model/paper_iqformer.yaml", False),
    "proposed": ("drc_dualnet", "configs/model/dualnet_iq_evm.yaml", True),
    "mcnet": ("paper_mcnet", "configs/model/paper_mcnet.yaml", False),
    "starnet": ("paper_starnet", "configs/model/paper_starnet.yaml", False),
}

def protocol_inputs(protocol):
    if protocol == "B":
        return {**{f"train_{k}": v for k, v in DATA.items()},
                **{f"eval_{k}": v for k, v in DATA.items()}}
    spec = importlib.util.spec_from_file_location("final_protocols", ROOT / "scripts/17_run_representation_ablation_seeds.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PROTOCOLS[protocol]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocols", nargs="+", choices=["A", "B"], default=["B"])
    parser.add_argument("--report", action="store_true", help="Generate the formal five-seed figures and comparison tables (alias for --five-seed-report).")
    parser.add_argument("--five-seed-report", action="store_true",
                        help="Publish matched five-seed A/B tables and curves only after all seven methods are complete.")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["train", "eval", "summary"],
        default=["train", "eval", "summary"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-config", default="configs/train/train_long.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique.")
    return args


def run(command: list[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if not dry_run:
        started = time.time()
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            # Windows can fail during native-library teardown after training
            # has saved every artifact. Never accept an unfinished/stale run.
            if ((error.returncode & 0xFFFFFFFF) != 0xC0000409
                    or Path(command[1]).name != "12_train_model.py"
                    or not completed_training_artifacts(command, started)):
                raise
            print("WARNING: training process exit 0xC0000409; fresh completion, "
                  "history and checkpoints verified. Continuing with strict test evaluation.", flush=True)


def completed_training_artifacts(command, started):
    import torch
    directory = ROOT / command[command.index("--output-dir") + 1]
    paths = [directory / name for name in ("train_result.json", "best.pt", "last.pt", "history.csv")]
    if not all(p.exists() for p in paths) or paths[0].stat().st_mtime < started:
        return False
    try:
        result = json.loads(paths[0].read_text())
        history = pd.read_csv(paths[3])
        if history.empty or not np.isfinite(history.val_acc).all():
            return False
        best = torch.load(paths[1], map_location="cpu", weights_only=False)
        last = torch.load(paths[2], map_location="cpu", weights_only=False)
        seed = int(command[command.index("--seed") + 1])
        model = command[command.index("--model") + 1]
        for checkpoint in (best, last):
            if checkpoint["config"]["seed"] != seed or checkpoint["config"]["model_name"] != model:
                return False
            if not all(torch.isfinite(v).all() for v in checkpoint["model_state_dict"].values()):
                return False
        return (int(last["epoch"]) == int(history.epoch.max())
                and np.isclose(result["best_acc"], history.val_acc.max())
                and np.isclose(best["metrics"]["val_acc"], result["best_acc"]))
    except (OSError, ValueError, KeyError, RuntimeError):
        return False


def accuracy_from_csv(path: Path) -> float:
    table = pd.read_csv(path)
    if "correct" in table:
        return float(table["correct"].astype(float).mean())
    prediction = "pred" if "pred" in table else "predicted"
    return float((table["label"].to_numpy() == table[prediction].to_numpy()).mean())


def validate_test_predictions(path, cfg):
    """Reject stale, mismatched, duplicate, or incorrectly summarized test rows."""
    table = pd.read_csv(path).sort_values("index").reset_index(drop=True)
    with np.load(ROOT / cfg["eval_splits"]) as splits:
        indices = np.sort(splits["test"].astype(int))
    if not np.array_equal(table["index"].to_numpy(), indices):
        raise ValueError(f"Test indices mismatch: {path}")
    with h5py.File(ROOT / cfg["eval_raw"], "r") as raw:
        if not np.array_equal(table.label.to_numpy(), raw["label"][indices]):
            raise ValueError(f"Test labels mismatch: {path}")
        np.testing.assert_allclose(table.snr_db, raw["snr_db"][indices], rtol=0, atol=1e-6)
    correct = (table.label.to_numpy() == table.pred.to_numpy()).astype(int)
    if not np.array_equal(table.correct.to_numpy(), correct):
        raise ValueError(f"Incorrect prediction flags: {path}")
    metadata = json.loads(path.with_suffix(".summary.json").read_text())
    if metadata["split"] != "test" or not np.isclose(metadata["accuracy"], correct.mean()):
        raise ValueError(f"Incorrect test metadata: {path}")


def run_protocol(args, protocol):
    cfg = protocol_inputs(protocol)
    missing = [path for path in cfg.values() if not (ROOT / path).exists()]
    if missing:
        raise SystemExit("Missing final protocol inputs: " + str(missing))
    family = "exact_mixture" if protocol == "A" else "hybrid_dfrft"
    checkpoint_root = ROOT / "outputs/checkpoints" / family / f"protocol_{protocol.lower()}"
    result_root = ROOT / "outputs/results" / family / f"protocol_{protocol.lower()}"
    for method in args.methods:
        model, model_config, uses_symbol = METHODS[method]
        for seed in args.seeds:
            if getattr(args, "five_seed_report", False) and not args.dry_run:
                write_campaign_status(args, "running", protocol=protocol, method=method, seed=seed)
            checkpoint_dir = checkpoint_root / method / f"seed_{seed}"
            best = checkpoint_dir / "best.pt"
            completed = checkpoint_dir / "train_result.json"
            output = result_root / method / f"seed_{seed}_test.csv"
            summary = result_root / method / f"seed_{seed}_test.summary.json"

            if "train" in args.stages and (args.overwrite or not (completed.exists() and best.exists())):
                command = [
                    sys.executable,
                    "scripts/12_train_model.py",
                    "--raw-data",
                    cfg["train_raw"],
                    "--feature-data",
                    cfg["train_feature"],
                    "--splits",
                    cfg["train_splits"],
                    "--model",
                    model,
                    "--model-config",
                    model_config,
                    "--train-config",
                    args.train_config,
                    "--output-dir",
                    str(checkpoint_dir.relative_to(ROOT)),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    "--num-workers",
                    "0",
                    "--iq-source",
                    "comp",
                    "--iq-representation",
                    "iq",
                    "--iq-normalize",
                    "power" if method == "cnn_lstm_dual" else "zscore",
                ]
                command.extend(["--cache-iq", "--no-hoc-standardize"])
                if uses_symbol:
                    command.extend(["--symbol-feature-data", cfg["train_symbol"]])
                if args.epochs is not None:
                    command.extend(["--epochs", str(args.epochs)])
                last = checkpoint_dir / "last.pt"
                if last.exists() and not args.overwrite:
                    command.extend(["--resume", str(last.relative_to(ROOT))])
                run(command, args.dry_run)

            if "eval" in args.stages and (args.overwrite or not (summary.exists() and output.exists())):
                if not args.dry_run:
                    output.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    "scripts/13_evaluate_model.py",
                    "--checkpoint",
                    str(best.relative_to(ROOT)),
                    "--raw-data",
                    cfg["eval_raw"],
                    "--feature-data",
                    cfg["eval_feature"],
                    "--splits",
                    cfg["eval_splits"],
                    "--split",
                    "test",
                    "--model",
                    model,
                    "--output",
                    str(output.relative_to(ROOT)),
                    "--summary-output",
                    str(summary.relative_to(ROOT)),
                    "--batch-size",
                    "256",
                    "--num-workers",
                    "0",
                    "--device",
                    args.device,
                    "--strict",
                    "--load-into-memory",
                ]
                if uses_symbol:
                    command.extend(["--symbol-feature-data", cfg["eval_symbol"]])
                run(command, args.dry_run)

            if output.exists() and not args.dry_run:
                validate_test_predictions(output, cfg)
                if getattr(args, "five_seed_report", False):
                    five_seed_report(require_complete=False)

    if "summary" not in args.stages or args.dry_run:
        return
    result_root.mkdir(parents=True, exist_ok=True)
    # Include earlier completed methods/seeds when running only the missing baselines.
    rows: list[dict[str, object]] = []
    for method in METHODS:
        for output in sorted((result_root / method).glob("seed_*_test.csv")):
            if not output.with_suffix(".summary.json").exists():
                continue
            seed = int(output.stem.split("_")[1])
            rows.append({"protocol": protocol, "front_end": "coherent" if protocol == "A" else "dcft_refined",
                         "method": method, "seed": seed,
                         "accuracy": accuracy_from_csv(output)})
    # Legacy A baseline results live outside the final-protocol directory.
    if protocol == "A":
        for method, legacy in {
            "cnn2": "paper_baselines_independent_test/c2_cnn2_test.csv",
            "cnn_lstm_dual": "paper_baselines_independent_test/c4_cnn_lstm_dual_test.csv",
            "satellite_cnn": "paper_baselines_independent_test/c5_satellite_cnn_test.csv",
        }.items():
            path = ROOT / "outputs/results" / legacy
            if path.exists() and not any(r["method"] == method and r["seed"] == 41 for r in rows):
                rows.append(dict(protocol=protocol, front_end="coherent", method=method,
                                 seed=41, accuracy=accuracy_from_csv(path)))
        for path in (ROOT / "outputs/results/starnet_seeds/protocol_a").glob("seed_*_test.csv"):
            seed = int(path.stem.split("_")[1])
            if not any(r["method"] == "starnet" and r["seed"] == seed for r in rows):
                rows.append(dict(protocol=protocol, front_end="coherent", method="starnet",
                                 seed=seed, accuracy=accuracy_from_csv(path)))
    runs = pd.DataFrame(rows)
    runs.to_csv(result_root / "seed_runs.csv", index=False)
    if runs.empty:
        return
    aggregate = (
        runs.groupby("method", sort=False)["accuracy"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    aggregate.to_csv(result_root / "seed_summary.csv", index=False)
    pivot = runs.pivot(index="seed", columns="method", values="accuracy")
    paired: dict[str, object] = {}
    if "proposed" in pivot:
        for baseline in (method for method in METHODS if method != "proposed"):
            if baseline in pivot:
                delta = 100.0 * (pivot["proposed"] - pivot[baseline]).dropna().to_numpy(float)
                if not delta.size:
                    continue
                paired[f"proposed_minus_{baseline}"] = {
                    "count": int(delta.size),
                    "mean_delta_pp": float(delta.mean()),
                    "std_delta_pp": float(delta.std(ddof=1)) if delta.size > 1 else 0.0,
                    "positive_seeds": int(np.count_nonzero(delta > 0.0)),
                }
    payload = {"protocol": protocol, "front_end": "coherent" if protocol == "A" else "dcft_refined", "paired": paired}
    (result_root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(aggregate.to_string(index=False))
    print(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    args.five_seed_report = args.five_seed_report or args.report
    tracked = args.five_seed_report and not args.dry_run
    if tracked:
        write_campaign_status(args, "running")
    try:
        # Interleave protocols with the same prespecified seed order.
        for seed in args.seeds:
            for protocol in args.protocols:
                current = argparse.Namespace(**vars(args))
                current.seeds = [seed]
                if tracked:
                    write_campaign_status(args, "running", protocol=protocol, seed=seed)
                run_protocol(current, protocol)
                if tracked:
                    five_seed_report(require_complete=False)
        if (args.report or args.five_seed_report) and not args.dry_run:
            comparison_report()
        if tracked:
            five_seed_report(require_complete=True)
            for protocol in ("A", "B"):
                run([sys.executable, "scripts/47_plot_protocol_b_accuracy_vs_snr.py",
                     "--protocol", protocol, "--five-seed"], False)
            write_campaign_status(args, "complete")
    except BaseException as error:
        if tracked:
            write_campaign_status(args, "failed", error=str(error))
        raise


def write_campaign_status(args, status, **details):
    root = ROOT / "outputs/results/baseline_comparison"
    root.mkdir(parents=True, exist_ok=True)
    payload = dict(status=status, updated_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   pid=os.getpid(), methods=args.methods, protocols=args.protocols,
                   seeds=args.seeds, **details)
    (root / "run_status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prediction_path(protocol, method, seed):
    family = "exact_mixture" if protocol == "A" else "hybrid_dfrft"
    result = ROOT / "outputs/results" / family / f"protocol_{protocol.lower()}" / method / f"seed_{seed}_test.csv"
    if not result.exists() and protocol == "A" and method == "starnet":
        result = ROOT / f"outputs/results/starnet_seeds/protocol_a/seed_{seed}_test.csv"
    # The three old Protocol-A seed-41 baselines lack checkpoints and are rerun.
    return result


def five_seed_report(require_complete=True):
    """Validate every test frame; never publish partial means as final five-seed results."""
    output = ROOT / "outputs/results/baseline_comparison"
    output.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for protocol in ("A", "B"):
        cfg = protocol_inputs(protocol)
        for method in METHODS:
            for seed in SEEDS:
                path = prediction_path(protocol, method, seed)
                if not path.exists() or not path.with_suffix(".summary.json").exists():
                    missing.append(dict(protocol=protocol, method=method, seed=seed))
                    continue
                validate_test_predictions(path, cfg)
                rows.append(dict(protocol=protocol, method=method, seed=seed,
                                 accuracy_percent=100 * accuracy_from_csv(path),
                                 eval_csv=path.relative_to(ROOT).as_posix()))
    payload = dict(complete=not missing, completed=len(rows), expected=70, seeds=SEEDS, missing=missing)
    (output / "completeness.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    runs = pd.DataFrame(rows)
    runs.to_csv(output / "progress_runs.csv", index=False)
    if missing:
        if require_complete:
            raise ValueError(f"Five-seed comparison incomplete: {len(rows)}/70; see {output / 'completeness.json'}")
        return
    runs.to_csv(output / "seed_runs.csv", index=False)
    summary = runs.groupby(["protocol", "method"]).accuracy_percent.agg(
        count="count", mean_percent="mean", sample_sd_percent="std").reset_index()
    summary.to_csv(output / "seed_summary.csv", index=False)
    paired = []
    for protocol in ("A", "B"):
        pivot = runs[runs.protocol == protocol].pivot(index="seed", columns="method", values="accuracy_percent")
        for method in METHODS:
            if method == "proposed":
                continue
            delta = pivot.proposed - pivot[method]
            paired.append(dict(protocol=protocol, baseline=method, count=len(delta),
                               mean_gain_pp=delta.mean(), sample_sd_gain_pp=delta.std(ddof=1),
                               positive_seeds=int((delta > 0).sum())))
    pd.DataFrame(paired).to_csv(output / "paired_gains.csv", index=False)
    # Accuracy comes exclusively from the matched-seed results. Structural costs
    # and hardware timings have a separate provenance; timing is not seed SD.
    cost_path = ROOT / "outputs/results/baseline_comparison/protocol_a_accuracy_complexity.csv"
    if cost_path.exists():
        cost = pd.read_csv(cost_path)
        mapping = dict(C1="cnn2", C2="mcnet", C3="cnn_lstm_dual", C4="satellite_cnn",
                       C5="iqformer", C6="starnet", Proposed="proposed")
        cost["baseline"] = cost.method_id.map(mapping)
        cost = cost.drop(columns=["accuracy", "accuracy_percent"], errors="ignore")
        cost = cost.merge(summary[summary.protocol == "A"], left_on="baseline", right_on="method",
                          how="left", validate="one_to_one", suffixes=("_implementation", ""))
        if cost.mean_percent.isna().any():
            raise ValueError("Complexity table contains an unmatched model")
        cost["accuracy_scope"] = "five matched seeds; mean and sample SD"
        cost["cost_scope"] = "existing Protocol-A seed-41 profiling; E2E is a component-sum estimate"
        cost.to_csv(output / "protocol_a_accuracy_complexity.csv", index=False)
        paper_table = cost.rename(columns={"count": "training_seed_count",
                                          "mean_percent": "accuracy_mean_percent",
                                          "sample_sd_percent": "accuracy_sample_sd_percent"})
        paper_table["accuracy_seeds"] = ",".join(map(str, SEEDS))
        paper_table["accuracy_display"] = paper_table.apply(
            lambda r: f"{r.accuracy_mean_percent:.2f} ± {r.accuracy_sample_sd_percent:.2f}", axis=1)
        names = dict(cnn2="CNN2", mcnet="MCNet", cnn_lstm_dual="CNN-LSTM", satellite_cnn="IQCNet",
                     iqformer="IQFormer", starnet="STARNet", proposed="Proposed")
        paper_table["display_label"] = paper_table.method.map(names)
        metric_columns = ["parameter_count", "estimated_mmacs", "network_cpu_median_ms", "end_to_end_cpu_median_ms"]
        if set(paper_table.method) != set(METHODS) or len(paper_table) != len(METHODS):
            raise ValueError("Table IV requires all seven models")
        if not np.isfinite(paper_table[metric_columns].to_numpy(float)).all():
            raise ValueError("Table IV contains missing computational-cost measurements")
        columns = ["method_id", "method", "display_label", "training_seed_count", "accuracy_seeds",
                   "accuracy_mean_percent", "accuracy_sample_sd_percent", "accuracy_display",
                   "parameter_count", "estimated_mmacs", "network_cpu_median_ms", "end_to_end_cpu_median_ms"]
        paper_table = paper_table.reindex(columns=columns)
        paper_table.to_csv(output / "table_iv.csv", index=False)
        (output / "table_iv_metadata.json").write_text(json.dumps(dict(
            title="Protocol-A five-seed accuracy and computational cost",
            accuracy_seeds=SEEDS, accuracy_statistic="mean ± sample SD (ddof=1)",
            structure_statistic="Parameters and neural MACs are properties of the fixed architecture and input size.",
            latency_statistic="Existing single-thread CPU benchmark medians, not training-seed means or SDs.",
            latency_source=cost_path.relative_to(ROOT).as_posix(),
            latency_limitation="Profiled with seed-41 models in separate sessions; not repeated across the five trained checkpoints.",
            e2e_statistic="Component-sum latency estimate, not a directly timed whole-pipeline execution.",
            caption="Protocol-A five-seed accuracy and computational cost. Accuracy is mean ± sample standard deviation."
        ), indent=2) + "\n", encoding="utf-8")
    labels = dict(cnn2="CNN2", mcnet="MCNet", cnn_lstm_dual="CNN-LSTM", satellite_cnn="IQCNet",
                  iqformer="IQFormer", starnet="STARNet", proposed="Proposed")
    lines = ["# Matched five-seed baseline comparison", "",
             "All accuracies are percentages; SD is the sample standard deviation (ddof=1).",
             "Protocol B uses DCFT acquisition plus coherent refinement.", "",
             "| Method | Protocol A | Protocol B |", "|---|---:|---:|"]
    for method, label in labels.items():
        cells = []
        for protocol in ("A", "B"):
            row = summary[(summary.protocol == protocol) & (summary.method == method)].iloc[0]
            cells.append(f"{row.mean_percent:.2f} ± {row.sample_sd_percent:.2f}")
        lines.append(f"| {label} | {' | '.join(cells)} |")
    lines.extend(["", "Curves show equal-weight means across the five seeds; shading shows sample SD, not confidence intervals.",
                  "The complexity table updates accuracy only. Existing seed-41 CPU timings were measured in separate sessions; E2E is a component-sum estimate.", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def comparison_report():
    frames = []
    for protocol, family in [("A", "exact_mixture"), ("B", "hybrid_dfrft")]:
        root = ROOT / "outputs/results" / family / f"protocol_{protocol.lower()}"
        path = root / "seed_runs.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame = frame[frame.method.isin(METHODS)]
            frames.append(frame)
    if not frames:
        return
    runs = pd.concat(frames, ignore_index=True)
    output = ROOT / "outputs/results/iqformer_comparison"
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "seed_runs.csv", index=False)
    summary = runs.groupby(["protocol", "method"]).accuracy.agg(["count", "mean", "std"]).reset_index()
    summary.to_csv(output / "seed_summary.csv", index=False)
    text = ["# IQFormer comparison", "", "Accuracy is a proportion in seed_runs.csv and seed_summary.csv.",
            "Only count=5 rows are five-seed results. Standard deviations use ddof=1.",
            "Protocol B: DCFT acquisition plus local coherent refinement.", ""]
    (output / "README.md").write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    main()
