#!/usr/bin/env python
"""Train and test all seven Protocol-B methods with the hybrid DFRFT front end."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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
    "nasa_hoc_nn": ("paper_nasa_hoc_nn", "configs/model/paper_nasa_hoc_nn.yaml", False),
    "proposed": ("drc_dualnet", "configs/model/dualnet_iq_evm.yaml", True),
    "mcnet": ("paper_mcnet", "configs/model/paper_mcnet.yaml", False),
    "starnet": ("paper_starnet", "configs/model/paper_starnet.yaml", False),
}
PAPER_FEATURE = "data/features/hybrid_dfrft/protocol_b_paper_baselines.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["features", "train", "eval", "summary"],
        default=["features", "train", "eval", "summary"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-config", default="configs/train/train_long.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def accuracy_from_csv(path: Path) -> float:
    table = pd.read_csv(path)
    if "correct" in table:
        return float(table["correct"].astype(float).mean())
    prediction = "pred" if "pred" in table else "predicted"
    return float((table["label"].to_numpy() == table[prediction].to_numpy()).mean())


def main() -> None:
    args = parse_args()
    missing = [path for path in DATA.values() if not (ROOT / path).exists()]
    if missing:
        raise SystemExit("Missing promoted-front-end inputs:\n" + "\n".join(missing))

    checkpoint_root = ROOT / "outputs/checkpoints/hybrid_dfrft/protocol_b"
    result_root = ROOT / "outputs/results/hybrid_dfrft/protocol_b"
    if "nasa_hoc_nn" in args.methods:
        if "features" in args.stages:
            command = [
                sys.executable, "scripts/12_precompute_paper_baseline_features.py",
                "--raw-data", DATA["raw"], "--comp-feature-data", DATA["feature"],
                "--symbol-feature-data", DATA["symbol"], "--output", PAPER_FEATURE,
            ]
            if args.overwrite:
                command.append("--overwrite")
            run(command, args.dry_run)
        elif {"train", "eval"}.intersection(args.stages) and not args.dry_run:
            import h5py
            with h5py.File(ROOT / PAPER_FEATURE, "r") as features:
                if not bool(features.attrs.get("complete", False)):
                    raise ValueError("NASA features are incomplete; run the features stage first.")
    for method in args.methods:
        model, model_config, uses_symbol = METHODS[method]
        is_nasa = method == "nasa_hoc_nn"
        feature_data = PAPER_FEATURE if is_nasa else DATA["feature"]
        for seed in args.seeds:
            checkpoint_dir = checkpoint_root / method / f"seed_{seed}"
            best = checkpoint_dir / "best.pt"
            completed = checkpoint_dir / "train_result.json"
            output = result_root / method / f"seed_{seed}_test.csv"
            summary = result_root / method / f"seed_{seed}_test.summary.json"

            if "train" in args.stages and (args.overwrite or not (completed.exists() and best.exists())):
                command = [
                    sys.executable,
                    "scripts/13_train_model.py",
                    "--raw-data",
                    DATA["raw"],
                    "--feature-data",
                    feature_data,
                    "--splits",
                    DATA["splits"],
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
                    "raw" if is_nasa else "comp",
                    "--iq-representation",
                    "iq",
                    "--iq-normalize",
                    "power" if method == "cnn_lstm_dual" else "zscore",
                ]
                if is_nasa:
                    command.extend(["--hoc-transform", "none"])
                else:
                    command.append("--cache-iq")
                    if method in {"cnn2", "cnn_lstm_dual", "satellite_cnn"}:
                        command.append("--no-hoc-standardize")
                if uses_symbol:
                    command.extend(["--symbol-feature-data", DATA["symbol"]])
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
                    "scripts/14_evaluate_model.py",
                    "--checkpoint",
                    str(best.relative_to(ROOT)),
                    "--raw-data",
                    DATA["raw"],
                    "--feature-data",
                    feature_data,
                    "--splits",
                    DATA["splits"],
                    "--split",
                    "test",
                    "--model",
                    model,
                    "--model-config",
                    model_config,
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
                    command.extend(["--symbol-feature-data", DATA["symbol"]])
                run(command, args.dry_run)

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
            rows.append({"protocol": "B", "front_end": "hybrid_dfrft",
                         "method": method, "seed": seed,
                         "accuracy": accuracy_from_csv(output)})
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
    payload = {"protocol": "B", "front_end": "hybrid_dfrft", "paired": paired}
    (result_root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(aggregate.to_string(index=False))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
