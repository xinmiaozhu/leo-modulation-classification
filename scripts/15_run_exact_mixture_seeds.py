#!/usr/bin/env python
"""Run resumable five-seed Proposed/MCNet tests after exact-mixture deployment."""

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
PROTOCOLS = {
    "A": {
        "train_raw": "data/processed/leo_7mods_snr_balanced_trainval_pilot.h5",
        "train_feature": "data/features/leo_7mods_snr_balanced_trainval_hoc_iqcomp_coherent.h5",
        "train_symbol": "data/features/exact_mixture/protocol_a_trainval_symbol_exact.h5",
        "train_splits": "data/splits/leo_7mods_snr_balanced_trainval_pilot_splits.npz",
        "eval_raw": "data/processed/leo_7mods_snr_balanced_independent_test_pilot.h5",
        "eval_feature": "data/features/leo_7mods_snr_balanced_independent_test_hoc_iqcomp_coherent.h5",
        "eval_symbol": "data/features/exact_mixture/protocol_a_test_symbol_exact.h5",
        "eval_splits": "data/splits/leo_7mods_snr_balanced_independent_test_pilot_splits.npz",
    },
    "B": {
        "train_raw": "data/processed/leo_7mods_joint_practical.h5",
        "train_feature": "data/features/leo_7mods_joint_practical_iqcomp.h5",
        "train_symbol": "data/features/exact_mixture/protocol_b_symbol_exact.h5",
        "train_splits": "data/splits/leo_7mods_joint_practical_splits.npz",
        "eval_raw": "data/processed/leo_7mods_joint_practical.h5",
        "eval_feature": "data/features/leo_7mods_joint_practical_iqcomp.h5",
        "eval_symbol": "data/features/exact_mixture/protocol_b_symbol_exact.h5",
        "eval_splits": "data/splits/leo_7mods_joint_practical_splits.npz",
    },
}
METHODS = {
    "proposed": ("drc_dualnet", "configs/model/dualnet_iq_evm.yaml", True),
    "mcnet": ("paper_mcnet", "configs/model/paper_mcnet.yaml", False),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocols", nargs="+", choices=["A", "B"], default=["A", "B"])
    p.add_argument("--methods", nargs="+", choices=list(METHODS), default=["proposed", "mcnet"])
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--stages", nargs="+", choices=["train", "eval", "summary"], default=["train", "eval", "summary"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def accuracy_from_csv(path: Path) -> float:
    table = pd.read_csv(path)
    if "correct" in table:
        return float(table["correct"].astype(float).mean())
    pred = "pred" if "pred" in table else "predicted"
    return float((table["label"].to_numpy() == table[pred].to_numpy()).mean())


def main() -> None:
    args = parse_args()
    for protocol in args.protocols:
        cfg = PROTOCOLS[protocol]
        required = [cfg[k] for k in cfg]
        missing = [path for path in required if not (ROOT / path).exists()]
        if missing:
            raise SystemExit(f"Protocol {protocol} missing inputs:\n" + "\n".join(missing))
        result_root = ROOT / "outputs" / "results" / "exact_mixture" / f"protocol_{protocol.lower()}"
        checkpoint_root = ROOT / "outputs" / "checkpoints" / "exact_mixture" / f"protocol_{protocol.lower()}"
        rows: list[dict[str, object]] = []
        for method in args.methods:
            model, model_config, uses_symbol = METHODS[method]
            for seed in args.seeds:
                checkpoint_dir = checkpoint_root / method / f"seed_{seed}"
                best = checkpoint_dir / "best.pt"
                complete = checkpoint_dir / "train_result.json"
                output = result_root / method / f"seed_{seed}_test.csv"
                summary = result_root / method / f"seed_{seed}_test.summary.json"
                common_train = [
                    sys.executable, "scripts/13_train_model.py",
                    "--raw-data", cfg["train_raw"], "--feature-data", cfg["train_feature"],
                    "--splits", cfg["train_splits"], "--model", model,
                    "--model-config", model_config, "--train-config", "configs/train/train_long.yaml",
                    "--output-dir", str(checkpoint_dir.relative_to(ROOT)), "--seed", str(seed),
                    "--device", args.device, "--num-workers", "0", "--iq-source", "comp",
                    "--iq-representation", "iq", "--iq-normalize", "zscore", "--cache-iq",
                ]
                if uses_symbol:
                    common_train.extend(["--symbol-feature-data", cfg["train_symbol"]])
                if "train" in args.stages and (args.overwrite or not complete.exists()):
                    last = checkpoint_dir / "last.pt"
                    if last.exists() and not args.overwrite:
                        common_train.extend(["--resume", str(last.relative_to(ROOT))])
                    run(common_train)
                if "eval" in args.stages and (args.overwrite or not summary.exists()):
                    output.parent.mkdir(parents=True, exist_ok=True)
                    command = [
                        sys.executable, "scripts/14_evaluate_model.py", "--checkpoint", str(best.relative_to(ROOT)),
                        "--raw-data", cfg["eval_raw"], "--feature-data", cfg["eval_feature"],
                        "--splits", cfg["eval_splits"], "--split", "test", "--model", model,
                        "--model-config", model_config, "--output", str(output.relative_to(ROOT)),
                        "--summary-output", str(summary.relative_to(ROOT)), "--batch-size", "256",
                        "--num-workers", "0", "--device", args.device, "--strict", "--load-into-memory",
                    ]
                    if uses_symbol:
                        command.extend(["--symbol-feature-data", cfg["eval_symbol"]])
                    run(command)
                if output.exists():
                    rows.append({"protocol": protocol, "method": method, "seed": seed, "accuracy": accuracy_from_csv(output)})
        if "summary" in args.stages:
            result_root.mkdir(parents=True, exist_ok=True)
            runs = pd.DataFrame(rows)
            runs.to_csv(result_root / "seed_runs.csv", index=False)
            if not runs.empty:
                summary_table = runs.groupby("method", sort=False)["accuracy"].agg(["count", "mean", "std", "min", "max"]).reset_index()
                summary_table.to_csv(result_root / "seed_summary.csv", index=False)
                pivot = runs.pivot(index="seed", columns="method", values="accuracy")
                paired = {}
                if {"proposed", "mcnet"}.issubset(pivot.columns):
                    delta = 100.0 * (pivot["proposed"] - pivot["mcnet"]).dropna().to_numpy(dtype=float)
                    paired = {"count": int(delta.size), "mean_delta_pp": float(np.mean(delta)), "std_delta_pp": float(np.std(delta, ddof=1)) if delta.size > 1 else 0.0}
                (result_root / "summary.json").write_text(json.dumps({"protocol": protocol, "paired_proposed_minus_mcnet": paired}, indent=2) + "\n", encoding="utf-8")
                print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
