#!/usr/bin/env python
"""Run resumable Proposed seeds for matched-pass and elevation-OOD tests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = "data/processed/pass_holdout/leo_orbital_pass_snr_balanced.h5"
FEATURE = "data/features/pass_holdout/leo_orbital_pass_snr_balanced_iqcomp.h5"
SYMBOL = "data/features/pass_holdout/leo_orbital_pass_snr_balanced_symbol_exact.h5"
TRAIN_SPLITS = "data/splits/pass_holdout/leo_orbital_pass_snr_balanced_splits.npz"
EVAL_SPLITS = {
    "matched_pass": TRAIN_SPLITS,
    "elevation_ood": "data/splits/pass_holdout/leo_orbital_elevation_ood_snr_balanced_splits.npz",
}
SEEDS = (41, 73, 107, 149, 211)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--stages", nargs="+", choices=["train", "eval", "summary"], default=["train", "eval", "summary"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def accuracy(path: Path) -> float:
    table = pd.read_csv(path)
    if "correct" in table:
        return float(table["correct"].astype(float).mean())
    pred = "pred" if "pred" in table else "predicted"
    return float((table["label"].to_numpy() == table[pred].to_numpy()).mean())


def main() -> None:
    args = parse_args()
    required = [RAW, FEATURE, SYMBOL, TRAIN_SPLITS, *EVAL_SPLITS.values()]
    missing = [path for path in required if not (ROOT / path).exists()]
    if missing:
        raise SystemExit("Missing inputs:\n" + "\n".join(missing))
    ckpt_root = ROOT / "outputs/checkpoints/pass_holdout_snr_balanced/proposed"
    result_root = ROOT / "outputs/results/pass_holdout_snr_balanced/proposed"
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        ckpt_dir = ckpt_root / f"seed_{seed}"
        best = ckpt_dir / "best.pt"
        complete = ckpt_dir / "train_result.json"
        if "train" in args.stages and (args.overwrite or not complete.exists()):
            command = [
                sys.executable, "scripts/13_train_model.py",
                "--raw-data", RAW, "--feature-data", FEATURE,
                "--symbol-feature-data", SYMBOL, "--splits", TRAIN_SPLITS,
                "--model", "drc_triplenet", "--model-config", "configs/model/triplenet_iq_evm.yaml",
                "--train-config", "configs/train/train_long.yaml", "--output-dir", str(ckpt_dir.relative_to(ROOT)),
                "--seed", str(seed), "--device", args.device, "--num-workers", "0",
                "--iq-source", "comp", "--iq-representation", "iq", "--iq-normalize", "zscore", "--cache-iq",
            ]
            last = ckpt_dir / "last.pt"
            if last.exists() and not args.overwrite:
                command.extend(["--resume", str(last.relative_to(ROOT))])
            run(command)
        for evaluation, splits in EVAL_SPLITS.items():
            output = result_root / evaluation / f"seed_{seed}_test.csv"
            summary = output.with_suffix(".summary.json")
            if "eval" in args.stages and (args.overwrite or not summary.exists()):
                output.parent.mkdir(parents=True, exist_ok=True)
                run([
                    sys.executable, "scripts/14_evaluate_model.py", "--checkpoint", str(best.relative_to(ROOT)),
                    "--raw-data", RAW, "--feature-data", FEATURE, "--symbol-feature-data", SYMBOL,
                    "--splits", splits, "--split", "test", "--model", "drc_triplenet",
                    "--model-config", "configs/model/triplenet_iq_evm.yaml", "--output", str(output.relative_to(ROOT)),
                    "--summary-output", str(summary.relative_to(ROOT)), "--batch-size", "256", "--num-workers", "0",
                    "--device", args.device, "--strict", "--load-into-memory",
                ])
            if output.exists():
                rows.append({"seed": seed, "evaluation": evaluation, "accuracy": accuracy(output)})
    if "summary" in args.stages:
        result_root.mkdir(parents=True, exist_ok=True)
        runs = pd.DataFrame(rows)
        runs.to_csv(result_root / "seed_runs.csv", index=False)
        if not runs.empty:
            summary = runs.groupby("evaluation")["accuracy"].agg(["count", "mean", "std", "min", "max"]).reset_index()
            summary.to_csv(result_root / "seed_summary.csv", index=False)
            pivot = runs.pivot(index="seed", columns="evaluation", values="accuracy")
            paired = {}
            if set(EVAL_SPLITS).issubset(pivot.columns):
                delta = 100 * (pivot["elevation_ood"] - pivot["matched_pass"]).dropna().to_numpy(float)
                paired = {"count": int(len(delta)), "mean_delta_pp": float(np.mean(delta)), "std_delta_pp": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0}
            (result_root / "summary.json").write_text(json.dumps({"paired_ood_minus_matched": paired}, indent=2) + "\n", encoding="utf-8")
            print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
