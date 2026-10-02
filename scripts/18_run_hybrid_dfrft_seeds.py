#!/usr/bin/env python
"""Run resumable Protocol-B five-seed tests with the promoted hybrid DFRFT front end."""

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
    "proposed": ("drc_triplenet", "configs/model/triplenet_iq_evm.yaml", True),
    "mcnet": ("paper_mcnet", "configs/model/paper_mcnet.yaml", False),
    "starnet": ("paper_starnet", "configs/model/paper_starnet.yaml", False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    rows: list[dict[str, object]] = []
    for method in args.methods:
        model, model_config, uses_symbol = METHODS[method]
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
                    DATA["feature"],
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
                    "comp",
                    "--iq-representation",
                    "iq",
                    "--iq-normalize",
                    "zscore",
                    "--cache-iq",
                ]
                if uses_symbol:
                    command.extend(["--symbol-feature-data", DATA["symbol"]])
                if args.epochs is not None:
                    command.extend(["--epochs", str(args.epochs)])
                last = checkpoint_dir / "last.pt"
                if last.exists() and not args.overwrite:
                    command.extend(["--resume", str(last.relative_to(ROOT))])
                run(command, args.dry_run)

            if "eval" in args.stages and (args.overwrite or not summary.exists()):
                output.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    "scripts/14_evaluate_model.py",
                    "--checkpoint",
                    str(best.relative_to(ROOT)),
                    "--raw-data",
                    DATA["raw"],
                    "--feature-data",
                    DATA["feature"],
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

            if output.exists():
                rows.append(
                    {
                        "protocol": "B",
                        "front_end": "hybrid_dfrft",
                        "method": method,
                        "seed": seed,
                        "accuracy": accuracy_from_csv(output),
                    }
                )

    if "summary" not in args.stages or args.dry_run:
        return
    result_root.mkdir(parents=True, exist_ok=True)
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
        for baseline in ("mcnet", "starnet"):
            if baseline in pivot:
                delta = 100.0 * (pivot["proposed"] - pivot[baseline]).dropna().to_numpy(float)
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
