#!/usr/bin/env python
"""Run Protocol-B representation ablations with the promoted hybrid DFRFT front end."""

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
    "iq_only": "configs/model/triplenet_iq_only.yaml",
    "constellation_only": "configs/model/triplenet_evm_only.yaml",
}
DISPLAY = {
    "iq_only": "I/Q only",
    "constellation_only": "Constellation only",
    "fusion": "Fusion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[*METHODS, "fusion"],
        default=["iq_only", "constellation_only", "fusion"],
    )
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


def accuracy(path: Path) -> float:
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

    checkpoint_root = ROOT / "outputs/checkpoints/hybrid_dfrft/representation_ablation/protocol_b"
    result_root = ROOT / "outputs/results/hybrid_dfrft/representation_ablation/protocol_b"
    rows: list[dict[str, object]] = []
    for method in args.methods:
        for seed in args.seeds:
            if method == "fusion":
                checkpoint = ROOT / f"outputs/checkpoints/hybrid_dfrft/protocol_b/proposed/seed_{seed}/best.pt"
                output = ROOT / f"outputs/results/hybrid_dfrft/protocol_b/proposed/seed_{seed}_test.csv"
                summary = output.with_suffix(".summary.json")
                if not args.dry_run and (not checkpoint.exists() or not output.exists()):
                    raise SystemExit(f"Missing promoted fusion artifact: {checkpoint} or {output}")
            else:
                checkpoint_dir = checkpoint_root / method / f"seed_{seed}"
                checkpoint = checkpoint_dir / "best.pt"
                complete = checkpoint_dir / "train_result.json"
                output = result_root / method / f"seed_{seed}_test.csv"
                summary = result_root / method / f"seed_{seed}_test.summary.json"
                if "train" in args.stages and (args.overwrite or not (complete.exists() and checkpoint.exists())):
                    command = [
                        sys.executable,
                        "scripts/13_train_model.py",
                        "--raw-data",
                        DATA["raw"],
                        "--feature-data",
                        DATA["feature"],
                        "--symbol-feature-data",
                        DATA["symbol"],
                        "--splits",
                        DATA["splits"],
                        "--model",
                        "drc_triplenet",
                        "--model-config",
                        METHODS[method],
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
                    ]
                    if method == "iq_only":
                        command.append("--cache-iq")
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
                        str(checkpoint.relative_to(ROOT)),
                        "--raw-data",
                        DATA["raw"],
                        "--feature-data",
                        DATA["feature"],
                        "--symbol-feature-data",
                        DATA["symbol"],
                        "--splits",
                        DATA["splits"],
                        "--split",
                        "test",
                        "--model",
                        "drc_triplenet",
                        "--model-config",
                        METHODS[method],
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
                    run(command, args.dry_run)

            if output.exists():
                rows.append(
                    {
                        "protocol": "B",
                        "front_end": "hybrid_dfrft",
                        "method": method,
                        "display": DISPLAY[method],
                        "seed": seed,
                        "accuracy": accuracy(output),
                        "checkpoint": str(checkpoint.relative_to(ROOT)),
                        "eval_csv": str(output.relative_to(ROOT)),
                    }
                )

    if "summary" not in args.stages or args.dry_run:
        return
    expected = len(args.methods) * len(args.seeds)
    if len(rows) != expected:
        raise SystemExit(f"Found {len(rows)}/{expected} evaluations; summary not written.")
    result_root.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows).sort_values(["seed", "method"]).reset_index(drop=True)
    runs.to_csv(result_root / "seed_runs.csv", index=False)
    aggregate = (
        runs.groupby(["method", "display"], sort=False)["accuracy"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    aggregate.to_csv(result_root / "seed_summary.csv", index=False)
    pivot = runs.pivot(index="seed", columns="method", values="accuracy")
    gains = 100.0 * (
        pivot["fusion"] - pivot[["iq_only", "constellation_only"]].max(axis=1)
    )
    paired = pivot.copy()
    paired["fusion_gain_over_stronger_single_pp"] = gains
    paired.reset_index().to_csv(result_root / "paired_fusion_gain.csv", index=False)
    payload = {
        "protocol": "B",
        "front_end": "hybrid_dfrft",
        "paired_fusion_gain": {
            "count": int(gains.size),
            "mean_pp": float(gains.mean()),
            "sample_sd_pp": float(gains.std(ddof=1)),
            "positive_seeds": int(np.count_nonzero(gains.to_numpy() > 0.0)),
        },
    }
    (result_root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(aggregate.to_string(index=False))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
