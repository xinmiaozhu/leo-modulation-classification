#!/usr/bin/env python
"""Run resumable five-seed STARNet experiments for Protocols A and B."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [41, 73, 107, 149, 211]
PROTOCOLS = {
    "A": {
        "train_raw": "data/processed/leo_7mods_snr_balanced_trainval_pilot.h5",
        "train_feature": "data/features/leo_7mods_snr_balanced_trainval_hoc_iqcomp_coherent.h5",
        "train_splits": "data/splits/leo_7mods_snr_balanced_trainval_pilot_splits.npz",
        "eval_raw": "data/processed/leo_7mods_snr_balanced_independent_test_pilot.h5",
        "eval_feature": "data/features/leo_7mods_snr_balanced_independent_test_hoc_iqcomp_coherent.h5",
        "eval_splits": "data/splits/leo_7mods_snr_balanced_independent_test_pilot_splits.npz",
    },
    "B": {
        "train_raw": "data/processed/leo_7mods_joint_practical.h5",
        "train_feature": "data/features/leo_7mods_joint_practical_iqcomp.h5",
        "train_splits": "data/splits/leo_7mods_joint_practical_splits.npz",
        "eval_raw": "data/processed/leo_7mods_joint_practical.h5",
        "eval_feature": "data/features/leo_7mods_joint_practical_iqcomp.h5",
        "eval_splits": "data/splits/leo_7mods_joint_practical_splits.npz",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocols", nargs="+", choices=["A", "B"], default=["A", "B"])
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--stages", nargs="+", choices=["train", "eval", "summary"], default=["train", "eval", "summary"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--train-config", default="configs/train/train_long.yaml")
    p.add_argument("--epochs", type=int)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run(command: list[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    for protocol in args.protocols:
        spec = PROTOCOLS[protocol]
        checkpoint_root = ROOT / "outputs" / "checkpoints" / "starnet_seeds" / f"protocol_{protocol.lower()}"
        result_root = ROOT / "outputs" / "results" / "starnet_seeds" / f"protocol_{protocol.lower()}"
        result_root.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            checkpoint_dir = checkpoint_root / f"seed_{seed}"
            best = checkpoint_dir / "best.pt"
            completed = checkpoint_dir / "train_result.json"
            if "train" in args.stages and (args.overwrite or not (completed.exists() and best.exists())):
                command = [
                    sys.executable, "scripts/13_train_model.py",
                    "--raw-data", spec["train_raw"], "--feature-data", spec["train_feature"],
                    "--splits", spec["train_splits"], "--model", "paper_starnet",
                    "--model-config", "configs/model/paper_starnet.yaml",
                    "--train-config", args.train_config, "--output-dir", str(checkpoint_dir.relative_to(ROOT)),
                    "--seed", str(seed), "--device", args.device, "--num-workers", "0",
                    "--iq-source", "comp", "--iq-representation", "iq", "--iq-normalize", "zscore",
                ]
                if args.epochs is not None:
                    command.extend(["--epochs", str(args.epochs)])
                last = checkpoint_dir / "last.pt"
                if last.exists() and not args.overwrite:
                    command.extend(["--resume", str(last.relative_to(ROOT))])
                run(command, args.dry_run)
            output = result_root / f"seed_{seed}_test.csv"
            summary = result_root / f"seed_{seed}_test.summary.json"
            if "eval" in args.stages and (args.overwrite or not summary.exists()):
                run([
                    sys.executable, "scripts/14_evaluate_model.py", "--checkpoint", str(best.relative_to(ROOT)),
                    "--raw-data", spec["eval_raw"], "--feature-data", spec["eval_feature"],
                    "--splits", spec["eval_splits"], "--split", "test", "--model", "paper_starnet",
                    "--model-config", "configs/model/paper_starnet.yaml", "--output", str(output.relative_to(ROOT)),
                    "--summary-output", str(summary.relative_to(ROOT)), "--batch-size", "256",
                    "--num-workers", "0", "--device", args.device, "--strict",
                ], args.dry_run)

        if "summary" in args.stages and not args.dry_run:
            records = []
            for seed in args.seeds:
                path = result_root / f"seed_{seed}_test.summary.json"
                if path.exists():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    accuracy = payload.get("overall_accuracy", payload.get("accuracy"))
                    if accuracy is None:
                        raise KeyError(f"No accuracy field in {path}")
                    records.append({"seed": seed, "accuracy": float(accuracy)})
            if records:
                values = np.asarray([r["accuracy"] for r in records])
                aggregate = {
                    "protocol": protocol,
                    "model": "STARNet",
                    "seeds": records,
                    "n": len(records),
                    "mean_accuracy": float(values.mean()),
                    "sample_std_accuracy": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                }
                (result_root / "five_seed_summary.json").write_text(
                    json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
                )
                print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
