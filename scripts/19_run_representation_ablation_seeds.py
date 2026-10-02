#!/usr/bin/env python
"""Run the final 48-D I/Q--constellation representation ablation.

The experiment compares I/Q-only, constellation-only, and deployed fusion
receivers with matched seeds under Protocols A and B.  Existing formal fusion
checkpoints/results are reused; the two single-stream variants are trained and
tested with the same front end, splits, optimizer, and exact-mixture features.
"""

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
    "iq_only": "configs/model/triplenet_iq_only.yaml",
    "constellation_only": "configs/model/triplenet_evm_only.yaml",
}
DISPLAY = {
    "iq_only": "I/Q only",
    "constellation_only": "Constellation only",
    "fusion": "Fusion",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocols", nargs="+", choices=["A", "B"], default=["A", "B"])
    p.add_argument(
        "--methods", nargs="+", choices=[*METHODS, "fusion"],
        default=["iq_only", "constellation_only", "fusion"],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument(
        "--stages", nargs="+", choices=["train", "eval", "summary"],
        default=["train", "eval", "summary"],
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(command)
    print("\n$ " + printable, flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + printable + "\n")
        log.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def accuracy_from_csv(path: Path) -> float:
    table = pd.read_csv(path)
    if "correct" in table:
        return float(table["correct"].astype(float).mean())
    pred = "pred" if "pred" in table else "predicted"
    return float((table["label"].to_numpy() == table[pred].to_numpy()).mean())


def branch_paths(protocol: str, method: str, seed: int) -> tuple[Path, Path, Path]:
    root = ROOT / "outputs" / "results" / "exact_mixture" / "representation_ablation" / f"protocol_{protocol.lower()}"
    checkpoint = (
        ROOT / "outputs" / "checkpoints" / "exact_mixture" / "representation_ablation"
        / f"protocol_{protocol.lower()}" / method / f"seed_{seed}"
    )
    output = root / method / f"seed_{seed}_test.csv"
    summary = root / method / f"seed_{seed}_test.summary.json"
    return checkpoint, output, summary


def fusion_paths(protocol: str, seed: int) -> tuple[Path, Path]:
    checkpoint = (
        ROOT / "outputs" / "checkpoints" / "exact_mixture"
        / f"protocol_{protocol.lower()}" / "proposed" / f"seed_{seed}" / "best.pt"
    )
    output = (
        ROOT / "outputs" / "results" / "exact_mixture"
        / f"protocol_{protocol.lower()}" / "proposed" / f"seed_{seed}_test.csv"
    )
    return checkpoint, output


def summarize(protocol: str, rows: list[dict[str, object]]) -> None:
    root = ROOT / "outputs" / "results" / "exact_mixture" / "representation_ablation" / f"protocol_{protocol.lower()}"
    root.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows).sort_values(["seed", "method"]).reset_index(drop=True)
    runs.to_csv(root / "seed_runs.csv", index=False)
    summary = (
        runs.groupby(["method", "display"], sort=False)["accuracy"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    summary.to_csv(root / "seed_summary.csv", index=False)

    pivot = runs.pivot(index="seed", columns="method", values="accuracy")
    required = {"iq_only", "constellation_only", "fusion"}
    paired_rows: list[dict[str, float | int]] = []
    if required.issubset(pivot.columns):
        for seed, row in pivot.iterrows():
            stronger = max(float(row["iq_only"]), float(row["constellation_only"]))
            paired_rows.append(
                {
                    "seed": int(seed),
                    "iq_only_accuracy": float(row["iq_only"]),
                    "constellation_only_accuracy": float(row["constellation_only"]),
                    "fusion_accuracy": float(row["fusion"]),
                    "fusion_gain_over_stronger_single_pp": 100.0 * (float(row["fusion"]) - stronger),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(root / "paired_fusion_gain.csv", index=False)
    gains = paired.get("fusion_gain_over_stronger_single_pp", pd.Series(dtype=float)).to_numpy(float)
    payload = {
        "protocol": protocol,
        "seeds": sorted(runs["seed"].astype(int).unique().tolist()),
        "feature_definition": "final 48-D exact-mixture/geometry descriptors",
        "summary_csv": str((root / "seed_summary.csv").relative_to(ROOT)),
        "paired_gain_csv": str((root / "paired_fusion_gain.csv").relative_to(ROOT)),
        "paired_fusion_gain": {
            "count": int(gains.size),
            "mean_pp": float(np.mean(gains)) if gains.size else None,
            "sample_sd_pp": float(np.std(gains, ddof=1)) if gains.size > 1 else None,
            "positive_seeds": int(np.sum(gains > 0.0)) if gains.size else 0,
        },
    }
    (root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n" + summary.to_string(index=False), flush=True)
    if gains.size:
        print(
            f"Protocol {protocol} paired fusion gain: {np.mean(gains):.3f} +/- "
            f"{np.std(gains, ddof=1):.3f} pp; positive {np.sum(gains > 0)}/{gains.size}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    for protocol in args.protocols:
        cfg = PROTOCOLS[protocol]
        missing = [path for path in cfg.values() if not (ROOT / path).exists()]
        if missing:
            raise SystemExit(f"Protocol {protocol} missing inputs:\n" + "\n".join(missing))
        rows: list[dict[str, object]] = []
        for method in args.methods:
            for seed in args.seeds:
                if method == "fusion":
                    best, output = fusion_paths(protocol, seed)
                    if not best.exists() or not output.exists():
                        raise SystemExit(
                            f"Missing formal fusion artifact for Protocol {protocol}, seed {seed}: "
                            f"{best} or {output}"
                        )
                else:
                    checkpoint_dir, output, eval_summary = branch_paths(protocol, method, seed)
                    best = checkpoint_dir / "best.pt"
                    complete = checkpoint_dir / "train_result.json"
                    log_path = checkpoint_dir / "runner.log"
                    train_cmd = [
                        sys.executable, "scripts/13_train_model.py",
                        "--raw-data", cfg["train_raw"],
                        "--feature-data", cfg["train_feature"],
                        "--symbol-feature-data", cfg["train_symbol"],
                        "--splits", cfg["train_splits"],
                        "--model", "drc_triplenet",
                        "--model-config", METHODS[method],
                        "--train-config", "configs/train/train_long.yaml",
                        "--output-dir", str(checkpoint_dir.relative_to(ROOT)),
                        "--seed", str(seed),
                        "--device", args.device,
                        "--num-workers", "0",
                        "--iq-source", "comp",
                        "--iq-representation", "iq",
                        "--iq-normalize", "zscore",
                    ]
                    if method == "iq_only":
                        train_cmd.append("--cache-iq")
                    if "train" in args.stages and (args.overwrite or not complete.exists()):
                        last = checkpoint_dir / "last.pt"
                        if last.exists() and not args.overwrite:
                            train_cmd.extend(["--resume", str(last.relative_to(ROOT))])
                        run(train_cmd, log_path)
                    if "eval" in args.stages and (args.overwrite or not eval_summary.exists()):
                        if not best.exists():
                            raise SystemExit(f"Missing checkpoint: {best}")
                        output.parent.mkdir(parents=True, exist_ok=True)
                        eval_cmd = [
                            sys.executable, "scripts/14_evaluate_model.py",
                            "--checkpoint", str(best.relative_to(ROOT)),
                            "--raw-data", cfg["eval_raw"],
                            "--feature-data", cfg["eval_feature"],
                            "--symbol-feature-data", cfg["eval_symbol"],
                            "--splits", cfg["eval_splits"],
                            "--split", "test",
                            "--model", "drc_triplenet",
                            "--model-config", METHODS[method],
                            "--output", str(output.relative_to(ROOT)),
                            "--summary-output", str(eval_summary.relative_to(ROOT)),
                            "--batch-size", "256",
                            "--num-workers", "0",
                            "--device", args.device,
                            "--strict",
                            "--load-into-memory",
                        ]
                        run(eval_cmd, log_path)
                if output.exists():
                    rows.append(
                        {
                            "protocol": protocol,
                            "method": method,
                            "display": DISPLAY[method],
                            "seed": seed,
                            "accuracy": accuracy_from_csv(output),
                            "eval_csv": str(output.relative_to(ROOT)),
                            "checkpoint": str(best.relative_to(ROOT)),
                        }
                    )
        if "summary" in args.stages:
            expected = len(args.methods) * len(args.seeds)
            if len(rows) != expected:
                raise SystemExit(
                    f"Protocol {protocol}: found {len(rows)}/{expected} completed evaluations; "
                    "summary not written."
                )
            summarize(protocol, rows)


if __name__ == "__main__":
    main()
