#!/usr/bin/env python
"""Matched-seed representation and added-HOC ablations for final Protocols A/B.

Protocol B uses DCFT acquisition and local coherent refinement; hybrid_dfrft
is the historical artifact directory name. Existing final test predictions
are reused, and HOC is computed from the exact frozen compensated I/Q.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
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
        "train_feature": "data/features/hybrid_dfrft/protocol_b_iqcomp.h5",
        "train_symbol": "data/features/hybrid_dfrft/protocol_b_symbol_exact.h5",
        "train_splits": "data/splits/leo_7mods_joint_practical_splits.npz",
        "eval_raw": "data/processed/leo_7mods_joint_practical.h5",
        "eval_feature": "data/features/hybrid_dfrft/protocol_b_iqcomp.h5",
        "eval_symbol": "data/features/hybrid_dfrft/protocol_b_symbol_exact.h5",
        "eval_splits": "data/splits/leo_7mods_joint_practical_splits.npz",
    },
}

METHODS = {
    "iq_only": "configs/model/dualnet_iq_only.yaml",
    "constellation_only": "configs/model/dualnet_evm_only.yaml",
    "iq_hoc": "configs/model/dualnet_iq_only.yaml",
    "fusion_hoc": "configs/model/dualnet_iq_evm.yaml",
}
DISPLAY = {"iq_only": "I/Q only", "constellation_only": "Constellation only",
           "fusion": "Fusion", "fusion_hoc": "I/Q + 48-D descriptor + HOC",
           "iq_hoc": "I/Q + HOC"}
HOC_METHODS = {"fusion_hoc", "iq_hoc"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocols", nargs="+", choices=["A", "B"], default=["A", "B"])
    p.add_argument("--methods", nargs="+", choices=list(DISPLAY), default=list(DISPLAY))
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--stages", nargs="+", choices=["features", "train", "eval", "summary"],
                   default=["features", "train", "eval", "summary"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        p.error("Seeds must be unique.")
    return args


def run(command, log_path, dry_run=False):
    print("$ " + " ".join(map(str, command)), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(map(str, command)) + "\n")
        log.flush()
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def family(protocol):
    return "exact_mixture" if protocol == "A" else "hybrid_dfrft"


def result_root(protocol):
    return ROOT / "outputs/results" / family(protocol) / "representation_ablation" / f"protocol_{protocol.lower()}"


def branch_paths(protocol, method, seed):
    root = result_root(protocol)
    checkpoint = ROOT / "outputs/checkpoints" / family(protocol) / "representation_ablation" / f"protocol_{protocol.lower()}" / method / f"seed_{seed}"
    output = root / method / f"seed_{seed}_test.csv"
    return checkpoint, output, output.with_suffix(".summary.json")


def fusion_paths(protocol, seed):
    suffix = Path(f"protocol_{protocol.lower()}") / "proposed" / f"seed_{seed}"
    checkpoint = ROOT / "outputs/checkpoints" / family(protocol) / suffix / "best.pt"
    output = ROOT / "outputs/results" / family(protocol) / suffix.parent / f"seed_{seed}_test.csv"
    return checkpoint, output


def hoc_path(feature):
    p = Path(feature)
    return str(p.with_name(p.stem + "_hoc_ablation.h5"))


def validate_inputs(cfg):
    for stage in ("train", "eval"):
        with h5py.File(ROOT / cfg[stage + "_raw"], "r") as raw, \
             h5py.File(ROOT / cfg[stage + "_feature"], "r") as feature, \
             h5py.File(ROOT / cfg[stage + "_symbol"], "r") as symbol:
            if symbol["evm_features"].shape != (len(raw["label"]), 48):
                raise ValueError("The final ablation requires the full 48-D descriptor.")
            for other in (feature, symbol):
                for key in ("label", "snr_db"):
                    if not np.array_equal(raw[key][:], other[key][:]):
                        raise ValueError(f"Raw/feature alignment mismatch: {stage} {key}")
        with np.load(ROOT / cfg[stage + "_splits"]) as splits:
            names = [key for key in ("train", "val", "test") if key in splits]
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    if np.intersect1d(splits[a], splits[b]).size:
                        raise ValueError(f"Overlapping {a}/{b} split.")


def prediction_table(path, cfg):
    table = pd.read_csv(path)
    required = {"index", "label", "pred", "snr_db"}
    if not required.issubset(table):
        raise ValueError(f"Missing per-sample fields in {path}")
    with np.load(ROOT / cfg["eval_splits"]) as splits:
        indices = np.sort(splits["test"].astype(int))
    table = table.sort_values("index").reset_index(drop=True)
    if not np.array_equal(table["index"].to_numpy(), indices):
        raise ValueError(f"Test indices mismatch or duplicate rows: {path}")
    with h5py.File(ROOT / cfg["eval_raw"], "r") as raw:
        if not np.array_equal(table.label.to_numpy(), raw["label"][indices]):
            raise ValueError(f"Test labels mismatch: {path}")
        if not np.allclose(table.snr_db.to_numpy(), raw["snr_db"][indices], rtol=0, atol=1e-6):
            raise ValueError(f"Test SNR mismatch: {path}")
    return table


def summarize(protocol, requested_methods, seeds):
    cfg = PROTOCOLS[protocol]
    rows = []
    for method in DISPLAY:
        for seed in seeds:
            if method == "fusion":
                checkpoint, output = fusion_paths(protocol, seed)
            else:
                directory, output, _ = branch_paths(protocol, method, seed)
                checkpoint = directory / "best.pt"
            if not output.exists():
                if method in requested_methods:
                    raise ValueError(f"Missing requested evaluation: {output}")
                continue
            table = prediction_table(output, cfg)
            rows.append({"protocol": protocol, "front_end": "coherent" if protocol == "A" else "dcft_coherent",
                         "method": method, "display": DISPLAY[method], "seed": seed,
                         "accuracy": float((table.label == table.pred).mean()),
                         "eval_csv": str(output.relative_to(ROOT)),
                         "checkpoint": str(checkpoint.relative_to(ROOT))})
    if not rows:
        raise ValueError("No completed evaluations to summarize.")
    root = result_root(protocol)
    root.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows).sort_values(["seed", "method"])
    # Rebuild from predictions, preserving earlier completed seeds/methods.
    old = root / "seed_runs.csv"
    if old.exists():
        previous = pd.read_csv(old)
        extra_seeds = sorted(set(previous.seed.astype(int)) - set(seeds))
        if extra_seeds:
            return summarize(protocol, requested_methods, sorted(set(seeds) | set(extra_seeds)))
    runs.to_csv(old, index=False)
    aggregate = runs.groupby(["method", "display"], sort=False).accuracy.agg(["count", "mean", "std", "min", "max"]).reset_index()
    aggregate.to_csv(root / "seed_summary.csv", index=False)
    pivot = runs.pivot(index="seed", columns="method", values="accuracy")
    payload = {"protocol": protocol, "front_end": "coherent" if protocol == "A" else "dcft_coherent",
               "seeds": sorted(runs.seed.unique().tolist()), "feature_definition": "full 48-D exact-mixture descriptor; optional 21-D HOC",
               "formal_five_seed_complete": all(set(runs.loc[runs.method == m, "seed"]) == set(SEEDS) for m in DISPLAY)}
    comparisons = [(["iq_only", "constellation_only", "fusion"], "paired_fusion_gain", "fusion_gain_over_stronger_single_pp"),
                   (["fusion", "fusion_hoc"], "paired_hoc_gain", "hoc_gain_over_fusion_pp"),
                   (["iq_hoc", "fusion"], "paired_descriptor_vs_hoc", "descriptor_gain_over_hoc_pp"),
                   (["iq_only", "iq_hoc"], "paired_iq_hoc_gain", "hoc_gain_over_iq_only_pp")]
    for columns, key, gain_column in comparisons:
        if not set(columns).issubset(pivot.columns):
            continue
        paired = pivot[columns].dropna().copy()
        if not len(paired):
            continue
        base = paired[["iq_only", "constellation_only"]].max(axis=1) if key == "paired_fusion_gain" else paired[columns[0]]
        gains = 100 * (paired[columns[-1]] - base)
        paired[gain_column] = gains
        paired.reset_index().to_csv(root / (key + ".csv"), index=False)
        payload[key] = {"count": len(gains), "mean_pp": float(gains.mean()),
                        "sample_sd_pp": float(gains.std(ddof=1)) if len(gains) > 1 else None,
                        "positive_seeds": int((gains > 0).sum())}
    (root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(aggregate.to_string(index=False), flush=True)
    print(json.dumps(payload, indent=2), flush=True)


def main(argv=None):
    args = parse_args(argv)
    for protocol in args.protocols:
        cfg = PROTOCOLS[protocol]
        missing = [path for path in cfg.values() if not (ROOT / path).exists()]
        if missing:
            raise SystemExit(f"Protocol {protocol} missing inputs: {missing}")
        if not args.dry_run:
            validate_inputs(cfg)
        root = result_root(protocol)
        if HOC_METHODS.intersection(args.methods) and "features" in args.stages:
            prepared = set()
            for stage in ("train", "eval"):
                feature = cfg[stage + "_feature"]
                if feature in prepared:
                    continue
                command = [sys.executable, "scripts/08_precompute_external_mu_hoc.py",
                           "--raw-data", cfg[stage + "_raw"], "--input-iq-feature-data", feature,
                           "--output", hoc_path(feature), "--no-progress"]
                if args.overwrite:
                    command.append("--overwrite")
                run(command, root / "hoc_features.log", args.dry_run)
                prepared.add(feature)
        for method in args.methods:
            for seed in args.seeds:
                if method == "fusion":
                    _, output = fusion_paths(protocol, seed)
                    if not args.dry_run and not output.exists():
                        raise SystemExit(f"Missing final fusion predictions: {output}; run the main experiment first.")
                    continue
                directory, output, summary = branch_paths(protocol, method, seed)
                best = directory / "best.pt"
                # Final archived predictions remain usable without archived checkpoints.
                if not args.overwrite and output.exists() and summary.exists():
                    continue
                feature_train = hoc_path(cfg["train_feature"]) if method in HOC_METHODS else cfg["train_feature"]
                feature_eval = hoc_path(cfg["eval_feature"]) if method in HOC_METHODS else cfg["eval_feature"]
                if "train" in args.stages and (args.overwrite or not (directory / "train_result.json").exists() or not best.exists()):
                    command = [sys.executable, "scripts/12_train_model.py", "--raw-data", cfg["train_raw"],
                               "--feature-data", feature_train, "--symbol-feature-data", cfg["train_symbol"],
                               "--splits", cfg["train_splits"], "--model", "drc_dualnet", "--model-config", METHODS[method],
                               "--train-config", "configs/train/train_long.yaml", "--output-dir", str(directory.relative_to(ROOT)),
                               "--seed", str(seed), "--device", args.device, "--num-workers", "0", "--iq-source", "comp",
                               "--iq-representation", "iq", "--iq-normalize", "zscore", "--cache-iq"]
                    if method in HOC_METHODS:
                        command.append("--use-hoc-stream")
                    if method == "iq_hoc":
                        # Match the descriptor encoder's hidden widths and
                        # 128-D embedding; only the input dimension differs.
                        command.extend(["--hoc-hidden-dims", "64", "64"])
                    if (directory / "last.pt").exists() and not args.overwrite:
                        command.extend(["--resume", str((directory / "last.pt").relative_to(ROOT))])
                    run(command, directory / "runner.log", args.dry_run)
                if "eval" in args.stages:
                    if not args.dry_run and not best.exists():
                        raise SystemExit(f"Missing checkpoint: {best}")
                    # Use the checkpoint model configuration, including the optional HOC branch.
                    command = [sys.executable, "scripts/13_evaluate_model.py", "--checkpoint", str(best.relative_to(ROOT)),
                               "--raw-data", cfg["eval_raw"], "--feature-data", feature_eval,
                               "--symbol-feature-data", cfg["eval_symbol"], "--splits", cfg["eval_splits"],
                               "--split", "test", "--model", "drc_dualnet", "--output", str(output.relative_to(ROOT)),
                               "--summary-output", str(summary.relative_to(ROOT)), "--batch-size", "256", "--num-workers", "0",
                               "--device", args.device, "--strict", "--load-into-memory"]
                    run(command, directory / "runner.log", args.dry_run)
        if "summary" in args.stages and not args.dry_run:
            summarize(protocol, args.methods, args.seeds)


if __name__ == "__main__":
    main()
