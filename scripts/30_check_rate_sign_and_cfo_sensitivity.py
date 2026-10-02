#!/usr/bin/env python
"""Check residual-rate sign symmetry and small residual-CFO sensitivity."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_scan_module():
    path = ROOT / "scripts" / "27_validate_feature_degradation.py"
    spec = importlib.util.spec_from_file_location("feature_scan_54", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rate-detail-csv", required=True)
    p.add_argument("--raw-data", required=True)
    p.add_argument("--feature-data", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--snr-db", type=float, nargs="+", default=[-7, -4, -1, 2, 5])
    p.add_argument("--cfo-grid-hz", type=float, nargs="+", default=[-10, -5, 0, 5, 10])
    p.add_argument("--samples-per-class-snr", type=int, default=20)
    p.add_argument("--seed", type=int, default=545454)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--oracle-correct-estimated-iq",
        action="store_true",
        help=(
            "Remove the stored estimator CFO/rate residual from iq_comp before "
            "injecting the controlled residual CFO."
        ),
    )
    p.add_argument("--output-root", default="outputs/results/exact_mixture/conditional_rule_checks")
    return p.parse_args()


def sign_table(path: str) -> pd.DataFrame:
    detail = pd.read_csv(path)
    detail = detail[~np.isclose(detail["gamma_res"].astype(float), 0.0)].copy()
    grouped = detail.groupby(
        ["model", "snr_db", "gamma_res", "injection_sign"], as_index=False
    )["correct"].mean()
    pivot = grouped.pivot(
        index=["model", "snr_db", "gamma_res"], columns="injection_sign", values="correct"
    ).reset_index()
    if -1.0 not in pivot or 1.0 not in pivot:
        raise ValueError("Both positive and negative rate injections are required.")
    pivot = pivot.rename(columns={-1.0: "accuracy_negative", 1.0: "accuracy_positive"})
    pivot["absolute_sign_gap_pp"] = 100.0 * np.abs(
        pivot["accuracy_positive"] - pivot["accuracy_negative"]
    )
    return pivot


def main() -> None:
    args = parse_args()
    module = load_scan_module()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    signs = sign_table(args.rate_detail_csv)
    signs.to_csv(output / "rate_sign_asymmetry.csv", index=False)
    sign_summary = signs.groupby(["model", "snr_db"])["absolute_sign_gap_pp"].agg(
        mean_gap_pp="mean", max_gap_pp="max"
    ).reset_index()
    sign_summary.to_csv(output / "rate_sign_asymmetry_summary.csv", index=False)

    with np.load(args.splits) as split_file:
        test_indices = np.asarray(split_file["test"], dtype=np.int64)
    with h5py.File(args.raw_data, "r") as raw:
        # _controlled_classification_scan expects labels/SNR arrays aligned
        # positionally with the supplied global frame indices.
        labels = np.asarray(raw["label"][test_indices], dtype=np.int64)
        snr = np.asarray(raw["snr_db"][test_indices], dtype=np.float64)

    cfo_rows = []
    for cfo in args.cfo_grid_hz:
        table, _, _ = module._controlled_classification_scan(
            raw_path=Path(args.raw_data), feature_path=Path(args.feature_data),
            indices=test_indices, labels=labels, snr_db=snr,
            gamma_grid=np.asarray([0.0]),
            samples_per_class_snr=int(args.samples_per_class_snr), seed=int(args.seed),
            checkpoints={"iq_evm": Path(args.checkpoint)}, device_name=args.device,
            batch_size=int(args.batch_size), snr_values=np.asarray(args.snr_db, dtype=float),
            residual_cfo_hz=float(cfo),
            oracle_correct_estimated_iq=bool(args.oracle_correct_estimated_iq),
        )
        cfo_rows.append(table)
    cfo_table = pd.concat(cfo_rows, ignore_index=True)
    cfo_table.to_csv(output / "residual_cfo_sensitivity.csv", index=False)
    zero = cfo_table[np.isclose(cfo_table["residual_cfo_hz"], 0.0)][
        ["snr_db", "accuracy"]
    ].rename(columns={"accuracy": "accuracy_cfo0"})
    cfo_table = cfo_table.merge(zero, on="snr_db", how="left")
    cfo_table["accuracy_change_pp"] = 100.0 * (
        cfo_table["accuracy"] - cfo_table["accuracy_cfo0"]
    )
    cfo_table.to_csv(output / "residual_cfo_sensitivity_with_delta.csv", index=False)
    cfo_table["abs_cfo_hz"] = np.abs(cfo_table["residual_cfo_hz"].astype(float))
    cfo_summary = (
        cfo_table.groupby("abs_cfo_hz")["accuracy_change_pp"]
        .agg(
            mean_change_pp="mean",
            median_change_pp="median",
            worst_change_pp="min",
            best_change_pp="max",
        )
        .reset_index()
    )
    cfo_summary["max_abs_change_pp"] = cfo_table.groupby("abs_cfo_hz")[
        "accuracy_change_pp"
    ].apply(lambda values: float(np.max(np.abs(values.to_numpy(dtype=float))))).to_numpy()
    cfo_summary.to_csv(output / "residual_cfo_sensitivity_summary.csv", index=False)
    summary = {
        "rate_detail_csv": args.rate_detail_csv,
        "checkpoint": args.checkpoint,
        "split": "test",
        "snr_db": [float(value) for value in args.snr_db],
        "cfo_grid_hz": [float(value) for value in args.cfo_grid_hz],
        "samples_per_class_snr": int(args.samples_per_class_snr),
        "seed": int(args.seed),
        "oracle_correct_estimated_iq": bool(args.oracle_correct_estimated_iq),
        "rate_sign_gap_pp": {
            "mean": float(signs["absolute_sign_gap_pp"].mean()),
            "median": float(signs["absolute_sign_gap_pp"].median()),
            "max": float(signs["absolute_sign_gap_pp"].max()),
        },
        "cfo_by_absolute_hz": cfo_summary.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(sign_summary.to_string(index=False))
    print(cfo_table.to_string(index=False))


if __name__ == "__main__":
    main()
