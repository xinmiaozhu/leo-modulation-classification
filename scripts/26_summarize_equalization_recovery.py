#!/usr/bin/env python
"""Summarize paired exact-mixture representation recovery after equalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unequalized", required=True)
    parser.add_argument("--pilot-ls", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=917)
    return parser.parse_args()


def read(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as src:
        label = src["label"][:].astype(np.int64)
        snr = src["snr_db"][:].astype(float)
        pred = np.argmin(src["all_candidate_penalized_evm"][:], axis=1)
    return label, snr, pred


def paired_ci(base: np.ndarray, changed: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[float, float]:
    delta = changed.astype(float) - base.astype(float)
    values = np.empty(draws, dtype=float)
    for draw in range(draws):
        values[draw] = np.mean(delta[rng.integers(0, len(delta), len(delta))])
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def main() -> None:
    args = parse_args()
    paths = {"unequalized": args.unequalized, "pilot_ls": args.pilot_ls, "oracle": args.oracle}
    data = {name: read(path) for name, path in paths.items()}
    label, snr, _ = data["unequalized"]
    if any(not np.array_equal(label, item[0]) or not np.array_equal(snr, item[1]) for item in data.values()):
        raise ValueError("All conditions must contain the same paired frames in the same order.")
    rows = []
    for name, (_, _, pred) in data.items():
        correct = pred == label
        rows.append({"condition": name, "snr_db": "all", "n": len(label), "accuracy": float(correct.mean())})
        for value in np.unique(snr):
            mask = np.isclose(snr, value)
            rows.append({"condition": name, "snr_db": float(value), "n": int(mask.sum()), "accuracy": float(correct[mask].mean())})
    table = pd.DataFrame(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "exact_mixture_accuracy_by_snr.csv", index=False)
    rng = np.random.default_rng(args.seed)
    base = data["unequalized"][2] == label
    summary = {"bootstrap_draws": args.bootstrap, "seed": args.seed, "conditions": {}}
    for name in ("pilot_ls", "oracle"):
        changed = data[name][2] == label
        lo, hi = paired_ci(base, changed, args.bootstrap, rng)
        summary["conditions"][name] = {
            "accuracy": float(changed.mean()),
            "paired_gain_pp": float(100 * (changed.mean() - base.mean())),
            "paired_gain_95ci_pp": [100 * lo, 100 * hi],
        }
    summary["conditions"]["unequalized"] = {"accuracy": float(base.mean())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
