#!/usr/bin/env python
"""Audit pass disjointness and covariate balance for four-way splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with h5py.File(args.raw_data, "r") as src:
        snr = src["snr_db"][:].astype(float)
        label = src["label"][:].astype(int)
        elevation = src["maximum_elevation_deg"][:].astype(float)
    report: dict[str, object] = {"raw_data": args.raw_data, "splits": {}}
    names = ("train", "val", "calibration", "test")
    for path in args.splits:
        data = np.load(path)
        track_sets = [set(data[f"{name}_track_id"].tolist()) for name in names]
        item: dict[str, object] = {
            "pairwise_track_disjoint": all(
                not track_sets[i].intersection(track_sets[j])
                for i in range(len(names)) for j in range(i + 1, len(names))
            ),
            "groups": {},
        }
        for name in names:
            idx = data[name]
            item["groups"][name] = {
                "frames": int(len(idx)),
                "passes": int(len(data[f"{name}_track_id"])),
                "snr_mean_db": float(np.mean(snr[idx])),
                "snr_std_db": float(np.std(snr[idx])),
                "snr_p10_db": float(np.quantile(snr[idx], 0.1)),
                "snr_p90_db": float(np.quantile(snr[idx], 0.9)),
                "class_counts": np.bincount(label[idx]).astype(int).tolist(),
                "elevations_deg": sorted(float(x) for x in np.unique(elevation[idx])),
            }
        report["splits"][path] = item
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
