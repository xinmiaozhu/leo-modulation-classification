#!/usr/bin/env python
"""Create auditable four-way pass splits from an existing orbital dataset."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.split import elevation_holdout_four_way_split_indices, save_split_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["elevation_ood", "matched"], default="elevation_ood")
    parser.add_argument("--train-elevations", nargs="+", type=float)
    parser.add_argument("--val-elevations", nargs="+", type=float)
    parser.add_argument("--calibration-elevations", nargs="+", type=float)
    parser.add_argument("--test-elevations", nargs="+", type=float)
    parser.add_argument("--seed", type=int, default=2408)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with h5py.File(args.raw_data, "r") as src:
        elevation = src["maximum_elevation_deg"][:]
        track_id = src["track_id"][:]
        labels = src["label"][:]
        snr_db = src["snr_db"][:]
    if args.mode == "matched":
        spec = importlib.util.spec_from_file_location("track_generator", ROOT / "scripts" / "02_generate_track_dataset.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        splits = module.track_stratified_four_way_frame_splits(
            labels, track_id, elevation, snr_db, 0.625, 0.125, 0.125, 0.125, args.seed
        )
    else:
        required = (args.train_elevations, args.val_elevations, args.calibration_elevations, args.test_elevations)
        if any(value is None for value in required):
            raise ValueError("Elevation lists are required in elevation_ood mode.")
        splits = elevation_holdout_four_way_split_indices(
            elevation,
            args.train_elevations,
            args.val_elevations,
            args.calibration_elevations,
            args.test_elevations,
        )
    for name in ("train", "val", "calibration", "test"):
        splits[f"{name}_track_id"] = np.unique(track_id[splits[name]]).astype(np.int64)
    save_split_indices(splits, args.output)
    print({name: len(splits[name]) for name in ("train", "val", "calibration", "test")})
    print({f"{name}_passes": len(splits[f"{name}_track_id"]) for name in ("train", "val", "calibration", "test")})


if __name__ == "__main__":
    main()
