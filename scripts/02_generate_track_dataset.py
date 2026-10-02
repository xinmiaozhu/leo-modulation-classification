#!/usr/bin/env python
"""Generate a trajectory-structured LEO signal dataset.

Each track contains consecutive frames from one smooth Doppler-rate profile and
stores ``track_id``, ``frame_id`` and ``frame_time_s`` for pass-level splitting
and trajectory generalization experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.build_track_dataset import build_track_dataset_from_config
from src.datasets.split import (
    elevation_holdout_four_way_split_indices,
    elevation_holdout_split_indices,
    save_split_indices,
    split_summary,
)
from src.utils.config import load_config, save_config
from src.utils.io import save_json
from src.utils.logger import setup_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate trajectory-structured LEO dataset.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/processed/leo_track_dataset.h5")
    parser.add_argument("--splits-output", type=str, default="data/splits/leo_track_dataset_splits.npz")
    parser.add_argument("--summary-output", type=str, default=None)
    parser.add_argument("--resolved-config-output", type=str, default=None)
    parser.add_argument("--override", action="append", default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def track_stratified_frame_splits(
    labels: np.ndarray,
    track_id: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Split by track, then expand selected tracks to frame indices."""

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    track_id = np.asarray(track_id)
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must be positive.")
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

    train_tracks: list[int] = []
    val_tracks: list[int] = []
    test_tracks: list[int] = []

    unique_tracks = np.unique(track_id)
    track_label = {}
    for tid in unique_tracks:
        idx = np.flatnonzero(track_id == tid)
        vals, counts = np.unique(labels[idx], return_counts=True)
        track_label[int(tid)] = int(vals[np.argmax(counts)])

    for label in sorted(set(track_label.values())):
        tracks = np.asarray([tid for tid, lab in track_label.items() if lab == label], dtype=np.int64)
        rng.shuffle(tracks)
        n = len(tracks)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        train_tracks.extend(int(x) for x in tracks[:n_train])
        val_tracks.extend(int(x) for x in tracks[n_train : n_train + n_val])
        test_tracks.extend(int(x) for x in tracks[n_train + n_val :])

    def frames_for_tracks(tracks: list[int]) -> np.ndarray:
        mask = np.isin(track_id, np.asarray(tracks, dtype=track_id.dtype))
        return np.flatnonzero(mask).astype(np.int64)

    return {
        "train": frames_for_tracks(train_tracks),
        "val": frames_for_tracks(val_tracks),
        "test": frames_for_tracks(test_tracks),
        "train_track_id": np.asarray(sorted(train_tracks), dtype=np.int64),
        "val_track_id": np.asarray(sorted(val_tracks), dtype=np.int64),
        "test_track_id": np.asarray(sorted(test_tracks), dtype=np.int64),
    }


def track_stratified_four_way_frame_splits(
    labels: np.ndarray,
    track_id: np.ndarray,
    maximum_elevation_deg: np.ndarray,
    snr_db: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    calibration_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Four-way split of complete passes, stratified by class and elevation."""

    labels = np.asarray(labels)
    track_id = np.asarray(track_id)
    elevation = np.asarray(maximum_elevation_deg)
    ratios = np.asarray([train_ratio, val_ratio, calibration_ratio, test_ratio], dtype=float)
    if np.any(ratios < 0) or ratios.sum() <= 0:
        raise ValueError("Four-way split ratios must be non-negative and have positive sum.")
    ratios /= ratios.sum()
    rng = np.random.default_rng(seed)
    split_names = ("train", "val", "calibration", "test")
    selected: dict[str, list[int]] = {name: [] for name in split_names}

    strata: dict[tuple[int, float], list[int]] = {}
    for tid in np.unique(track_id):
        idx = np.flatnonzero(track_id == tid)
        label = int(np.bincount(labels[idx].astype(np.int64)).argmax())
        elev = float(np.median(elevation[idx]))
        strata.setdefault((label, elev), []).append(int(tid))

    for stratum_index, tracks_list in enumerate(strata.values()):
        tracks = np.asarray(tracks_list, dtype=np.int64)
        raw = ratios * len(tracks)
        counts = np.floor(raw).astype(int)
        for j in np.argsort(-(raw - counts))[: len(tracks) - counts.sum()]:
            counts[j] += 1
        if len(tracks) == 8 and np.array_equal(counts, np.asarray([5, 1, 1, 1])):
            track_snr = {
                int(tid): float(np.mean(np.asarray(snr_db)[track_id == tid])) for tid in tracks
            }
            ordered = np.asarray(sorted(tracks, key=lambda tid: track_snr[int(tid)]), dtype=np.int64)
            # Cycle the three singleton holdouts through the ordered SNR grid.
            # Across class/elevation strata, each split therefore receives the
            # same low-to-high pass-SNR support without inspecting outcomes.
            held_positions = {
                "val": stratum_index % 8,
                "calibration": (stratum_index + 3) % 8,
                "test": (stratum_index + 5) % 8,
            }
            held = set(held_positions.values())
            selected["train"].extend(int(ordered[j]) for j in range(8) if j not in held)
            for name, position in held_positions.items():
                selected[name].append(int(ordered[position]))
        else:
            rng.shuffle(tracks)
            start = 0
            for name, count in zip(split_names, counts):
                selected[name].extend(int(x) for x in tracks[start : start + count])
                start += count

    result: dict[str, np.ndarray] = {}
    for name in split_names:
        tids = np.asarray(sorted(selected[name]), dtype=np.int64)
        result[name] = np.flatnonzero(np.isin(track_id, tids)).astype(np.int64)
        result[f"{name}_track_id"] = tids
    return result


def add_split_track_ids(splits: dict[str, np.ndarray], track_id: np.ndarray) -> dict[str, np.ndarray]:
    """Add auditable pass identifiers to any frame-index split."""

    result = dict(splits)
    for name in ("train", "val", "calibration", "test"):
        if name in result:
            result[f"{name}_track_id"] = np.unique(track_id[result[name]]).astype(np.int64)
    return result


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    logger = setup_logger("leo_drc_dualnet", Path(args.output).with_suffix(".log"))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resolved_config_output:
        save_config(cfg, args.resolved_config_output)

    arrays, summary = build_track_dataset_from_config(
        cfg,
        output_h5=str(output_path),
        show_progress=not args.no_progress,
    )

    split_cfg = cfg.get("split", {})
    train_ratio = float(split_cfg.get("train_ratio", 0.7))
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
    test_ratio = float(split_cfg.get("test_ratio", 0.2))
    calibration_ratio = float(split_cfg.get("calibration_ratio", 0.0))
    split_mode = str(split_cfg.get("mode", "track_stratified"))
    if split_mode == "elevation_holdout_four_way":
        splits = elevation_holdout_four_way_split_indices(
            arrays["maximum_elevation_deg"],
            list(split_cfg.get("train_elevations_deg", [])),
            list(split_cfg.get("val_elevations_deg", [])),
            list(split_cfg.get("calibration_elevations_deg", [])),
            list(split_cfg.get("test_elevations_deg", [])),
        )
        splits = add_split_track_ids(splits, arrays["track_id"])
    elif split_mode == "elevation_holdout":
        splits = elevation_holdout_split_indices(
            arrays["maximum_elevation_deg"],
            list(split_cfg.get("train_elevations_deg", [])),
            list(split_cfg.get("val_elevations_deg", [])),
            list(split_cfg.get("test_elevations_deg", [])),
        )
        splits = add_split_track_ids(splits, arrays["track_id"])
    elif split_mode == "track_stratified_four_way":
        splits = track_stratified_four_way_frame_splits(
            arrays["label"], arrays["track_id"], arrays["maximum_elevation_deg"], arrays["snr_db"],
            train_ratio, val_ratio, calibration_ratio, test_ratio, seed,
        )
    else:
        splits = track_stratified_frame_splits(
            arrays["label"], arrays["track_id"], train_ratio, val_ratio, test_ratio, seed
        )
    splits_output = Path(args.splits_output)
    splits_output.parent.mkdir(parents=True, exist_ok=True)
    save_split_indices(splits, splits_output)

    summary["split_mode"] = split_mode
    summary["splits"] = split_summary({k: v for k, v in splits.items() if not k.endswith("_track_id")})
    summary["output_h5"] = str(output_path)
    summary["splits_output"] = str(splits_output)

    summary_output = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    save_json(summary, summary_output)
    logger.info("Trajectory dataset generation completed.")
    logger.info(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
