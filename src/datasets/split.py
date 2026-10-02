"""Dataset splitting utilities.

The project often needs two kinds of splits:
    1. standard random/stratified train-val-test splits;
    2. domain-holdout splits for cross-domain generalization tests.

All functions return integer index arrays, which are easy to save as .npz files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.utils.io import ensure_parent


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if any(r < 0 for r in [train_ratio, val_ratio, test_ratio]):
        raise ValueError("Split ratios must be non-negative.")
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def random_split_indices(
    num_samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Randomly split indices into train/val/test."""

    _validate_ratios(train_ratio, val_ratio, test_ratio)

    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(indices)

    n_train = int(round(num_samples * train_ratio))
    n_val = int(round(num_samples * val_ratio))
    n_train = min(n_train, num_samples)
    n_val = min(n_val, num_samples - n_train)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def stratified_split_indices(
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Stratified split by class label."""

    _validate_ratios(train_ratio, val_ratio, test_ratio)

    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    train_list: list[np.ndarray] = []
    val_list: list[np.ndarray] = []
    test_list: list[np.ndarray] = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)

        n = len(cls_idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        train_list.append(cls_idx[:n_train])
        val_list.append(cls_idx[n_train : n_train + n_val])
        test_list.append(cls_idx[n_train + n_val :])

    train = np.sort(np.concatenate(train_list).astype(np.int64))
    val = np.sort(np.concatenate(val_list).astype(np.int64))
    test = np.sort(np.concatenate(test_list).astype(np.int64))

    return {"train": train, "val": val, "test": test}


def domain_holdout_split_indices(
    domain_ids: np.ndarray,
    test_domain_ids: Sequence[int],
    val_ratio_within_train: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Hold out one or more domains as the test set.

    Args:
        domain_ids: Domain ID for every sample.
        test_domain_ids: Domain IDs assigned to test set.
        val_ratio_within_train: Fraction of non-test-domain samples used for val.
    """

    domain_ids = np.asarray(domain_ids)
    test_domain_ids = set(int(x) for x in test_domain_ids)

    all_idx = np.arange(len(domain_ids), dtype=np.int64)
    test_mask = np.isin(domain_ids, list(test_domain_ids))

    test_idx = all_idx[test_mask]
    trainval_idx = all_idx[~test_mask]

    rng = np.random.default_rng(seed)
    rng.shuffle(trainval_idx)

    n_val = int(round(len(trainval_idx) * val_ratio_within_train))
    val_idx = np.sort(trainval_idx[:n_val])
    train_idx = np.sort(trainval_idx[n_val:])
    test_idx = np.sort(test_idx)

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def elevation_holdout_split_indices(
    maximum_elevation_deg: np.ndarray,
    train_elevations_deg: Sequence[float],
    val_elevations_deg: Sequence[float],
    test_elevations_deg: Sequence[float],
) -> dict[str, np.ndarray]:
    """Assign every complete pass to one explicitly disjoint elevation group."""

    elevation = np.asarray(maximum_elevation_deg, dtype=np.float64)
    groups = {
        "train": np.asarray(train_elevations_deg, dtype=np.float64),
        "val": np.asarray(val_elevations_deg, dtype=np.float64),
        "test": np.asarray(test_elevations_deg, dtype=np.float64),
    }
    assigned = np.zeros(len(elevation), dtype=np.int64)
    result: dict[str, np.ndarray] = {}
    for split_name, values in groups.items():
        mask = np.any(np.isclose(elevation[:, None], values[None, :], atol=1e-6), axis=1)
        result[split_name] = np.flatnonzero(mask).astype(np.int64)
        assigned += mask.astype(np.int64)
    if np.any(assigned != 1):
        raise ValueError("Every pass must map to exactly one elevation holdout split.")
    return result


def elevation_holdout_four_way_split_indices(
    maximum_elevation_deg: np.ndarray,
    train_elevations_deg: Sequence[float],
    val_elevations_deg: Sequence[float],
    calibration_elevations_deg: Sequence[float],
    test_elevations_deg: Sequence[float],
) -> dict[str, np.ndarray]:
    """Assign complete passes to four explicitly disjoint elevation groups."""

    elevation = np.asarray(maximum_elevation_deg, dtype=np.float64)
    groups = {
        "train": np.asarray(train_elevations_deg, dtype=np.float64),
        "val": np.asarray(val_elevations_deg, dtype=np.float64),
        "calibration": np.asarray(calibration_elevations_deg, dtype=np.float64),
        "test": np.asarray(test_elevations_deg, dtype=np.float64),
    }
    assigned = np.zeros(len(elevation), dtype=np.int64)
    result: dict[str, np.ndarray] = {}
    for split_name, values in groups.items():
        mask = np.any(np.isclose(elevation[:, None], values[None, :], atol=1e-6), axis=1)
        result[split_name] = np.flatnonzero(mask).astype(np.int64)
        assigned += mask.astype(np.int64)
    if np.any(assigned != 1):
        raise ValueError("Every pass must map to exactly one four-way elevation split.")
    return result


def save_split_indices(splits: dict[str, np.ndarray], path: str | Path) -> None:
    """Save split indices as compressed npz."""

    path = ensure_parent(path)
    np.savez_compressed(path, **{k: np.asarray(v, dtype=np.int64) for k, v in splits.items()})


def load_split_indices(path: str | Path) -> dict[str, np.ndarray]:
    """Load split indices from npz."""

    data = np.load(path)
    return {k: data[k].astype(np.int64) for k in data.files}


def split_summary(splits: dict[str, np.ndarray]) -> dict[str, int]:
    """Return number of samples in each split."""

    return {k: int(len(v)) for k, v in splits.items()}
