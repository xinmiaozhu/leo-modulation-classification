"""Smoke tests for dataset generation and stratified splitting."""

from pathlib import Path

import numpy as np

from src.datasets.build_dataset import DatasetBuildConfig, build_dataset_arrays
from src.datasets.split import stratified_split_indices


def test_build_dataset_arrays_small():
    cfg = DatasetBuildConfig(
        seed=0,
        modulations=("BPSK", "QPSK"),
        num_samples_per_class=2,
        num_symbols=32,
        samples_per_symbol=4,
        sample_rate_hz=1e6,
        target_num_samples=128,
        snr_db_min=5,
        snr_db_max=10,
        mu_min_hz_per_s=-1000,
        mu_max_hz_per_s=1000,
    )

    arrays = build_dataset_arrays(cfg, show_progress=False)
    assert arrays["iq"].shape == (4, 2, 128)
    assert arrays["label"].shape == (4,)
    assert arrays["gamma"].shape == (4,)
    assert np.all(np.isfinite(arrays["iq"]))


def test_stratified_split():
    labels = np.array([0, 0, 0, 1, 1, 1])
    splits = stratified_split_indices(labels, train_ratio=0.5, val_ratio=0.0, test_ratio=0.5)
    assert set(splits.keys()) == {"train", "val", "test"}
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(labels)
