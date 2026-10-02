#!/usr/bin/env python
"""Generate a synthetic LEO signal-recognition dataset.

Example:
    python scripts/01_generate_dataset.py \
        --config configs/dataset/leo_sband.yaml \
        --output data/processed/leo_sband.h5 \
        --splits-output data/splits/leo_sband_splits.npz

You can override YAML values from command line:
    python scripts/01_generate_dataset.py \
        --config configs/dataset/leo_sband.yaml \
        --override signal.num_samples_per_class=100 \
        --override channel.snr_db_range='[-5, 15]'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running the script directly from project root without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.build_dataset import build_dataset_from_config
from src.datasets.split import (
    domain_holdout_split_indices,
    random_split_indices,
    save_split_indices,
    split_summary,
    stratified_split_indices,
)
from src.utils.config import load_config, save_config
from src.utils.io import save_json
from src.utils.logger import setup_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LEO synthetic signal dataset.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to dataset YAML config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/leo_dataset.h5",
        help="Output HDF5 path.",
    )
    parser.add_argument(
        "--splits-output",
        type=str,
        default="data/splits/leo_dataset_splits.npz",
        help="Output split npz path.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="Optional summary JSON path. Defaults to output path with .summary.json suffix.",
    )
    parser.add_argument(
        "--resolved-config-output",
        type=str,
        default=None,
        help="Optional YAML path to save resolved config.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default=None,
        choices=[None, "random", "stratified", "domain_holdout"],
        help="Split mode. If omitted, use cfg.split.mode or stratified.",
    )
    parser.add_argument(
        "--test-domain-id",
        type=int,
        nargs="*",
        default=None,
        help="Domain IDs held out for testing when split-mode=domain_holdout.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        help="Override config, e.g. --override signal.num_samples_per_class=100",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger = setup_logger(
        name="leo_drc_dualnet",
        log_file=Path(args.output).with_suffix(".log"),
    )

    cfg = load_config(args.config, overrides=args.override)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resolved_config_output:
        save_config(cfg, args.resolved_config_output)

    arrays, summary = build_dataset_from_config(
        cfg,
        output_h5=str(output_path),
        show_progress=not args.no_progress,
    )

    split_cfg = cfg.get("split", {})
    train_ratio = float(split_cfg.get("train_ratio", 0.7))
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
    test_ratio = float(split_cfg.get("test_ratio", 0.2))

    split_mode = args.split_mode or split_cfg.get("mode", "stratified")

    if split_mode == "random":
        splits = random_split_indices(
            len(arrays["label"]),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    elif split_mode == "stratified":
        splits = stratified_split_indices(
            arrays["label"],
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    elif split_mode == "domain_holdout":
        test_domain_ids = args.test_domain_id
        if not test_domain_ids:
            test_domain_ids = split_cfg.get("test_domain_ids", None)
        if not test_domain_ids:
            raise ValueError(
                "domain_holdout split requires --test-domain-id or split.test_domain_ids"
            )
        splits = domain_holdout_split_indices(
            arrays["domain_id"],
            test_domain_ids=test_domain_ids,
            val_ratio_within_train=val_ratio,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported split mode: {split_mode}")

    splits_output = Path(args.splits_output)
    splits_output.parent.mkdir(parents=True, exist_ok=True)
    save_split_indices(splits, splits_output)

    summary["split_mode"] = split_mode
    summary["splits"] = split_summary(splits)
    summary["output_h5"] = str(output_path)
    summary["splits_output"] = str(splits_output)

    summary_output = (
        Path(args.summary_output)
        if args.summary_output is not None
        else output_path.with_suffix(".summary.json")
    )
    save_json(summary, summary_output)

    logger.info("Dataset generation completed.")
    logger.info(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
