#!/usr/bin/env python
"""Evaluate the paper receiver or a published baseline model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.feature_dataset import (
    LEOSignalFeatureDataset,
    compute_evm_stats,
    compute_hoc_stats,
    compute_metadata_stats,
)
from src.datasets.split import load_split_indices
from src.models.drc_triplenet import DRCTripleNet
from src.models.paper_baselines import PAPER_BASELINE_MODELS, build_paper_baseline
from src.training.evaluator import Evaluator
from src.training.metrics import (
    grouped_accuracy,
    snr_gamma_grid_accuracy,
    summarize_eval_dataframe,
)
from src.utils.config import load_config
from src.utils.io import ensure_parent, save_json
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LEO paper receiver or baseline.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--raw-data", type=str, required=True)
    parser.add_argument("--feature-data", type=str, default=None)
    parser.add_argument("--splits", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--model",
        type=str,
        default="drc_triplenet",
        choices=[
            "drc_triplenet",
            *sorted(PAPER_BASELINE_MODELS),
        ],
    )
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--symbol-feature-data", type=str, default=None)
    parser.add_argument(
        "--evm-feature-indices",
        type=int,
        nargs="+",
        default=None,
        help="Override the checkpoint EVM column subset. Defaults to the checkpoint subset.",
    )
    parser.add_argument("--output", type=str, default="outputs/results/eval.csv")
    parser.add_argument("--summary-output", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-into-memory", action="store_true")
    parser.add_argument(
        "--cache-iq",
        action="store_true",
        help="Cache only the selected I/Q dataset in RAM.",
    )
    parser.add_argument(
        "--iq-source",
        type=str,
        default=None,
        choices=["auto", "raw", "comp"],
        help=(
            "I/Q input for feature datasets. Defaults to checkpoint iq_source "
            "when available, otherwise auto."
        ),
    )
    parser.add_argument(
        "--iq-representation",
        type=str,
        default=None,
        choices=["iq", "ap"],
        help="Raw stream representation. Defaults to checkpoint setting when available.",
    )
    parser.add_argument(
        "--iq-normalize",
        type=str,
        default=None,
        choices=["none", "zscore", "power"],
        help="Per-sample raw-stream normalization. Defaults to checkpoint setting when available.",
    )
    parser.add_argument(
        "--hoc-transform",
        type=str,
        default=None,
        choices=["none", "signed_log1p", "log1p_abs"],
        help="HOC transform. Defaults to checkpoint setting when available.",
    )
    parser.add_argument(
        "--no-hoc-standardize",
        action="store_true",
        help="Disable HOC z-score standardization even if checkpoint has HOC stats.",
    )
    parser.add_argument(
        "--no-evm-standardize",
        action="store_true",
        help="Disable EVM-feature z-score standardization even if checkpoint has EVM stats.",
    )
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading.")
    parser.add_argument("--no-aux", action="store_true", help="Disable aux outputs such as gate diagnostics.")
    return parser.parse_args()


def _read_num_classes(raw_data: str) -> int:
    with h5py.File(raw_data, "r") as f:
        if "label_names" in f:
            return int(len(f["label_names"]))
        labels = f["label"][()]
        return int(labels.max() + 1)


def _read_hoc_dim(feature_data: str) -> int:
    with h5py.File(feature_data, "r") as f:
        return int(f["h_drc"].shape[1])


def _resolve_evm_feature_indices(
    symbol_feature_data: str,
    feature_indices: list[int] | None,
) -> list[int]:
    with h5py.File(symbol_feature_data, "r") as f:
        feature_dim = int(f["evm_features"].shape[1])
    if feature_indices is None:
        return list(range(feature_dim))
    selected = [int(index) for index in feature_indices]
    if not selected:
        raise ValueError("--evm-feature-indices must not be empty.")
    if len(set(selected)) != len(selected):
        raise ValueError("--evm-feature-indices must not contain duplicates.")
    if min(selected) < 0 or max(selected) >= feature_dim:
        raise ValueError(f"--evm-feature-indices must lie in [0, {feature_dim - 1}].")
    return selected


def _read_evm_dim(symbol_feature_data: str, feature_indices: list[int] | None = None) -> int:
    return len(_resolve_evm_feature_indices(symbol_feature_data, feature_indices))


def _load_model_config(args: argparse.Namespace, checkpoint_config: dict[str, Any] | None = None) -> dict[str, Any]:
    if args.model_config is not None:
        cfg = load_config(args.model_config)
        if "model" in cfg:
            return cfg.model.to_dict() if hasattr(cfg.model, "to_dict") else dict(cfg.model)
        return cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)

    # Checkpoint stores original run_config inside state["config"].
    if checkpoint_config is not None and "model_config" in checkpoint_config:
        return checkpoint_config["model_config"]

    if args.model == "drc_triplenet":
        return load_config(PROJECT_ROOT / "configs/model/triplenet_iq_evm.yaml").model.to_dict()
    return {}


def _model_input_flags(
    args: argparse.Namespace,
    checkpoint_config: dict[str, Any] | None = None,
) -> tuple[bool, bool, bool]:
    if args.model in PAPER_BASELINE_MODELS:
        return False, False, False
    if args.model != "drc_triplenet":
        return True, False, False
    model_cfg = _load_model_config(args, checkpoint_config)
    return (
        bool(model_cfg.get("use_metadata", True)),
        bool(model_cfg.get("use_constellation_image", True)),
        bool(model_cfg.get("use_evm_features", True)),
    )


def _model_branch_inputs(
    args: argparse.Namespace,
    checkpoint_config: dict[str, Any] | None = None,
) -> tuple[bool, bool]:
    if args.model == "paper_nasa_hoc_nn":
        return False, True
    if args.model in PAPER_BASELINE_MODELS:
        return True, False
    if args.model == "drc_triplenet":
        model_cfg = _load_model_config(args, checkpoint_config)
        return (
            bool(model_cfg.get("use_iq_stream", True)),
            bool(model_cfg.get("use_hoc_stream", True)),
        )
    return True, True


def build_model(
    args: argparse.Namespace,
    num_classes: int,
    hoc_dim: int | None,
    evm_dim: int | None = None,
    checkpoint_config=None,
):
    model_cfg = _load_model_config(args, checkpoint_config)
    feature_dim = int(model_cfg.pop("feature_dim", 128))
    dropout = float(model_cfg.pop("dropout", 0.2))


    if args.model in PAPER_BASELINE_MODELS:
        model_cfg.setdefault("feature_dim", feature_dim)
        model_cfg.setdefault("dropout", dropout)
        return build_paper_baseline(
            args.model,
            num_classes=num_classes,
            hoc_dim=hoc_dim,
            **model_cfg,
        )

    if hoc_dim is None:
        raise ValueError(f"Model {args.model} requires --feature-data.")

    if args.model == "drc_triplenet":
        if evm_dim is None:
            raise ValueError("Model drc_triplenet requires --symbol-feature-data.")
        model_cfg.setdefault("hoc_dim", hoc_dim)
        model_cfg.setdefault("evm_dim", evm_dim)
        model_cfg.setdefault("num_classes", num_classes)
        model_cfg.setdefault("feature_dim", feature_dim)
        model_cfg.setdefault("dropout", dropout)
        return DRCTripleNet(**model_cfg)

    raise ValueError(f"Unsupported model: {args.model}")


def build_dataset(
    args: argparse.Namespace,
    metadata_stats=None,
    hoc_stats=None,
    evm_stats=None,
    checkpoint_config: dict[str, Any] | None = None,
):
    splits = load_split_indices(args.splits)
    indices = splits[args.split]


    if args.feature_data is None:
        raise ValueError("--feature-data is required for dual/HOC models.")
    if args.model == "drc_triplenet" and args.symbol_feature_data is None:
        raise ValueError("--symbol-feature-data is required for drc_triplenet.")

    use_metadata, use_constellation, use_evm_features = _model_input_flags(
        args,
        checkpoint_config,
    )
    include_iq, include_hoc = _model_branch_inputs(args, checkpoint_config)
    if use_metadata and metadata_stats is None:
        # Fall back: compute on train split to avoid test leakage.
        metadata_stats = compute_metadata_stats(args.feature_data, indices=splits["train"])
    standardize_hoc = bool(args.standardize_hoc)
    if standardize_hoc and hoc_stats is None:
        hoc_stats = compute_hoc_stats(
            args.feature_data,
            indices=splits["train"],
            transform=args.hoc_transform,
        )
    standardize_evm = use_evm_features and bool(args.standardize_evm)
    if standardize_evm and evm_stats is None:
        evm_stats = compute_evm_stats(
            args.symbol_feature_data,
            indices=splits["train"],
            feature_indices=args.evm_feature_indices,
        )

    return LEOSignalFeatureDataset(
        args.raw_data,
        args.feature_data,
        symbol_h5_path=args.symbol_feature_data,
        indices=indices,
        load_into_memory=args.load_into_memory,
        cache_iq=args.cache_iq,
        standardize_metadata=use_metadata,
        metadata_stats=metadata_stats,
        iq_source=args.iq_source,
        iq_representation=args.iq_representation,
        iq_normalize=args.iq_normalize,
        standardize_hoc=standardize_hoc,
        hoc_transform=args.hoc_transform,
        hoc_stats=hoc_stats,
        standardize_evm=standardize_evm,
        evm_stats=evm_stats,
        evm_feature_indices=args.evm_feature_indices,
        include_metadata=use_metadata,
        include_iq=include_iq,
        include_hoc=include_hoc,
        include_constellation=use_constellation,
        include_evm_features=use_evm_features,
        return_diagnostics=True,
    )


def main() -> None:
    args = parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("leo_drc_dualnet", output.with_suffix(".log"), reset=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_config = checkpoint.get("config", {})
    metadata_stats = ckpt_config.get("metadata_stats", None)
    if args.iq_source is None:
        args.iq_source = ckpt_config.get("iq_source", "auto")
    if args.iq_representation is None:
        args.iq_representation = ckpt_config.get("iq_representation", "iq")
    if args.iq_normalize is None:
        args.iq_normalize = ckpt_config.get("iq_normalize", "none")
    if args.hoc_transform is None:
        args.hoc_transform = ckpt_config.get("hoc_transform", "none")
    if args.symbol_feature_data is None:
        args.symbol_feature_data = ckpt_config.get("symbol_feature_data", None)
    if args.evm_feature_indices is None:
        args.evm_feature_indices = ckpt_config.get("evm_feature_indices", None)
    if args.symbol_feature_data is not None:
        args.evm_feature_indices = _resolve_evm_feature_indices(
            args.symbol_feature_data,
            args.evm_feature_indices,
        )
    args.standardize_hoc = bool(ckpt_config.get("standardize_hoc", False)) and not args.no_hoc_standardize
    hoc_stats = ckpt_config.get("hoc_stats", None) if args.standardize_hoc else None
    args.standardize_evm = bool(ckpt_config.get("standardize_evm", False)) and not args.no_evm_standardize
    evm_stats = ckpt_config.get("evm_stats", None) if args.standardize_evm else None

    num_classes = _read_num_classes(args.raw_data)
    hoc_dim = _read_hoc_dim(args.feature_data) if args.feature_data is not None else None
    evm_dim = _read_evm_dim(args.symbol_feature_data, args.evm_feature_indices) if args.symbol_feature_data is not None else None

    model = build_model(args, num_classes=num_classes, hoc_dim=hoc_dim, evm_dim=evm_dim, checkpoint_config=ckpt_config)

    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=args.strict)

    dataset = build_dataset(
        args,
        metadata_stats=metadata_stats,
        hoc_stats=hoc_stats,
        evm_stats=evm_stats,
        checkpoint_config=ckpt_config,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    evaluator = Evaluator(model, device=args.device)
    df, summary = evaluator.evaluate_and_summarize(loader, return_aux=not args.no_aux)

    df.to_csv(output, index=False)
    logger.info(f"Saved per-sample evaluation to {output}")

    summary["checkpoint"] = args.checkpoint
    summary["split"] = args.split
    summary["model"] = args.model
    summary["iq_source"] = args.iq_source
    summary["iq_representation"] = args.iq_representation
    summary["iq_normalize"] = args.iq_normalize
    summary["hoc_transform"] = args.hoc_transform
    summary["standardize_hoc"] = args.standardize_hoc
    summary["symbol_feature_data"] = args.symbol_feature_data
    summary["standardize_evm"] = args.standardize_evm
    summary["evm_feature_indices"] = args.evm_feature_indices

    # Grouped analyses
    grouped = {}
    for col in ["snr_db", "domain_id", "selected_k"]:
        if col in df.columns:
            grouped[col] = grouped_accuracy(df, col).to_dict(orient="records")

    # Gamma is continuous, so use quantile bins if enough samples exist.
    if "gamma" in df.columns and len(df) >= 10:
        try:
            gamma_bins = np.unique(np.quantile(df["gamma"].to_numpy(), np.linspace(0, 1, 6)))
            if len(gamma_bins) >= 3:
                work = df.copy()
                work["gamma_bin_id"] = np.digitize(work["gamma"], gamma_bins[1:-1], right=False)
                grouped["gamma_bin"] = grouped_accuracy(work, "gamma_bin_id").to_dict(orient="records")
        except Exception as exc:
            logger.warning(f"Gamma-bin grouped analysis failed: {exc}")

    summary["grouped_accuracy"] = grouped

    summary_output = Path(args.summary_output) if args.summary_output else output.with_suffix(".summary.json")
    save_json(summary, summary_output)

    logger.info(f"Summary: {summary}")
    logger.info(f"Saved summary to {summary_output}")


if __name__ == "__main__":
    main()
