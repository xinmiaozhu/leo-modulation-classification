#!/usr/bin/env python
"""Train the paper receiver or a published baseline model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import torch
from torch.utils.data import DataLoader

from src.datasets.feature_dataset import (
    LEOSignalFeatureDataset,
    compute_evm_stats,
    compute_hoc_stats,
    compute_metadata_stats,
)
from src.datasets.split import load_split_indices
from src.models.drc_dualnet import DRCDualNet
from src.models.losses import DRCTrainingLoss, STARNetTrainingLoss
from src.models.paper_baselines import PAPER_BASELINE_MODELS, build_paper_baseline
from src.training.trainer import TrainConfig, Trainer
from src.utils.config import load_config
from src.utils.io import save_json
from src.utils.logger import setup_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LEO paper receiver or baseline.")
    parser.add_argument("--raw-data", type=str, required=True, help="Raw HDF5 dataset path.")
    parser.add_argument("--feature-data", type=str, default=None, help="DRC-HOC HDF5 feature path.")
    parser.add_argument("--splits", type=str, required=True, help="Split npz path.")
    parser.add_argument(
        "--model",
        type=str,
        default="drc_dualnet",
        choices=[
            "drc_dualnet",
            *sorted(PAPER_BASELINE_MODELS),
        ],
    )
    parser.add_argument("--model-config", type=str, default=None, help="Optional model YAML config.")
    parser.add_argument("--train-config", type=str, default=None, help="Optional train YAML config.")
    parser.add_argument("--symbol-feature-data", type=str, default=None, help="Symbol-level constellation HDF5 feature path.")
    parser.add_argument(
        "--evm-feature-indices",
        type=int,
        nargs="+",
        default=None,
        help="Ordered evm_features columns to use. Defaults to all available EVM features.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoints/run")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume model/optimizer/scheduler state from a last.pt checkpoint.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help=(
            "Initialize model weights from a checkpoint while resetting the "
            "optimizer, scheduler, epoch counter, and early-stopping state."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda-gate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--load-into-memory", action="store_true")
    parser.add_argument(
        "--cache-iq",
        action="store_true",
        help="Cache only the selected I/Q dataset in RAM; useful for random access to compressed HDF5.",
    )
    parser.add_argument(
        "--iq-source",
        type=str,
        default="auto",
        choices=["auto", "raw", "comp"],
        help=(
            "I/Q input for feature datasets: raw uses raw-data iq, comp uses "
            "feature-data iq_comp, auto uses iq_comp when present."
        ),
    )
    parser.add_argument(
        "--iq-representation",
        type=str,
        default="iq",
        choices=["iq", "ap"],
        help="Raw stream representation: I/Q channels or amplitude/phase channels.",
    )
    parser.add_argument(
        "--iq-normalize",
        type=str,
        default="zscore",
        choices=["none", "zscore", "power"],
        help="Per-sample raw-stream normalization.",
    )
    parser.add_argument(
        "--hoc-transform",
        type=str,
        default="signed_log1p",
        choices=["none", "signed_log1p", "log1p_abs"],
        help="Transform applied to h_drc before train-set standardization.",
    )
    parser.add_argument(
        "--no-hoc-standardize",
        action="store_true",
        help="Disable train-set HOC z-score standardization.",
    )
    parser.add_argument(
        "--no-evm-standardize",
        action="store_true",
        help="Disable train-set EVM-feature z-score standardization.",
    )
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


def _read_evm_feature_names(symbol_feature_data: str, feature_indices: list[int]) -> list[str]:
    with h5py.File(symbol_feature_data, "r") as f:
        if "evm_feature_names" not in f:
            return [f"evm_{index}" for index in feature_indices]
        names = f["evm_feature_names"][:]
    return [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in names[feature_indices]
    ]


def _load_train_config(args: argparse.Namespace) -> TrainConfig:
    cfg_dict: dict[str, Any] = {}
    if args.train_config is not None:
        cfg = load_config(args.train_config)
        if "train" in cfg:
            cfg_dict.update(cfg.train.to_dict() if hasattr(cfg.train, "to_dict") else dict(cfg.train))
        else:
            cfg_dict.update(cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg))

        # Some configs place device at root.
        if "device" in cfg:
            cfg_dict["device"] = cfg.device

        if "loss" in cfg and "lambda_gate" in cfg.loss:
            cfg_dict["lambda_gate"] = cfg.loss.lambda_gate

    if args.epochs is not None:
        cfg_dict["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg_dict["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg_dict["learning_rate"] = args.lr
    if args.lambda_gate is not None:
        cfg_dict["lambda_gate"] = args.lambda_gate
    if args.num_workers is not None:
        cfg_dict["num_workers"] = args.num_workers
    if args.device is not None:
        cfg_dict["device"] = args.device

    return TrainConfig(**cfg_dict)


def _load_model_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.model_config is None:
        if args.model == "drc_dualnet":
            return load_config(PROJECT_ROOT / "configs/model/dualnet_iq_evm.yaml").model.to_dict()
        return {}
    cfg = load_config(args.model_config)
    if "model" in cfg:
        return cfg.model.to_dict() if hasattr(cfg.model, "to_dict") else dict(cfg.model)
    return cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)


def _model_input_flags(args: argparse.Namespace) -> tuple[bool, bool, bool]:
    if args.model in PAPER_BASELINE_MODELS:
        return False, False, False
    if args.model != "drc_dualnet":
        return True, False, False
    model_cfg = _load_model_config(args)
    return (
        bool(model_cfg.get("use_metadata", True)),
        bool(model_cfg.get("use_constellation_image", True)),
        bool(model_cfg.get("use_evm_features", True)),
    )


def _model_branch_inputs(args: argparse.Namespace) -> tuple[bool, bool]:
    if args.model == "paper_nasa_hoc_nn":
        return False, True
    if args.model in PAPER_BASELINE_MODELS:
        return True, False
    if args.model == "drc_dualnet":
        model_cfg = _load_model_config(args)
        return (
            bool(model_cfg.get("use_iq_stream", True)),
            bool(model_cfg.get("use_hoc_stream", False)),
        )
    return True, True


def build_model(
    args: argparse.Namespace,
    num_classes: int,
    hoc_dim: int | None,
    evm_dim: int | None = None,
) -> torch.nn.Module:
    model_cfg = _load_model_config(args)
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

    if args.model == "drc_dualnet":
        if evm_dim is None:
            raise ValueError("Model drc_dualnet requires --symbol-feature-data.")
        model_cfg.setdefault("hoc_dim", hoc_dim)
        model_cfg.setdefault("evm_dim", evm_dim)
        model_cfg.setdefault("num_classes", num_classes)
        model_cfg.setdefault("feature_dim", feature_dim)
        model_cfg.setdefault("dropout", dropout)
        return DRCDualNet(**model_cfg)

    raise ValueError(f"Unsupported model: {args.model}")


def build_datasets(args: argparse.Namespace):
    splits = load_split_indices(args.splits)
    train_idx = splits["train"]
    val_idx = splits.get("val", None)

    if args.feature_data is None:
        raise ValueError("--feature-data is required for dual/HOC models.")
    if args.model == "drc_dualnet" and args.symbol_feature_data is None:
        raise ValueError("--symbol-feature-data is required for drc_dualnet.")
    use_metadata, use_constellation, use_evm_features = _model_input_flags(args)
    include_iq, include_hoc = _model_branch_inputs(args)
    metadata_stats = (
        compute_metadata_stats(args.feature_data, indices=train_idx)
        if use_metadata
        else None
    )
    standardize_hoc = not args.no_hoc_standardize
    hoc_stats = (
        compute_hoc_stats(args.feature_data, indices=train_idx, transform=args.hoc_transform)
        if standardize_hoc
        else None
    )
    standardize_evm = use_evm_features and not args.no_evm_standardize
    evm_stats = (
        compute_evm_stats(
            args.symbol_feature_data,
            indices=train_idx,
            feature_indices=args.evm_feature_indices,
        )
        if standardize_evm
        else None
    )
    train_set = LEOSignalFeatureDataset(
        args.raw_data,
        args.feature_data,
        symbol_h5_path=args.symbol_feature_data,
        indices=train_idx,
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
        return_diagnostics=False,
    )
    val_set = (
        LEOSignalFeatureDataset(
            args.raw_data,
            args.feature_data,
            symbol_h5_path=args.symbol_feature_data,
            indices=val_idx,
            load_into_memory=args.load_into_memory,
            cache_iq=False,
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
            return_diagnostics=False,
        )
        if val_idx is not None and len(val_idx) > 0
        else None
    )
    if val_set is not None and args.cache_iq:
        val_set.iq_cache = train_set.iq_cache

    return train_set, val_set, metadata_stats, hoc_stats, evm_stats


def main() -> None:
    args = parse_args()
    if args.resume is not None and args.init_checkpoint is not None:
        raise SystemExit("--resume and --init-checkpoint are mutually exclusive.")
    seed_everything(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("leo_drc_dualnet", out_dir / "train.log", reset=True)

    train_cfg = _load_train_config(args)
    num_classes = _read_num_classes(args.raw_data)
    hoc_dim = _read_hoc_dim(args.feature_data) if args.feature_data is not None else None
    if args.symbol_feature_data is not None:
        args.evm_feature_indices = _resolve_evm_feature_indices(
            args.symbol_feature_data,
            args.evm_feature_indices,
        )
    evm_dim = _read_evm_dim(args.symbol_feature_data, args.evm_feature_indices) if args.symbol_feature_data is not None else None

    logger.info(f"num_classes={num_classes}, hoc_dim={hoc_dim}, evm_dim={evm_dim}")
    logger.info(f"train_cfg={train_cfg.to_dict()}")

    train_set, val_set, metadata_stats, hoc_stats, evm_stats = build_datasets(args)

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        persistent_workers=train_cfg.num_workers > 0,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=train_cfg.pin_memory,
            persistent_workers=train_cfg.num_workers > 0,
            drop_last=False,
        )
        if val_set is not None
        else None
    )

    model_config_for_ckpt = _load_model_config(args)
    model = build_model(args, num_classes=num_classes, hoc_dim=hoc_dim, evm_dim=evm_dim)
    if args.init_checkpoint is not None:
        init_state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model_state = init_state.get("model_state_dict", init_state)
        model.load_state_dict(model_state, strict=True)
        logger.info(
            "Initialized model weights from %s with fresh optimizer and scheduler state.",
            args.init_checkpoint,
        )
    criterion = (
        STARNetTrainingLoss(
            reconstruction_weight=float(model_config_for_ckpt.get("reconstruction_weight", 0.6))
        )
        if args.model == "paper_starnet"
        else DRCTrainingLoss(lambda_gate=0.0)
    )

    config_for_ckpt = {
        "model_name": args.model,
        "num_classes": num_classes,
        "hoc_dim": hoc_dim,
        "evm_dim": evm_dim,
        "evm_feature_indices": args.evm_feature_indices,
        "evm_feature_names": (
            _read_evm_feature_names(args.symbol_feature_data, args.evm_feature_indices)
            if args.symbol_feature_data is not None and args.evm_feature_indices is not None
            else None
        ),
        "train_config": train_cfg.to_dict(),
        "model_config": model_config_for_ckpt,
        "metadata_stats": metadata_stats,
        "raw_data": args.raw_data,
        "feature_data": args.feature_data,
        "symbol_feature_data": args.symbol_feature_data,
        "splits": args.splits,
        "iq_source": args.iq_source,
        "iq_representation": args.iq_representation,
        "iq_normalize": args.iq_normalize,
        "cache_iq": args.cache_iq,
        "hoc_transform": args.hoc_transform,
        "standardize_hoc": not args.no_hoc_standardize,
        "hoc_stats": hoc_stats,
        "standardize_evm": not args.no_evm_standardize,
        "evm_stats": evm_stats,
        "init_checkpoint": args.init_checkpoint,
    }
    save_json(config_for_ckpt, out_dir / "run_config.json")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=train_cfg,
        output_dir=out_dir,
        criterion=criterion,
        config_for_checkpoint=config_for_ckpt,
    )
    result = trainer.fit(resume_from=args.resume)
    save_json(result, out_dir / "train_result.json")

    logger.info(f"Training done: {result}")


if __name__ == "__main__":
    main()
