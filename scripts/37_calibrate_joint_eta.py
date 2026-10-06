#!/usr/bin/env python
"""Calibrate a conservative scalar envelope for joint residual CFO and rate.

For a prescribed phase-budget severity eta and allocation u, the injected
errors are

    e_f  = s_f u eta / (2 pi T_f),
    e_mu = s_mu (1-u) eta / (pi T_f^2).

Pure-CFO, pure-rate, mixed allocations, and all nonzero sign combinations are
evaluated on the same independently selected calibration frames.  The output
retains per-frame outcomes so a paired cluster bootstrap can take the
worst-direction envelope without using the final test set.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import pandas as pd
import torch

from src.datasets.feature_dataset import _preprocess_iq
from src.signal.modulation import get_constellation
from src.signal.pulse_shape import rrc_filter


def _load_degradation_helpers() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "33_validate_feature_degradation.py"
    spec = importlib.util.spec_from_file_location("joint_eta_degradation_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-data", required=True)
    p.add_argument("--feature-data", required=True, help="Oracle CFO/rate compensated I/Q HDF5.")
    p.add_argument("--splits", required=True)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--snr-db", type=float, nargs="+", required=True)
    p.add_argument("--samples-per-class-snr", type=int, default=100)
    p.add_argument(
        "--eta-grid",
        type=float,
        nargs="+",
        default=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2),
    )
    p.add_argument(
        "--allocation-grid",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
        help="CFO share u of the total conservative phase budget.",
    )
    p.add_argument("--seed", type=int, default=636363)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output-csv", required=True, help="Per-direction accuracy CSV.")
    p.add_argument("--detail-output", required=True, help="Per-frame paired outcome CSV.")
    p.add_argument("--metadata-output", required=True)
    return p.parse_args()


def _directions(eta: float, allocations: np.ndarray, duration: float) -> list[dict[str, float | str]]:
    if np.isclose(eta, 0.0):
        return [
            {
                "direction_id": "zero",
                "allocation_u": 0.0,
                "cfo_sign": 0.0,
                "rate_sign": 0.0,
                "residual_cfo_hz": 0.0,
                "mu_res_hz_per_s": 0.0,
            }
        ]

    result: list[dict[str, float | str]] = []
    for allocation in allocations:
        cfo_magnitude = float(allocation * eta / (2.0 * np.pi * duration))
        rate_magnitude = float((1.0 - allocation) * eta / (np.pi * duration**2))
        cfo_signs = (0.0,) if np.isclose(cfo_magnitude, 0.0) else (-1.0, 1.0)
        rate_signs = (0.0,) if np.isclose(rate_magnitude, 0.0) else (-1.0, 1.0)
        for cfo_sign in cfo_signs:
            for rate_sign in rate_signs:
                result.append(
                    {
                        "direction_id": (
                            f"u{allocation:.2f}_sf{int(cfo_sign):+d}_sm{int(rate_sign):+d}"
                        ),
                        "allocation_u": float(allocation),
                        "cfo_sign": float(cfo_sign),
                        "rate_sign": float(rate_sign),
                        "residual_cfo_hz": float(cfo_sign * cfo_magnitude),
                        "mu_res_hz_per_s": float(rate_sign * rate_magnitude),
                    }
                )
    return result


def main() -> None:
    args = parse_args()
    helper = _load_degradation_helpers()
    eta_grid = np.unique(np.asarray(args.eta_grid, dtype=np.float64))
    allocations = np.unique(np.asarray(args.allocation_grid, dtype=np.float64))
    if eta_grid.size == 0 or np.any(eta_grid < 0.0):
        raise ValueError("--eta-grid must contain nonnegative values.")
    if not np.any(np.isclose(eta_grid, 0.0)):
        raise ValueError("--eta-grid must include zero.")
    if allocations.size == 0 or np.any((allocations < 0.0) | (allocations > 1.0)):
        raise ValueError("--allocation-grid values must lie in [0,1].")
    if not np.any(np.isclose(allocations, 0.0)) or not np.any(np.isclose(allocations, 1.0)):
        raise ValueError("--allocation-grid must include pure-rate u=0 and pure-CFO u=1.")

    raw_path = Path(args.raw_data)
    feature_path = Path(args.feature_data)
    split_path = Path(args.splits)
    checkpoint_path = Path(args.checkpoint)
    for path in (raw_path, feature_path, split_path, checkpoint_path):
        if not path.exists():
            raise FileNotFoundError(path)

    indices = helper._load_split_indices(split_path, args.split)
    with h5py.File(raw_path, "r") as raw:
        labels_for_selection = np.asarray(raw["label"][indices], dtype=np.int64)
        snr_for_selection = np.asarray(raw["snr_db"][indices], dtype=np.float64)
    selected = helper._select_controlled_indices(
        indices,
        labels_for_selection,
        snr_for_selection,
        int(args.samples_per_class_snr),
        int(args.seed),
        snr_values=np.asarray(args.snr_db, dtype=np.float64),
    )

    with h5py.File(feature_path, "r") as feature, h5py.File(raw_path, "r") as raw:
        iq = np.asarray(feature["iq_comp"][selected], dtype=np.float32)
        base = (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64, copy=False)
        frame_labels = np.asarray(raw["label"][selected], dtype=np.int64)
        frame_snr = np.asarray(raw["snr_db"][selected], dtype=np.float64)
        pilot_indices = np.asarray(raw["pilot_indices"][selected], dtype=np.int64)
        pilot_symbols = np.asarray(raw["pilot_symbols"][selected])
        fs_values = np.asarray(raw["fs"][selected], dtype=np.float64)
        label_names = tuple(name.upper() for name in helper._decode(raw["label_names"][:]))

    fs = float(np.median(fs_values))
    if not np.allclose(fs_values, fs):
        raise ValueError("Joint eta calibration requires one common sample rate.")
    duration = float(base.shape[1] / fs)
    time_s = np.arange(base.shape[1], dtype=np.float64) / fs

    requested = torch.device(args.device)
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_config = dict(payload.get("config", {}))
    model, run_config, model_config = helper._load_fixed_classifier(
        checkpoint_path,
        num_classes=len(label_names),
        hoc_dim=21,
        evm_dim=int(run_config.get("evm_dim", 48)),
        device=device,
    )
    symbol_config = helper._resolve_symbol_config(run_config)
    sps = int(symbol_config["samples_per_symbol"])
    taps = rrc_filter(
        beta=float(symbol_config["rrc_beta"]),
        span=int(symbol_config["rrc_span"]),
        sps=sps,
    )
    symbol_helpers = helper._load_symbol_helpers()
    modulations = tuple(name.upper() for name in symbol_config["modulations"])
    if modulations != label_names:
        raise ValueError(f"Candidate order does not match labels: {modulations} != {label_names}")
    constellations = {name: get_constellation(name).astype(np.complex128) for name in modulations}
    feature_indices = np.asarray(
        run_config.get("evm_feature_indices", np.arange(int(run_config.get("evm_dim", 48)))),
        dtype=np.int64,
    )
    feature_stats = run_config.get("evm_stats") if bool(run_config.get("standardize_evm", False)) else None

    rows: list[dict[str, float | int | str]] = []
    detail_parts: list[pd.DataFrame] = []
    total_directions = sum(len(_directions(float(eta), allocations, duration)) for eta in eta_grid)
    completed = 0
    started = time.perf_counter()

    for eta in eta_grid:
        for direction in _directions(float(eta), allocations, duration):
            predicted = np.empty(len(selected), dtype=np.int64)
            valid = np.empty(len(selected), dtype=bool)
            residual_cfo = float(direction["residual_cfo_hz"])
            residual_rate = float(direction["mu_res_hz_per_s"])
            phase = np.exp(
                1j * (2.0 * np.pi * residual_cfo * time_s + np.pi * residual_rate * time_s**2)
            ).astype(np.complex64)

            for start in range(0, len(selected), int(args.batch_size)):
                stop = min(start + int(args.batch_size), len(selected))
                distorted = base[start:stop] * phase[None, :]
                descriptor_values: list[np.ndarray] = []
                descriptor_valid: list[bool] = []
                for frame, frame_pilots, frame_symbols in zip(
                    distorted,
                    pilot_indices[start:stop],
                    pilot_symbols[start:stop],
                ):
                    vector, is_valid = helper._build_evm_features_for_frame(
                        frame,
                        pilot_indices=frame_pilots,
                        pilot_symbols=frame_symbols,
                        symbol_helpers=symbol_helpers,
                        taps=taps,
                        sps=sps,
                        modulations=modulations,
                        constellations=constellations,
                        config=symbol_config,
                    )
                    descriptor_values.append(vector)
                    descriptor_valid.append(is_valid)
                descriptor_raw = np.vstack(descriptor_values).astype(np.float32)
                descriptor_input = helper._standardize_evm(
                    descriptor_raw[:, feature_indices], feature_stats
                )
                iq_input = np.stack(
                    [
                        _preprocess_iq(
                            np.stack((np.real(frame), np.imag(frame)), axis=0),
                            representation="iq",
                            normalize="zscore",
                        )
                        for frame in distorted
                    ]
                )
                predicted[start:stop] = helper._predict_fixed_classifier(
                    model,
                    model_config,
                    iq_input,
                    descriptor_input,
                    device=device,
                    batch_size=int(args.batch_size),
                )
                valid[start:stop] = np.asarray(descriptor_valid, dtype=bool)

            correct = predicted == frame_labels
            common = {
                "model": "iq_constellation",
                "eta_res": float(eta),
                "direction_id": str(direction["direction_id"]),
                "allocation_u": float(direction["allocation_u"]),
                "cfo_sign": float(direction["cfo_sign"]),
                "rate_sign": float(direction["rate_sign"]),
                "residual_cfo_hz": residual_cfo,
                "mu_res_hz_per_s": residual_rate,
            }
            detail_parts.append(
                pd.DataFrame(
                    {
                        **common,
                        "frame_index": selected,
                        "label": frame_labels,
                        "snr_db": frame_snr,
                        "predicted": predicted,
                        "correct": correct.astype(np.int8),
                    }
                )
            )
            for snr in np.asarray(args.snr_db, dtype=np.float64):
                mask = np.isclose(frame_snr, snr)
                count = int(mask.sum())
                hits = int(correct[mask].sum())
                low, high = helper._wilson_interval(hits, count)
                rows.append(
                    {
                        **common,
                        "snr_db": float(snr),
                        "count": count,
                        "accuracy": float(hits / count),
                        "wilson95_low": float(low),
                        "wilson95_high": float(high),
                        "symbol_valid_fraction": float(valid[mask].mean()),
                    }
                )

            completed += 1
            elapsed = time.perf_counter() - started
            eta_seconds = elapsed / completed * (total_directions - completed)
            print(
                f"[{completed:03d}/{total_directions:03d}] eta={eta:.3f} "
                f"direction={direction['direction_id']} elapsed={elapsed/60:.1f} min "
                f"ETA={eta_seconds/60:.1f} min",
                flush=True,
            )

    result = pd.DataFrame(rows).sort_values(["snr_db", "eta_res", "direction_id"])
    detail = pd.concat(detail_parts, ignore_index=True).sort_values(
        ["snr_db", "label", "frame_index", "eta_res", "direction_id"]
    )
    output_csv = Path(args.output_csv)
    detail_output = Path(args.detail_output)
    metadata_output = Path(args.metadata_output)
    for path in (output_csv, detail_output, metadata_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    detail.to_csv(detail_output, index=False)
    np.save(output_csv.with_name(f"{output_csv.stem}_indices.npy"), selected)
    metadata = {
        "raw_data": str(raw_path),
        "feature_data": str(feature_path),
        "split_file": str(split_path),
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "snr_db": [float(value) for value in args.snr_db],
        "eta_grid": eta_grid.tolist(),
        "allocation_grid": allocations.tolist(),
        "direction_definition": (
            "e_f=s_f*u*eta/(2*pi*T_f); e_mu=s_mu*(1-u)*eta/(pi*T_f^2)"
        ),
        "envelope_definition": "minimum accuracy over sampled allocation/sign directions",
        "frame_duration_s": duration,
        "sample_rate_hz": fs,
        "samples_per_class_snr": int(args.samples_per_class_snr),
        "base_frames": int(len(selected)),
        "direction_evaluations": int(total_directions),
        "seed": int(args.seed),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Wrote {output_csv}")
    print(f"Wrote {detail_output}")


if __name__ == "__main__":
    main()
