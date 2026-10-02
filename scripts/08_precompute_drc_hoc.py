#!/usr/bin/env python
"""Precompute DRC-HOC features from a raw HDF5 dataset.

Example:
    python scripts/08_precompute_drc_hoc.py \
        --raw-data data/processed/leo_sband.h5 \
        --output data/features/leo_sband_drc_hoc.h5 \
        --config configs/experiment/exp_drc_hoc.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
from tqdm import tqdm

from src.drc_hoc.extractor import DRCHOCConfig, DRCHOCExtractor
from src.utils.config import load_config
from src.utils.io import ensure_parent, save_json
from src.utils.logger import setup_logger
from src.utils.math_utils import to_complex
from src.utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Precompute DRC-HOC features.")
    p.add_argument("--raw-data", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--mu-grid-min", type=float, default=None)
    p.add_argument("--mu-grid-max", type=float, default=None)
    p.add_argument("--mu-grid-size", type=int, default=None)
    p.add_argument("--estimator-type", type=str, default=None, choices=["paper_strict_dfrft", "dechirp_fft", "frft"])
    p.add_argument("--alpha-steps", type=int, default=None)
    p.add_argument("--calibration-scale", type=float, default=None)
    p.add_argument("--peak-exclude-bins", type=int, default=None)
    p.add_argument("--candidate-orders", type=int, nargs="*", default=None)
    p.add_argument("--num-subwindows", type=int, default=None)
    p.add_argument("--compensate-fd0", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_config(args, raw_h5_path: str) -> tuple[DRCHOCConfig, int]:
    cfg_dict = {}
    if args.config:
        cfg = load_config(args.config)
        if "drc_hoc" in cfg:
            cfg_dict = cfg.drc_hoc.to_dict() if hasattr(cfg.drc_hoc, "to_dict") else dict(cfg.drc_hoc)
        else:
            cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)

    # Infer sample_rate and observation_time from raw file if not explicitly provided.
    with h5py.File(raw_h5_path, "r") as f:
        N = f["iq"].shape[-1]
        if "fs" in f:
            fs = float(f["fs"][0])
        else:
            fs = float(cfg_dict.get("sample_rate_hz", 1e6))
        T = N / fs

    cfg_dict.setdefault("sample_rate_hz", fs)
    cfg_dict.setdefault("observation_time_s", T)

    if args.mu_grid_min is not None:
        cfg_dict["mu_grid_min"] = args.mu_grid_min
    if args.mu_grid_max is not None:
        cfg_dict["mu_grid_max"] = args.mu_grid_max
    if args.mu_grid_size is not None:
        cfg_dict["mu_grid_size"] = args.mu_grid_size
    if args.estimator_type is not None:
        cfg_dict["estimator_type"] = args.estimator_type
    if args.alpha_steps is not None:
        cfg_dict["alpha_steps"] = args.alpha_steps
    if args.calibration_scale is not None:
        cfg_dict["calibration_scale"] = args.calibration_scale
    if args.peak_exclude_bins is not None:
        cfg_dict["peak_exclude_bins"] = args.peak_exclude_bins
    if args.candidate_orders:
        cfg_dict["candidate_orders"] = tuple(args.candidate_orders)
    if args.num_subwindows is not None:
        cfg_dict["num_subwindows"] = args.num_subwindows
    if args.compensate_fd0:
        cfg_dict["compensate_fd0"] = True

    batch_size = int(cfg_dict.pop("batch_size", args.batch_size or 64))
    return DRCHOCConfig(**cfg_dict), batch_size


def modulation_hint_at(h5_file: h5py.File, index: int) -> str | None:
    if "modulation" not in h5_file:
        return None
    value = h5_file["modulation"][index]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def main():
    args = parse_args()
    seed_everything(args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("leo_drc_dualnet", output.with_suffix(".log"), reset=True)

    drc_cfg, batch_size = build_config(args, args.raw_data)
    extractor = DRCHOCExtractor(drc_cfg)

    logger.info(f"DRC-HOC config: {drc_cfg.to_dict()}")
    logger.info(f"batch_size={batch_size}")

    with h5py.File(args.raw_data, "r") as fr:
        num_samples = len(fr["label"])
        first = extractor.extract(fr["iq"][0], fd0_hat_hz=float(fr["fd0"][0]) if "fd0" in fr else 0.0,
                                  mu_true_hz_per_s=float(fr["mu"][0]) if "mu" in fr else None,
                                  modulation_hint=modulation_hint_at(fr, 0))
        hoc_dim = len(first["h_drc"])

        with h5py.File(output, "w") as fw:
            fw.create_dataset("h_drc", shape=(num_samples, hoc_dim), dtype="float32", compression="gzip")
            fw.create_dataset("mu_hat", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("gamma_hat", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("S_peak", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("V_hoc", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("selected_k", shape=(num_samples,), dtype="int64", compression="gzip")
            fw.create_dataset("estimator_score", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("selected_second_score", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("selected_margin_rel", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("selected_margin_factor", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("selected_boundary_factor", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("fallback_used", shape=(num_samples,), dtype="bool", compression="gzip")
            fw.create_dataset("fallback_reason", shape=(num_samples,), dtype="S32", compression="gzip")
            fw.create_dataset("pre_fallback_mu", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("pre_fallback_selected_k", shape=(num_samples,), dtype="int64", compression="gzip")
            fw.create_dataset("pre_fallback_margin_rel", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("selected_rejected_boundary_low_margin", shape=(num_samples,), dtype="bool", compression="gzip")
            fw.create_dataset("rejected_boundary_low_margin_count", shape=(num_samples,), dtype="int64", compression="gzip")
            fw.create_dataset("selected_hoc_v", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("selected_hoc_score_norm", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("selected_hoc_stability_norm", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("selected_hoc_penalty_norm", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("selected_hoc_rerank_score", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("hoc_rerank_used", shape=(num_samples,), dtype="bool", compression="gzip")
            fw.create_dataset("hoc_rerank_reason", shape=(num_samples,), dtype="S32", compression="gzip")
            fw.create_dataset("hoc_rerank_eligible_count", shape=(num_samples,), dtype="int64", compression="gzip")
            fw.create_dataset("pre_evm_mu", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("selected_evm_score", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("selected_evm_modulation", shape=(num_samples,), dtype="S16", compression="gzip")
            fw.create_dataset("selected_evm_timing", shape=(num_samples,), dtype="int64", compression="gzip")
            fw.create_dataset("selected_evm_phase", shape=(num_samples,), dtype="float32", compression="gzip")
            fw.create_dataset("evm_rerank_used", shape=(num_samples,), dtype="bool", compression="gzip")
            fw.create_dataset("evm_fine_used", shape=(num_samples,), dtype="bool", compression="gzip")
            fw.create_dataset("evm_candidate_count", shape=(num_samples,), dtype="int64", compression="gzip")
            fw.create_dataset("evm_label_aware_used", shape=(num_samples,), dtype="bool", compression="gzip")
            fw.create_dataset("evm_allowed_modulations", shape=(num_samples,), dtype="S64", compression="gzip")
            fw.create_dataset("valid_flag", shape=(num_samples,), dtype="bool", compression="gzip")
            if "mu" in fr:
                fw.create_dataset("gamma_res", shape=(num_samples,), dtype="float32", compression="gzip")
                fw.create_dataset("mu_error", shape=(num_samples,), dtype="float32", compression="gzip")

            # Save feature names as fixed bytes
            feature_names = np.asarray([s.encode("utf-8") for s in first["feature_names"]], dtype="S64")
            fw.create_dataset("feature_names", data=feature_names)
            fw.attrs["config"] = str(drc_cfg.to_dict())

            iterator = range(num_samples)
            if not args.no_progress:
                iterator = tqdm(iterator, desc="Precomputing DRC-HOC")

            for i in iterator:
                try:
                    fd0 = float(fr["fd0"][i]) if "fd0" in fr else 0.0
                    mu_true = float(fr["mu"][i]) if "mu" in fr else None
                    out = extractor.extract(fr["iq"][i], fd0_hat_hz=fd0, mu_true_hz_per_s=mu_true, modulation_hint=modulation_hint_at(fr, i))
                    fw["h_drc"][i] = out["h_drc"]
                    fw["mu_hat"][i] = out["mu_hat"]
                    fw["gamma_hat"][i] = out["gamma_hat"]
                    fw["S_peak"][i] = out["S_peak"]
                    fw["V_hoc"][i] = out["V_hoc"]
                    fw["selected_k"][i] = out["selected_k"]
                    fw["estimator_score"][i] = out["estimator_score"]
                    extra = out.get("estimator_extra", {})
                    fw["selected_second_score"][i] = extra.get("selected_second_score", np.nan)
                    fw["selected_margin_rel"][i] = extra.get("selected_margin_rel", np.nan)
                    fw["selected_margin_factor"][i] = extra.get("selected_margin_factor", np.nan)
                    fw["selected_boundary_factor"][i] = extra.get("selected_boundary_factor", np.nan)
                    fw["fallback_used"][i] = bool(extra.get("fallback_used", False))
                    fw["fallback_reason"][i] = str(extra.get("fallback_reason", "")).encode("utf-8")
                    fw["pre_fallback_mu"][i] = extra.get("pre_fallback_mu", np.nan)
                    fw["pre_fallback_selected_k"][i] = int(extra.get("pre_fallback_selected_k", -1))
                    fw["pre_fallback_margin_rel"][i] = extra.get("pre_fallback_margin_rel", np.nan)
                    fw["selected_rejected_boundary_low_margin"][i] = bool(extra.get("selected_rejected_boundary_low_margin", False))
                    fw["rejected_boundary_low_margin_count"][i] = int(extra.get("rejected_boundary_low_margin_count", 0))
                    fw["selected_hoc_v"][i] = extra.get("selected_hoc_v", np.nan)
                    fw["selected_hoc_score_norm"][i] = extra.get("selected_hoc_score_norm", np.nan)
                    fw["selected_hoc_stability_norm"][i] = extra.get("selected_hoc_stability_norm", np.nan)
                    fw["selected_hoc_penalty_norm"][i] = extra.get("selected_hoc_penalty_norm", np.nan)
                    fw["selected_hoc_rerank_score"][i] = extra.get("selected_hoc_rerank_score", np.nan)
                    fw["hoc_rerank_used"][i] = bool(extra.get("hoc_rerank_used", False))
                    fw["hoc_rerank_reason"][i] = str(extra.get("hoc_rerank_reason", "")).encode("utf-8")
                    fw["hoc_rerank_eligible_count"][i] = int(extra.get("hoc_rerank_eligible_count", 0))
                    fw["pre_evm_mu"][i] = extra.get("pre_evm_mu", np.nan)
                    fw["selected_evm_score"][i] = extra.get("selected_evm_score", np.nan)
                    fw["selected_evm_modulation"][i] = str(extra.get("selected_evm_modulation", "")).encode("utf-8")
                    fw["selected_evm_timing"][i] = int(extra.get("selected_evm_timing", -1))
                    fw["selected_evm_phase"][i] = extra.get("selected_evm_phase", np.nan)
                    fw["evm_rerank_used"][i] = bool(extra.get("evm_rerank_used", False))
                    fw["evm_fine_used"][i] = bool(extra.get("evm_fine_used", False))
                    fw["evm_candidate_count"][i] = int(extra.get("evm_candidate_count", 0))
                    fw["evm_label_aware_used"][i] = bool(extra.get("evm_label_aware_used", False))
                    fw["evm_allowed_modulations"][i] = ",".join(extra.get("evm_allowed_modulations", ())).encode("utf-8")
                    fw["valid_flag"][i] = True
                    if "mu" in fr:
                        fw["gamma_res"][i] = out.get("gamma_res", np.nan)
                        fw["mu_error"][i] = out.get("mu_error", np.nan)
                except Exception as exc:
                    fw["h_drc"][i] = np.zeros(hoc_dim, dtype=np.float32)
                    fw["mu_hat"][i] = np.nan
                    fw["gamma_hat"][i] = np.nan
                    fw["S_peak"][i] = np.nan
                    fw["V_hoc"][i] = np.nan
                    fw["selected_k"][i] = -1
                    fw["estimator_score"][i] = np.nan
                    fw["selected_second_score"][i] = np.nan
                    fw["selected_margin_rel"][i] = np.nan
                    fw["selected_margin_factor"][i] = np.nan
                    fw["selected_boundary_factor"][i] = np.nan
                    fw["fallback_used"][i] = False
                    fw["fallback_reason"][i] = b""
                    fw["pre_fallback_mu"][i] = np.nan
                    fw["pre_fallback_selected_k"][i] = -1
                    fw["pre_fallback_margin_rel"][i] = np.nan
                    fw["selected_rejected_boundary_low_margin"][i] = False
                    fw["rejected_boundary_low_margin_count"][i] = 0
                    fw["selected_hoc_v"][i] = np.nan
                    fw["selected_hoc_score_norm"][i] = np.nan
                    fw["selected_hoc_stability_norm"][i] = np.nan
                    fw["selected_hoc_penalty_norm"][i] = np.nan
                    fw["selected_hoc_rerank_score"][i] = np.nan
                    fw["hoc_rerank_used"][i] = False
                    fw["hoc_rerank_reason"][i] = b""
                    fw["hoc_rerank_eligible_count"][i] = 0
                    fw["pre_evm_mu"][i] = np.nan
                    fw["selected_evm_score"][i] = np.nan
                    fw["selected_evm_modulation"][i] = b""
                    fw["selected_evm_timing"][i] = -1
                    fw["selected_evm_phase"][i] = np.nan
                    fw["evm_rerank_used"][i] = False
                    fw["evm_fine_used"][i] = False
                    fw["evm_candidate_count"][i] = 0
                    fw["evm_label_aware_used"][i] = False
                    fw["evm_allowed_modulations"][i] = b""
                    fw["valid_flag"][i] = False
                    if "mu" in fr:
                        fw["gamma_res"][i] = np.nan
                        fw["mu_error"][i] = np.nan
                    logger.warning(f"Failed at sample {i}: {exc}")

    summary = {"raw_data": args.raw_data, "output": str(output), "num_samples": int(num_samples), "hoc_dim": int(hoc_dim), "config": drc_cfg.to_dict()}
    save_json(summary, output.with_suffix(".summary.json"))
    logger.info(f"Saved DRC-HOC features to {output}")


if __name__ == "__main__":
    main()
