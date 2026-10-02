#!/usr/bin/env python
"""Screen a label-independent pilot DFRFT--coherent hybrid estimator.

The formal Protocol-B coherent-grid estimates are read from the released
feature file.  The hybrid is evaluated on the same stratified validation
frames.  Promotion requires tail noninferiority and either a material CPU
speedup or a material joint-tail improvement; failing the gate leaves the
formal receiver unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import pandas as pd

from src.drc_hoc.hybrid_dfrft_estimator import HybridDFRFTConfig, HybridDFRFTPilotEstimator
from src.drc_hoc.pilot_estimator import PilotMuEstimator, PilotMuEstimatorConfig


TRANSITION_SNRS = (-7.0, -4.0, -1.0, 2.0, 5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", default="data/processed/leo_7mods_joint_practical.h5")
    parser.add_argument("--splits", default="data/splits/leo_7mods_joint_practical_splits.npz")
    parser.add_argument(
        "--baseline-estimates",
        default="data/features/leo_7mods_joint_practical_pilot_joint.h5",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--samples-per-class-snr", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--latency-frames", type=int, default=77)
    parser.add_argument("--coarse-mu-step", type=float, default=160.0)
    parser.add_argument("--fft-size", type=int, default=2048)
    parser.add_argument("--fine-mu-radius", type=float, default=200.0)
    parser.add_argument("--fine-mu-step", type=float, default=5.0)
    parser.add_argument("--fine-cfo-radius", type=float, default=30.0)
    parser.add_argument("--fine-cfo-step", type=float, default=1.0)
    parser.add_argument("--tail-noninferiority-ratio", type=float, default=1.10)
    parser.add_argument("--speedup-threshold", type=float, default=1.25)
    parser.add_argument("--tail-improvement-ratio", type=float, default=0.90)
    parser.add_argument(
        "--output-dir",
        default="outputs/results/estimator_screening/hybrid_dfrft_protocol_b",
    )
    return parser.parse_args()


def _stratified_indices(
    candidates: np.ndarray,
    labels: np.ndarray,
    snr: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in np.unique(labels[candidates]):
        for level in np.unique(snr[candidates]):
            cell = candidates[(labels[candidates] == label) & np.isclose(snr[candidates], level)]
            if cell.size < count:
                raise ValueError(f"Cell label={label}, SNR={level} has only {cell.size} frames.")
            selected.extend(rng.choice(cell, size=count, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _metrics(table: pd.DataFrame, method: str) -> dict[str, float]:
    subset = table[table["method"] == method]
    mu = subset["abs_mu_error_hz_per_s"].to_numpy(float)
    cfo = subset["abs_cfo_error_hz"].to_numpy(float)
    eta = subset["eta"].to_numpy(float)
    return {
        "valid_fraction": float(subset["valid"].mean()),
        "mu_mae_hz_per_s": float(np.mean(mu)),
        "mu_rmse_hz_per_s": float(np.sqrt(np.mean(mu**2))),
        "mu_p90_hz_per_s": float(np.quantile(mu, 0.90)),
        "mu_p95_hz_per_s": float(np.quantile(mu, 0.95)),
        "cfo_mae_hz": float(np.mean(cfo)),
        "cfo_rmse_hz": float(np.sqrt(np.mean(cfo**2))),
        "cfo_p90_hz": float(np.quantile(cfo, 0.90)),
        "cfo_p95_hz": float(np.quantile(cfo, 0.95)),
        "eta_p90": float(np.quantile(eta, 0.90)),
        "eta_p95": float(np.quantile(eta, 0.95)),
        "outlier_gt_1000_fraction": float(np.mean(mu > 1000.0)),
    }


def _timing_ms(
    estimator,
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    if not frames:
        return np.asarray([], dtype=np.float64)
    estimator.estimate(*frames[0])
    elapsed = []
    for frame in frames:
        start = time.perf_counter()
        estimator.estimate(*frame)
        elapsed.append(1000.0 * (time.perf_counter() - start))
    return np.asarray(elapsed, dtype=np.float64)


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pilot_config = PilotMuEstimatorConfig(
        sample_rate_hz=200_000.0,
        samples_per_symbol=8,
        rrc_beta=0.35,
        rrc_span=8,
        timing_phases=(0,),
        mu_min=-8160.0,
        mu_max=-180.0,
        coarse_step_hz_per_s=40.0,
        fine_radius_hz_per_s=120.0,
        fine_step_hz_per_s=5.0,
        fd0_min_hz=-550.0,
        fd0_max_hz=550.0,
        fd0_step_hz=25.0,
        fine_fd0_radius_hz=30.0,
        fine_fd0_step_hz=1.0,
        pilot_weighting="coherent",
    )
    hybrid_config = HybridDFRFTConfig(
        coarse_mu_step_hz_per_s=args.coarse_mu_step,
        fft_size=args.fft_size,
        fine_mu_radius_hz_per_s=args.fine_mu_radius,
        fine_mu_step_hz_per_s=args.fine_mu_step,
        fine_fd0_radius_hz=args.fine_cfo_radius,
        fine_fd0_step_hz=args.fine_cfo_step,
    )
    baseline_estimator = PilotMuEstimator(pilot_config)
    hybrid_estimator = HybridDFRFTPilotEstimator(pilot_config, hybrid_config)

    split_data = np.load(ROOT / args.splits)
    candidate_indices = np.asarray(split_data[args.split], dtype=np.int64)
    with h5py.File(ROOT / args.raw_data, "r") as raw, h5py.File(
        ROOT / args.baseline_estimates, "r"
    ) as baseline:
        labels = np.asarray(raw["label"][:], dtype=np.int64)
        snr = np.asarray(raw["snr_db"][:], dtype=np.float64)
        selected = _stratified_indices(
            candidate_indices,
            labels,
            snr,
            args.samples_per_class_snr,
            args.seed,
        )
        rows: list[dict[str, float | int | str | bool]] = []
        latency_payload: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        latency_set = set(selected[: max(0, min(args.latency_frames, selected.size))].tolist())
        for position, index in enumerate(selected):
            iq = np.asarray(raw["iq"][index], dtype=np.float32)
            pilot_indices = np.asarray(raw["pilot_indices"][index], dtype=np.int64)
            pilot_symbols = np.asarray(raw["pilot_symbols"][index], dtype=np.complex64)
            if int(index) in latency_set:
                latency_payload.append((iq, pilot_indices, pilot_symbols))
            truth_mu = float(raw["mu"][index])
            truth_cfo = float(raw["fd0"][index])
            duration = float(raw["T"][index])

            formal_mu = float(baseline["pilot_mu_hat"][index])
            formal_cfo = float(baseline["pilot_fd0_hat"][index])
            hybrid = hybrid_estimator.estimate(iq, pilot_indices, pilot_symbols)
            estimates = {
                "coherent_grid": (formal_mu, formal_cfo, True, np.nan, np.nan),
                "hybrid_dfrft": (
                    hybrid.estimate.mu_hat_hz_per_s,
                    hybrid.estimate.fd0_hat_hz,
                    hybrid.estimate.valid,
                    hybrid.coarse_mu_hz_per_s,
                    hybrid.coarse_fd0_hz,
                ),
            }
            for method, (mu_hat, cfo_hat, valid, coarse_mu, coarse_cfo) in estimates.items():
                mu_error = truth_mu - float(mu_hat)
                cfo_error = truth_cfo - float(cfo_hat)
                rows.append(
                    {
                        "index": int(index),
                        "position": int(position),
                        "method": method,
                        "label": int(labels[index]),
                        "snr_db": float(snr[index]),
                        "mu_true_hz_per_s": truth_mu,
                        "mu_hat_hz_per_s": float(mu_hat),
                        "cfo_true_hz": truth_cfo,
                        "cfo_hat_hz": float(cfo_hat),
                        "abs_mu_error_hz_per_s": abs(mu_error),
                        "abs_cfo_error_hz": abs(cfo_error),
                        "eta": 2.0 * np.pi * abs(cfo_error) * duration
                        + np.pi * abs(mu_error) * duration**2,
                        "valid": bool(valid),
                        "coarse_mu_hz_per_s": float(coarse_mu),
                        "coarse_cfo_hz": float(coarse_cfo),
                    }
                )

    frame_table = pd.DataFrame(rows)
    frame_table.to_csv(output_dir / "frame_results.csv", index=False)
    aggregate = {method: _metrics(frame_table, method) for method in ("coherent_grid", "hybrid_dfrft")}
    by_snr = (
        frame_table.groupby(["method", "snr_db"], sort=True)
        .agg(
            count=("eta", "size"),
            mu_mae_hz_per_s=("abs_mu_error_hz_per_s", "mean"),
            cfo_mae_hz=("abs_cfo_error_hz", "mean"),
            eta_p90=("eta", lambda x: float(np.quantile(x, 0.90))),
            eta_p95=("eta", lambda x: float(np.quantile(x, 0.95))),
        )
        .reset_index()
    )
    by_snr.to_csv(output_dir / "metrics_by_snr.csv", index=False)

    baseline_latency = _timing_ms(baseline_estimator, latency_payload)
    hybrid_latency = _timing_ms(hybrid_estimator, latency_payload)
    latency = {
        "frames": int(len(latency_payload)),
        "coherent_grid_median_ms": float(np.median(baseline_latency)),
        "hybrid_dfrft_median_ms": float(np.median(hybrid_latency)),
        "speedup": float(np.median(baseline_latency) / np.median(hybrid_latency)),
    }

    base = aggregate["coherent_grid"]
    candidate = aggregate["hybrid_dfrft"]
    ratios = {
        "mu_p95": candidate["mu_p95_hz_per_s"] / base["mu_p95_hz_per_s"],
        "cfo_p95": candidate["cfo_p95_hz"] / base["cfo_p95_hz"],
        "eta_p95": candidate["eta_p95"] / base["eta_p95"],
    }
    transition = by_snr[by_snr["snr_db"].isin(TRANSITION_SNRS)].pivot(
        index="snr_db", columns="method", values="eta_p95"
    )
    maximum_bin_ratio = float(
        np.max(transition["hybrid_dfrft"] / transition["coherent_grid"])
    )
    noninferior = bool(
        max(*ratios.values(), maximum_bin_ratio) <= args.tail_noninferiority_ratio
        and candidate["outlier_gt_1000_fraction"]
        <= base["outlier_gt_1000_fraction"] + 0.01
    )
    material_benefit = bool(
        latency["speedup"] >= args.speedup_threshold
        or ratios["eta_p95"] <= args.tail_improvement_ratio
    )
    promote = bool(noninferior and material_benefit)
    summary = {
        "screening_only": True,
        "formal_receiver_changed": False,
        "split": args.split,
        "num_frames": int(len(selected)),
        "samples_per_class_snr": int(args.samples_per_class_snr),
        "selection_seed": int(args.seed),
        "pilot_config": pilot_config.to_dict(),
        "hybrid_config": hybrid_config.to_dict(),
        "aggregate": aggregate,
        "latency": latency,
        "promotion_gate": {
            "tail_noninferiority_ratio": float(args.tail_noninferiority_ratio),
            "speedup_threshold": float(args.speedup_threshold),
            "tail_improvement_ratio": float(args.tail_improvement_ratio),
            "observed_tail_ratios": ratios,
            "maximum_transition_snr_eta_p95_ratio": maximum_bin_ratio,
            "tail_noninferior": noninferior,
            "material_benefit": material_benefit,
            "promote": promote,
        },
        "information_policy": {
            "known_pilots_only": True,
            "uses_true_modulation_label": False,
            "uses_true_snr": False,
            "uses_true_cfo_or_rate": False,
            "uses_hoc_or_constellation_reranking": False,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("PROMOTE" if promote else "DO NOT PROMOTE")


if __name__ == "__main__":
    main()
