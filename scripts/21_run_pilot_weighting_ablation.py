#!/usr/bin/env python
"""Compare coherent, phase-only, soft, and amplitude-gated pilot searches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.drc_hoc.pilot_estimator import PilotMuEstimator, PilotMuEstimatorConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-data", required=True)
    p.add_argument(
        "--splits",
        default=None,
        help="Optional NPZ split file used to restrict the screening population.",
    )
    p.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default=None,
        help="Split to screen when --splits is supplied.",
    )
    p.add_argument("--output", default="outputs/results/pilot_weighting_ablation.csv")
    p.add_argument("--sample-rate-hz", type=float, default=None)
    p.add_argument("--samples-per-symbol", type=int, default=8)
    p.add_argument("--rrc-beta", type=float, default=0.35)
    p.add_argument("--rrc-span", type=int, default=8)
    p.add_argument("--mu-min", type=float, default=-8160.0)
    p.add_argument("--mu-max", type=float, default=-180.0)
    p.add_argument("--coarse-step", type=float, default=40.0)
    p.add_argument("--fine-radius", type=float, default=120.0)
    p.add_argument("--fine-step", type=float, default=5.0)
    p.add_argument("--soft-weight-scale", type=float, default=1.0)
    p.add_argument(
        "--amplitude-threshold-rel",
        type=float,
        default=0.35,
        help="Median-relative amplitude floor in the thresholded phase-only mode.",
    )
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--samples-per-class-snr", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def _summary_rows(mode: str, snr: np.ndarray, error: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    groups = [("overall", np.ones(error.size, dtype=bool))]
    groups += [(f"{value:g} dB", np.isclose(snr, value)) for value in np.unique(snr)]
    for group, mask in groups:
        absolute = np.abs(error[mask])
        rows.append({
            "pilot_weighting": mode,
            "snr_group": group,
            "count": int(absolute.size),
            "mae_hz_per_s": float(np.mean(absolute)),
            "rmse_hz_per_s": float(np.sqrt(np.mean(error[mask] ** 2))),
            "p95_abs_hz_per_s": float(np.quantile(absolute, 0.95)),
            "outlier_gt_200": float(np.mean(absolute > 200.0)),
            "outlier_gt_500": float(np.mean(absolute > 500.0)),
        })
    return rows


def main() -> None:
    args = parse_args()
    if (args.splits is None) != (args.split is None):
        raise SystemExit("--splits and --split must be supplied together.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.raw_data, "r") as raw:
        fs = float(args.sample_rate_hz) if args.sample_rate_hz else float(raw["fs"][0])
        total = int(raw["iq"].shape[0])
        all_snr = np.asarray(raw["snr_db"][:], dtype=np.float64)
        all_labels = np.asarray(raw["label"][:], dtype=np.int64)
        eligible = np.arange(total, dtype=np.int64)
        if args.splits is not None:
            with np.load(args.splits) as split_data:
                if args.split not in split_data:
                    raise SystemExit(f"Split {args.split!r} is absent from {args.splits}.")
                eligible = np.asarray(split_data[args.split], dtype=np.int64)
            if eligible.size == 0:
                raise SystemExit(f"Split {args.split!r} is empty in {args.splits}.")
            if np.any((eligible < 0) | (eligible >= total)):
                raise SystemExit("Split indices fall outside the raw-data frame range.")
        rng = np.random.default_rng(args.seed)
        selected: list[int] = []
        eligible_labels = all_labels[eligible]
        eligible_snr = all_snr[eligible]
        for label in np.unique(eligible_labels):
            for snr_value in np.unique(eligible_snr):
                local = np.flatnonzero(
                    (eligible_labels == label) & np.isclose(eligible_snr, snr_value)
                )
                candidates = eligible[local]
                count = min(int(args.samples_per_class_snr), candidates.size)
                selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
        indices = np.asarray(sorted(selected), dtype=np.int64)
        if args.max_frames is not None:
            indices = indices[: min(indices.size, int(args.max_frames))]
        n = int(indices.size)
        snr = all_snr[indices]
        truth = np.asarray(raw["mu"][indices], dtype=np.float64)
        estimates = {
            mode: np.full(n, np.nan)
            for mode in ("coherent", "phase", "soft", "thresholded_phase")
        }
        estimators = {
            mode: PilotMuEstimator(PilotMuEstimatorConfig(
                sample_rate_hz=fs,
                samples_per_symbol=args.samples_per_symbol,
                rrc_beta=args.rrc_beta,
                rrc_span=args.rrc_span,
                mu_min=args.mu_min,
                mu_max=args.mu_max,
                coarse_step_hz_per_s=args.coarse_step,
                fine_radius_hz_per_s=args.fine_radius,
                fine_step_hz_per_s=args.fine_step,
                pilot_weighting=mode,
                soft_weight_scale=args.soft_weight_scale,
                amplitude_threshold_rel=args.amplitude_threshold_rel,
            )) for mode in estimates
        }
        iterator = range(n) if args.no_progress else tqdm(range(n), desc="Pilot weighting ablation")
        for position in iterator:
            i = int(indices[position])
            for mode, estimator in estimators.items():
                result = estimator.estimate(raw["iq"][i], raw["pilot_indices"][i], raw["pilot_symbols"][i])
                if result.valid:
                    estimates[mode][position] = result.mu_hat_hz_per_s

    rows: list[dict] = []
    for mode, estimate in estimates.items():
        valid = np.isfinite(estimate)
        rows.extend(_summary_rows(mode, snr[valid], estimate[valid] - truth[valid]))
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).query("snr_group == 'overall'").to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
