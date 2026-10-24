#!/usr/bin/env python
"""Paired cluster bootstrap for SNR-conditioned AMC severity thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detail-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--delta", type=float, default=0.02)
    p.add_argument("--bootstrap-replicates", type=int, default=20000)
    p.add_argument("--seed", type=int, default=515151)
    p.add_argument("--frame-duration-s", type=float, default=0.04096)
    p.add_argument("--model", default="iq_evm")
    p.add_argument("--envelope-csv", default=None, help="Optional worst-sign response CSV.")
    p.add_argument("--simultaneous-alpha", type=float, default=0.05)
    return p.parse_args()


def contiguous_threshold(gamma: np.ndarray, accuracy: np.ndarray, delta: float) -> float:
    order = np.argsort(gamma)
    gamma = gamma[order]
    accuracy = accuracy[order]
    zero = np.flatnonzero(np.isclose(gamma, 0.0))
    if zero.size != 1:
        raise ValueError("Expected exactly one zero-severity point.")
    floor = float(accuracy[int(zero[0])]) - float(delta)
    accepted = 0.0
    for value, score in zip(gamma, accuracy):
        if float(score) + 1e-12 < floor:
            break
        accepted = float(value)
    return accepted


def main() -> None:
    args = parse_args()
    detail = pd.read_csv(args.detail_csv)
    detail = detail[detail["model"].astype(str).eq(args.model)].copy()
    required = {
        "frame_index", "label", "snr_db", "gamma_res", "injection_sign", "correct"
    }
    missing = sorted(required.difference(detail.columns))
    if missing:
        raise SystemExit(f"Missing detail columns: {missing}")

    rng = np.random.default_rng(int(args.seed))
    records: list[dict[str, float | int | str]] = []
    envelope_records: list[dict[str, float | int | str]] = []
    distributions: dict[str, list[dict[str, float | int]]] = {}
    bootstrap_by_snr: dict[float, np.ndarray] = {}

    for snr, snr_group in detail.groupby("snr_db", sort=True):
        gamma = np.sort(snr_group["gamma_res"].unique().astype(float))
        zero_index = np.flatnonzero(np.isclose(gamma, 0.0))
        if zero_index.size != 1:
            raise ValueError(f"Expected one zero-severity column at SNR={snr:g} dB.")
        strata: list[dict[float, np.ndarray]] = []
        for _, label_group in snr_group.groupby("label", sort=True):
            frames = np.sort(label_group["frame_index"].unique())
            sign_matrices: dict[float, np.ndarray] = {}
            zero = (
                label_group[np.isclose(label_group["gamma_res"], 0.0)]
                .drop_duplicates("frame_index")
                .set_index("frame_index")["correct"]
                .reindex(frames)
                .to_numpy(dtype=float)
            )
            if np.isnan(zero).any():
                raise ValueError(f"Incomplete zero-severity frames at SNR={snr:g} dB.")
            for sign in (-1.0, 1.0):
                nonzero = label_group[
                    (~np.isclose(label_group["gamma_res"], 0.0))
                    & np.isclose(label_group["injection_sign"], sign)
                ]
                matrix = nonzero.pivot(
                    index="frame_index", columns="gamma_res", values="correct"
                ).reindex(index=frames, columns=gamma[~np.isclose(gamma, 0.0)])
                values = np.empty((len(frames), len(gamma)), dtype=float)
                values[:, int(zero_index[0])] = zero
                values[:, ~np.isclose(gamma, 0.0)] = matrix.to_numpy(dtype=float)
                if np.isnan(values).any():
                    raise ValueError(
                        f"Incomplete paired sign/severity grid at SNR={snr:g} dB, sign={sign:+g}."
                    )
                sign_matrices[sign] = values
            strata.append(sign_matrices)

        sign_accuracy = {
            sign: np.vstack([item[sign] for item in strata]).mean(axis=0)
            for sign in (-1.0, 1.0)
        }
        full_accuracy = np.minimum(sign_accuracy[-1.0], sign_accuracy[1.0])
        point = contiguous_threshold(gamma, full_accuracy, args.delta)
        point_minus = contiguous_threshold(gamma, sign_accuracy[-1.0], args.delta)
        point_plus = contiguous_threshold(gamma, sign_accuracy[1.0], args.delta)
        base_frames = int(sum(item[-1.0].shape[0] for item in strata))
        for index, severity in enumerate(gamma):
            minus = float(sign_accuracy[-1.0][index])
            plus = float(sign_accuracy[1.0][index])
            envelope_records.append(
                {
                    "model": str(args.model),
                    "snr_db": float(snr),
                    "gamma_res": float(severity),
                    "count": base_frames,
                    "accuracy": float(min(minus, plus)),
                    "accuracy_minus": minus,
                    "accuracy_plus": plus,
                    "sign_gap_pp": float(100.0 * abs(plus - minus)),
                    "worst_sign": -1 if minus <= plus else 1,
                }
            )
        bootstrap = np.empty(int(args.bootstrap_replicates), dtype=float)
        for replicate in range(int(args.bootstrap_replicates)):
            sampled = {-1.0: [], 1.0: []}
            for matrices in strata:
                rows = matrices[-1.0].shape[0]
                draw = rng.integers(0, rows, size=rows)
                for sign in (-1.0, 1.0):
                    sampled[sign].append(matrices[sign][draw])
            sampled_accuracy = {
                sign: np.vstack(sampled[sign]).mean(axis=0) for sign in (-1.0, 1.0)
            }
            accuracy = np.minimum(sampled_accuracy[-1.0], sampled_accuracy[1.0])
            bootstrap[replicate] = contiguous_threshold(gamma, accuracy, args.delta)

        lower = float(np.quantile(bootstrap, 0.025, method="lower"))
        upper = float(np.quantile(bootstrap, 0.975, method="higher"))
        scale = 1.0 / (np.pi * float(args.frame_duration_s) ** 2)
        records.append(
            {
                "snr_db": float(snr),
                "delta": float(args.delta),
                "gamma_amc_star": float(point),
                "gamma_amc_star_minus": float(point_minus),
                "gamma_amc_star_plus": float(point_plus),
                "gamma_ci95_low": float(lower),
                "gamma_ci95_high": float(upper),
                "gamma_ci95_pointwise_low": float(lower),
                "gamma_ci95_pointwise_high": float(upper),
                "e_mu_max_hz_per_s": float(point * scale),
                "e_mu_ci95_low_hz_per_s": float(lower * scale),
                "e_mu_ci95_high_hz_per_s": float(upper * scale),
                "base_frames": base_frames,
                "bootstrap_replicates": int(args.bootstrap_replicates),
                "point_threshold_right_censored": bool(np.isclose(point, float(np.max(gamma)))),
            }
        )
        bootstrap_by_snr[float(snr)] = bootstrap
        values, counts = np.unique(bootstrap, return_counts=True)
        distributions[f"{float(snr):g}"] = [
            {"gamma": float(value), "count": int(count)}
            for value, count in zip(values, counts)
        ]

    result = pd.DataFrame(records).sort_values("snr_db").reset_index(drop=True)
    # Pointwise and Bonferroni-adjusted simultaneous lower confidence bounds.
    pointwise_lcb = result["gamma_ci95_low"].to_numpy(dtype=float)
    pointwise_monotone = np.minimum.accumulate(pointwise_lcb[::-1])[::-1]
    family_size = int(len(result))
    marginal_alpha = float(args.simultaneous_alpha) / family_size
    simultaneous_lcb = np.asarray(
        [
            np.quantile(
                bootstrap_by_snr[float(snr)], marginal_alpha, method="lower"
            )
            for snr in result["snr_db"]
        ],
        dtype=float,
    )
    simultaneous_monotone = np.minimum.accumulate(simultaneous_lcb[::-1])[::-1]
    scale = 1.0 / (np.pi * float(args.frame_duration_s) ** 2)
    result["gamma_lcb95_pointwise_monotone"] = pointwise_monotone
    result["e_mu_lcb95_pointwise_monotone_hz_per_s"] = pointwise_monotone * scale
    result["gamma_lcb95_bonferroni"] = simultaneous_lcb
    result["gamma_lcb95_bonferroni_monotone"] = simultaneous_monotone
    result["e_mu_lcb95_bonferroni_hz_per_s"] = simultaneous_lcb * scale
    result["e_mu_lcb95_bonferroni_monotone_hz_per_s"] = simultaneous_monotone * scale
    # Backward-compatible names now point to the simultaneous envelope used by the paper.
    result["gamma_lcb95_monotone"] = simultaneous_monotone
    result["e_mu_lcb95_monotone_hz_per_s"] = simultaneous_monotone * scale
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    if args.envelope_csv:
        envelope_path = Path(args.envelope_csv)
        envelope_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(envelope_records).sort_values(["snr_db", "gamma_res"]).to_csv(
            envelope_path, index=False
        )
    payload = {
        "calibration_source": args.detail_csv,
        "model": args.model,
        "delta": float(args.delta),
        "method": "worst-sign modulation-stratified paired cluster percentile bootstrap",
        "accuracy_envelope": "min(A_minus,A_plus) at every sampled severity",
        "cluster": "base frame; the same resample indices are used for both signs",
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "seed": int(args.seed),
        "grid_resolved": bool(not result["point_threshold_right_censored"].any()),
        "simultaneous_method": "Bonferroni-adjusted one-sided percentile lower bounds",
        "simultaneous_family_size": family_size,
        "simultaneous_alpha": float(args.simultaneous_alpha),
        "marginal_lower_tail_probability": marginal_alpha,
        "conservative_envelope": (
            "suffix-minimum of Bonferroni-adjusted lower bounds; nondecreasing in SNR"
        ),
        "results": result.to_dict(orient="records"),
        "bootstrap_histograms": distributions,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
