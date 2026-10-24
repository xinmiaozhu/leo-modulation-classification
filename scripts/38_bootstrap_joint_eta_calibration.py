#!/usr/bin/env python
"""Paired cluster bootstrap for the worst-direction joint-eta AMC tolerance."""

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
    p.add_argument("--envelope-csv", required=True)
    p.add_argument("--delta", type=float, default=0.02)
    p.add_argument("--bootstrap-replicates", type=int, default=20000)
    p.add_argument("--seed", type=int, default=646464)
    p.add_argument("--model", default="iq_constellation")
    p.add_argument("--simultaneous-alpha", type=float, default=0.05)
    return p.parse_args()


def contiguous_threshold(eta: np.ndarray, accuracy: np.ndarray, delta: float) -> float:
    order = np.argsort(eta)
    eta = eta[order]
    accuracy = accuracy[order]
    zero = np.flatnonzero(np.isclose(eta, 0.0))
    if zero.size != 1:
        raise ValueError("Expected exactly one eta=0 envelope point.")
    floor = float(accuracy[int(zero[0])]) - float(delta)
    accepted = 0.0
    for value, score in zip(eta, accuracy):
        if float(score) + 1e-12 < floor:
            break
        accepted = float(value)
    return accepted


def _column_groups(columns: pd.MultiIndex) -> tuple[np.ndarray, list[np.ndarray]]:
    eta = np.sort(np.unique(columns.get_level_values("eta_res").astype(float)))
    groups = [
        np.flatnonzero(np.isclose(columns.get_level_values("eta_res").astype(float), value))
        for value in eta
    ]
    return eta, groups


def _envelope(column_accuracy: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    return np.asarray([np.min(column_accuracy[index]) for index in groups], dtype=float)


def main() -> None:
    args = parse_args()
    detail = pd.read_csv(args.detail_csv)
    detail = detail[detail["model"].astype(str).eq(args.model)].copy()
    required = {
        "frame_index",
        "label",
        "snr_db",
        "eta_res",
        "direction_id",
        "correct",
        "allocation_u",
        "cfo_sign",
        "rate_sign",
    }
    missing = sorted(required.difference(detail.columns))
    if missing:
        raise SystemExit(f"Missing detail columns: {missing}")
    if detail.duplicated(["snr_db", "label", "frame_index", "eta_res", "direction_id"]).any():
        raise ValueError("Duplicate base-frame/direction outcomes in joint eta detail.")

    rng = np.random.default_rng(int(args.seed))
    records: list[dict[str, float | int | str]] = []
    envelope_records: list[dict[str, float | int | str]] = []
    distributions: dict[str, list[dict[str, float | int]]] = {}
    bootstrap_by_snr: dict[float, np.ndarray] = {}

    for snr, snr_group in detail.groupby("snr_db", sort=True):
        strata: list[pd.DataFrame] = []
        reference_columns: pd.MultiIndex | None = None
        for _, label_group in snr_group.groupby("label", sort=True):
            matrix = label_group.pivot(
                index="frame_index",
                columns=["eta_res", "direction_id"],
                values="correct",
            ).sort_index(axis=1)
            if matrix.isna().any().any():
                raise ValueError(f"Incomplete paired eta/direction grid at SNR={snr:g} dB.")
            if reference_columns is None:
                reference_columns = matrix.columns
            elif not matrix.columns.equals(reference_columns):
                raise ValueError(f"Direction columns differ across labels at SNR={snr:g} dB.")
            strata.append(matrix)
        if reference_columns is None:
            continue
        eta, eta_column_groups = _column_groups(reference_columns)
        stacked = np.vstack([matrix.to_numpy(dtype=float) for matrix in strata])
        column_accuracy = stacked.mean(axis=0)
        point_envelope = _envelope(column_accuracy, eta_column_groups)
        point = contiguous_threshold(eta, point_envelope, float(args.delta))

        direction_metadata = (
            snr_group[
                ["eta_res", "direction_id", "allocation_u", "cfo_sign", "rate_sign",
                 "residual_cfo_hz", "mu_res_hz_per_s"]
            ]
            .drop_duplicates(["eta_res", "direction_id"])
            .set_index(["eta_res", "direction_id"])
        )
        for eta_value, column_group, envelope_accuracy in zip(
            eta, eta_column_groups, point_envelope
        ):
            local = column_accuracy[column_group]
            worst_column = int(column_group[int(np.argmin(local))])
            direction_id = str(reference_columns[worst_column][1])
            meta = direction_metadata.loc[(float(eta_value), direction_id)]
            envelope_records.append(
                {
                    "snr_db": float(snr),
                    "eta_res": float(eta_value),
                    "accuracy": float(envelope_accuracy),
                    "baseline_accuracy": float(point_envelope[np.flatnonzero(np.isclose(eta, 0.0))[0]]),
                    "loss_pp": float(100.0 * (point_envelope[0] - envelope_accuracy)),
                    "worst_direction_id": direction_id,
                    "worst_allocation_u": float(meta["allocation_u"]),
                    "worst_cfo_sign": float(meta["cfo_sign"]),
                    "worst_rate_sign": float(meta["rate_sign"]),
                    "worst_residual_cfo_hz": float(meta["residual_cfo_hz"]),
                    "worst_mu_res_hz_per_s": float(meta["mu_res_hz_per_s"]),
                    "base_frames": int(stacked.shape[0]),
                    "direction_count": int(len(column_group)),
                }
            )

        bootstrap = np.empty(int(args.bootstrap_replicates), dtype=float)
        for replicate in range(int(args.bootstrap_replicates)):
            sampled = []
            for matrix in strata:
                values = matrix.to_numpy(dtype=float)
                draw = rng.integers(0, values.shape[0], size=values.shape[0])
                sampled.append(values[draw])
            sampled_accuracy = np.vstack(sampled).mean(axis=0)
            sampled_envelope = _envelope(sampled_accuracy, eta_column_groups)
            bootstrap[replicate] = contiguous_threshold(
                eta, sampled_envelope, float(args.delta)
            )

        lower = float(np.quantile(bootstrap, 0.025, method="lower"))
        upper = float(np.quantile(bootstrap, 0.975, method="higher"))
        records.append(
            {
                "snr_db": float(snr),
                "delta": float(args.delta),
                "eta_amc_star": float(point),
                "eta_ci95_low": float(lower),
                "eta_ci95_high": float(upper),
                "eta_ci95_pointwise_low": float(lower),
                "eta_ci95_pointwise_high": float(upper),
                "point_threshold_right_censored": bool(np.isclose(point, float(np.max(eta)))),
                "base_frames": int(stacked.shape[0]),
                "sampled_directions_nonzero": int(max(len(group) for group in eta_column_groups)),
                "bootstrap_replicates": int(args.bootstrap_replicates),
            }
        )
        bootstrap_by_snr[float(snr)] = bootstrap
        values, counts = np.unique(bootstrap, return_counts=True)
        distributions[f"{float(snr):g}"] = [
            {"eta": float(value), "count": int(count)}
            for value, count in zip(values, counts)
        ]

    result = pd.DataFrame(records).sort_values("snr_db").reset_index(drop=True)
    pointwise_lcb = result["eta_ci95_low"].to_numpy(dtype=float)
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
    result["eta_lcb95_pointwise_monotone"] = pointwise_monotone
    result["eta_lcb95_bonferroni"] = simultaneous_lcb
    result["eta_lcb95_bonferroni_monotone"] = simultaneous_monotone
    # Backward-compatible paper field now denotes the simultaneous envelope.
    result["eta_lcb95_monotone"] = simultaneous_monotone
    envelope_result = pd.DataFrame(envelope_records).sort_values(["snr_db", "eta_res"])

    for path_string in (args.output_csv, args.output_json, args.envelope_csv):
        Path(path_string).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    envelope_result.to_csv(args.envelope_csv, index=False)
    payload = {
        "calibration_source": args.detail_csv,
        "model": args.model,
        "delta": float(args.delta),
        "method": "modulation-stratified paired cluster percentile bootstrap",
        "cluster": "base frame; every eta allocation/sign direction remains paired",
        "direction_envelope": "minimum accuracy over the predeclared sampled directions",
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "seed": int(args.seed),
        "grid_resolved": bool(not result["point_threshold_right_censored"].any()),
        "simultaneous_method": "Bonferroni-adjusted one-sided percentile lower bounds",
        "simultaneous_family_size": family_size,
        "simultaneous_alpha": float(args.simultaneous_alpha),
        "marginal_lower_tail_probability": marginal_alpha,
        "conservative_snr_envelope": (
            "suffix-minimum of Bonferroni-adjusted lower bounds; nondecreasing in SNR"
        ),
        "results": result.to_dict(orient="records"),
        "bootstrap_histograms": distributions,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(result.to_string(index=False))
    print(f"Wrote {args.envelope_csv}")


if __name__ == "__main__":
    main()
