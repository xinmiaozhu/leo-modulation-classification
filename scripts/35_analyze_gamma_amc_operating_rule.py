#!/usr/bin/env python
"""Calibrate SNR-conditioned AMC severity tolerances and test estimator coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.common import format_ieee_axis, set_ieee_trans_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-scan", required=True)
    parser.add_argument("--raw-data", required=True)
    parser.add_argument("--pilot-estimates", required=True)
    parser.add_argument("--baseline-pilot-estimates", default=None)
    parser.add_argument(
        "--estimator-label",
        default="Diagnostic",
        help="Legend prefix for --pilot-estimates (for example, 'Hybrid DFRFT').",
    )
    parser.add_argument(
        "--calibration-ci",
        default=None,
        help="Optional paired-bootstrap calibration CSV with gamma/e_mu 95%% intervals.",
    )
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.01, 0.02])
    parser.add_argument("--primary-delta", type=float, default=0.02)
    return parser.parse_args()


def contiguous_tolerance(group: pd.DataFrame, delta: float) -> tuple[float, float]:
    ordered = group.sort_values("gamma_res").reset_index(drop=True)
    zero = ordered[np.isclose(ordered["gamma_res"], 0.0)]
    if len(zero) != 1:
        raise ValueError("Each SNR stratum must contain exactly one gamma=0 row.")
    baseline = float(zero.iloc[0]["accuracy"])
    threshold = baseline - float(delta)
    accepted = 0.0
    for row in ordered.itertuples(index=False):
        if float(row.accuracy) + 1e-12 < threshold:
            break
        accepted = float(row.gamma_res)
    return accepted, baseline


def main() -> None:
    args = parse_args()
    scan = pd.read_csv(args.validation_scan)
    if "model" in scan:
        scan = scan[scan["model"] == "iq_evm"].copy()
    scan = (
        scan.groupby(["snr_db", "gamma_res"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), count=("count", "sum"))
        .sort_values(["snr_db", "gamma_res"])
    )

    with np.load(args.splits) as split_file:
        test_indices = np.asarray(split_file["test"], dtype=np.int64)
    with h5py.File(args.raw_data, "r") as raw, h5py.File(args.pilot_estimates, "r") as est:
        snr_test = np.asarray(raw["snr_db"][test_indices], dtype=np.float64)
        mu_true = np.asarray(raw["mu"][test_indices], dtype=np.float64)
        mu_hat = np.asarray(est["pilot_mu_hat"][test_indices], dtype=np.float64)
        duration = np.asarray(raw["T"][test_indices], dtype=np.float64)
        fd_true = np.asarray(raw["fd0"][test_indices], dtype=np.float64)
        fd_hat = np.asarray(est["pilot_fd0_hat"][test_indices], dtype=np.float64)
    error = np.abs(mu_true - mu_hat)
    fd_error = np.abs(fd_true - fd_hat)
    baseline_error = None
    baseline_fd_error = None
    if args.baseline_pilot_estimates:
        with h5py.File(args.baseline_pilot_estimates, "r") as baseline_est:
            baseline_mu_hat = np.asarray(baseline_est["pilot_mu_hat"][test_indices], dtype=np.float64)
            baseline_fd_hat = np.asarray(baseline_est["pilot_fd0_hat"][test_indices], dtype=np.float64)
        baseline_error = np.abs(mu_true - baseline_mu_hat)
        baseline_fd_error = np.abs(fd_true - baseline_fd_hat)

    records: list[dict[str, float | bool | int]] = []
    for delta in args.deltas:
        for snr, group in scan.groupby("snr_db", sort=True):
            gamma_star, baseline = contiguous_tolerance(group, float(delta))
            mask = np.isclose(snr_test, float(snr))
            if not np.any(mask):
                raise ValueError(f"No independent-test estimator samples at SNR={snr:g} dB.")
            tf = float(np.median(duration[mask]))
            limit = gamma_star / (np.pi * tf**2)
            p90 = float(np.quantile(error[mask], 0.90))
            p95 = float(np.quantile(error[mask], 0.95))
            fd_p90 = float(np.quantile(fd_error[mask], 0.90))
            fd_p95 = float(np.quantile(fd_error[mask], 0.95))
            # Conservative accumulated phase severity for coexisting residual
            # CFO and rate.  This coordinate is evaluated against its own
            # independently calibrated eta tolerance downstream; it must not
            # be compared with the conditional rate-only gamma tolerance.
            joint_phase_bound = (
                2.0 * np.pi * fd_error[mask] * duration[mask]
                + np.pi * error[mask] * duration[mask] ** 2
            )
            joint_p90 = float(np.quantile(joint_phase_bound, 0.90))
            joint_p95 = float(np.quantile(joint_phase_bound, 0.95))
            response = scan[np.isclose(scan["snr_db"], float(snr))].sort_values("gamma_res")
            response_gamma = response["gamma_res"].to_numpy(dtype=float)
            response_accuracy = response["accuracy"].to_numpy(dtype=float)
            observed_gamma = np.pi * error[mask] * duration[mask] ** 2
            distributional_accuracy = float(
                np.mean(np.interp(observed_gamma, response_gamma, response_accuracy))
            )
            records.append(
                {
                    "snr_db": float(snr),
                    "delta": float(delta),
                    "baseline_accuracy": baseline,
                    "gamma_amc_star": gamma_star,
                    "e_mu_max_hz_per_s": limit,
                    "test_count": int(np.sum(mask)),
                    "test_mu_abs_error_p90": p90,
                    "test_mu_abs_error_p95": p95,
                    "test_fd0_abs_error_p90": fd_p90,
                    "test_fd0_abs_error_p95": fd_p95,
                    "test_joint_phase_bound_p90": joint_p90,
                    "test_joint_phase_bound_p95": joint_p95,
                    "p90_satisfied": bool(p90 <= limit),
                    "p95_satisfied": bool(p95 <= limit),
                    "joint_p90_satisfied": bool(joint_p90 <= gamma_star),
                    "joint_p95_satisfied": bool(joint_p95 <= gamma_star),
                    "distributional_accuracy": distributional_accuracy,
                    "distributional_loss": float(baseline - distributional_accuracy),
                }
            )
            if baseline_error is not None:
                records[-1]["baseline_test_mu_abs_error_p90"] = float(np.quantile(baseline_error[mask], 0.90))
                records[-1]["baseline_test_mu_abs_error_p95"] = float(np.quantile(baseline_error[mask], 0.95))
                records[-1]["baseline_test_fd0_abs_error_p90"] = float(np.quantile(baseline_fd_error[mask], 0.90))
                records[-1]["baseline_test_fd0_abs_error_p95"] = float(np.quantile(baseline_fd_error[mask], 0.95))
                baseline_joint_phase = (
                    2.0 * np.pi * baseline_fd_error[mask] * duration[mask]
                    + np.pi * baseline_error[mask] * duration[mask] ** 2
                )
                records[-1]["baseline_test_joint_phase_bound_p90"] = float(np.quantile(baseline_joint_phase, 0.90))
                records[-1]["baseline_test_joint_phase_bound_p95"] = float(np.quantile(baseline_joint_phase, 0.95))
                baseline_gamma = np.pi * baseline_error[mask] * duration[mask] ** 2
                baseline_distributional_accuracy = float(
                    np.mean(np.interp(baseline_gamma, response_gamma, response_accuracy))
                )
                records[-1]["baseline_distributional_accuracy"] = baseline_distributional_accuracy
                records[-1]["baseline_distributional_loss"] = float(baseline - baseline_distributional_accuracy)
    result = pd.DataFrame(records).sort_values(["delta", "snr_db"])
    if args.calibration_ci:
        ci = pd.read_csv(args.calibration_ci)
        ci_columns = [
            "snr_db",
            "delta",
            "gamma_ci95_low",
            "gamma_ci95_high",
            "e_mu_ci95_low_hz_per_s",
            "e_mu_ci95_high_hz_per_s",
            "gamma_lcb95_monotone",
            "e_mu_lcb95_monotone_hz_per_s",
            "base_frames",
            "bootstrap_replicates",
        ]
        optional_ci_columns = [
            "gamma_amc_star_minus",
            "gamma_amc_star_plus",
            "gamma_ci95_pointwise_low",
            "gamma_ci95_pointwise_high",
            "gamma_lcb95_pointwise_monotone",
            "e_mu_lcb95_pointwise_monotone_hz_per_s",
            "gamma_lcb95_bonferroni",
            "gamma_lcb95_bonferroni_monotone",
            "e_mu_lcb95_bonferroni_hz_per_s",
            "e_mu_lcb95_bonferroni_monotone_hz_per_s",
        ]
        ci_columns.extend(column for column in optional_ci_columns if column in ci.columns)
        result = result.merge(ci[ci_columns], on=["snr_db", "delta"], how="left")
        result["p90_satisfied_lcb95"] = (
            result["test_mu_abs_error_p90"] <= result["e_mu_ci95_low_hz_per_s"]
        )
        result["p95_satisfied_lcb95"] = (
            result["test_mu_abs_error_p95"] <= result["e_mu_ci95_low_hz_per_s"]
        )
        result["p90_satisfied_monotone_lcb95"] = (
            result["test_mu_abs_error_p90"] <= result["e_mu_lcb95_monotone_hz_per_s"]
        )
        result["p95_satisfied_monotone_lcb95"] = (
            result["test_mu_abs_error_p95"] <= result["e_mu_lcb95_monotone_hz_per_s"]
        )
        result["joint_p90_satisfied_lcb95"] = (
            result["test_joint_phase_bound_p90"] <= result["gamma_ci95_low"]
        )
        result["joint_p95_satisfied_lcb95"] = (
            result["test_joint_phase_bound_p95"] <= result["gamma_ci95_low"]
        )
        if "e_mu_lcb95_bonferroni_monotone_hz_per_s" in result:
            result["p90_satisfied_simultaneous_lcb95"] = (
                result["test_mu_abs_error_p90"]
                <= result["e_mu_lcb95_bonferroni_monotone_hz_per_s"]
            )
            result["p95_satisfied_simultaneous_lcb95"] = (
                result["test_mu_abs_error_p95"]
                <= result["e_mu_lcb95_bonferroni_monotone_hz_per_s"]
            )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)

    primary = result[np.isclose(result["delta"], float(args.primary_delta))].copy()
    if primary.empty:
        raise ValueError("primary-delta must be included in --deltas.")
    summary = {
        "response_scan": str(args.validation_scan),
        "response_role": "caller-supplied validation or independent-calibration scan",
        "estimator_evaluation_split": "test",
        "primary_delta": float(args.primary_delta),
        "snr_db": primary["snr_db"].tolist(),
        "gamma_amc_star": primary["gamma_amc_star"].tolist(),
        "e_mu_max_hz_per_s": primary["e_mu_max_hz_per_s"].tolist(),
        "test_mu_abs_error_p90": primary["test_mu_abs_error_p90"].tolist(),
        "test_mu_abs_error_p95": primary["test_mu_abs_error_p95"].tolist(),
        "test_fd0_abs_error_p90": primary["test_fd0_abs_error_p90"].tolist(),
        "test_fd0_abs_error_p95": primary["test_fd0_abs_error_p95"].tolist(),
        "test_joint_phase_bound_p90": primary["test_joint_phase_bound_p90"].tolist(),
        "test_joint_phase_bound_p95": primary["test_joint_phase_bound_p95"].tolist(),
        "p90_satisfied": primary["p90_satisfied"].tolist(),
        "p95_satisfied": primary["p95_satisfied"].tolist(),
        "joint_p90_satisfied": primary["joint_p90_satisfied"].tolist(),
        "joint_p95_satisfied": primary["joint_p95_satisfied"].tolist(),
        "distributional_accuracy": primary["distributional_accuracy"].tolist(),
        "distributional_loss": primary["distributional_loss"].tolist(),
        "boundary_definition": "largest contiguous sampled gamma from zero meeting A(gamma,rho)>=A(0,rho)-delta",
        "joint_phase_bound_definition": "eta=2*pi*abs(e_f)*T + pi*abs(e_mu)*T^2; a conservative joint phase-severity coordinate evaluated against independently calibrated eta_AMC_star",
    }
    if "gamma_ci95_low" in primary:
        summary["gamma_ci95_low"] = primary["gamma_ci95_low"].tolist()
        summary["gamma_ci95_high"] = primary["gamma_ci95_high"].tolist()
        summary["e_mu_ci95_low_hz_per_s"] = primary["e_mu_ci95_low_hz_per_s"].tolist()
        summary["e_mu_ci95_high_hz_per_s"] = primary["e_mu_ci95_high_hz_per_s"].tolist()
        summary["gamma_lcb95_monotone"] = primary["gamma_lcb95_monotone"].tolist()
        summary["e_mu_lcb95_monotone_hz_per_s"] = primary["e_mu_lcb95_monotone_hz_per_s"].tolist()
        summary["p90_satisfied_lcb95"] = primary["p90_satisfied_lcb95"].tolist()
        summary["p95_satisfied_lcb95"] = primary["p95_satisfied_lcb95"].tolist()
        summary["p90_satisfied_monotone_lcb95"] = primary["p90_satisfied_monotone_lcb95"].tolist()
        summary["p95_satisfied_monotone_lcb95"] = primary["p95_satisfied_monotone_lcb95"].tolist()
        summary["joint_p90_satisfied_lcb95"] = primary["joint_p90_satisfied_lcb95"].tolist()
        summary["joint_p95_satisfied_lcb95"] = primary["joint_p95_satisfied_lcb95"].tolist()
        summary["calibration_ci"] = str(args.calibration_ci)
        if "gamma_lcb95_bonferroni_monotone" in primary:
            summary["gamma_lcb95_bonferroni_monotone"] = primary[
                "gamma_lcb95_bonferroni_monotone"
            ].tolist()
            summary["e_mu_lcb95_bonferroni_monotone_hz_per_s"] = primary[
                "e_mu_lcb95_bonferroni_monotone_hz_per_s"
            ].tolist()
            summary["simultaneous_confidence_method"] = (
                "Bonferroni-adjusted one-sided 99% marginal bootstrap LCBs over five SNR bins"
            )
    if baseline_error is not None:
        summary["baseline_pilot_estimates"] = str(args.baseline_pilot_estimates)
        summary["baseline_test_mu_abs_error_p90"] = primary["baseline_test_mu_abs_error_p90"].tolist()
        summary["baseline_test_mu_abs_error_p95"] = primary["baseline_test_mu_abs_error_p95"].tolist()
        summary["baseline_test_fd0_abs_error_p90"] = primary["baseline_test_fd0_abs_error_p90"].tolist()
        summary["baseline_test_fd0_abs_error_p95"] = primary["baseline_test_fd0_abs_error_p95"].tolist()
        summary["baseline_test_joint_phase_bound_p90"] = primary["baseline_test_joint_phase_bound_p90"].tolist()
        summary["baseline_test_joint_phase_bound_p95"] = primary["baseline_test_joint_phase_bound_p95"].tolist()
        summary["baseline_distributional_accuracy"] = primary["baseline_distributional_accuracy"].tolist()
        summary["baseline_distributional_loss"] = primary["baseline_distributional_loss"].tolist()
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Match the 10:9 single-column aspect used by Fig. 6 and place the
    # calibration and estimator test in one visual coordinate system.
    set_ieee_trans_style()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8.2,
        }
    )
    fig, ax_error = plt.subplots(figsize=(3.45, 3.105))
    ax_gamma = ax_error.twinx()
    x = primary["snr_db"].to_numpy(dtype=float)

    gamma_bars = ax_gamma.bar(
        x,
        primary["gamma_amc_star"],
        width=1.25,
        color="#D9EAF7",
        edgecolor="#1677C8",
        linewidth=0.8,
        alpha=0.82,
        label=r"Tolerance $\gamma_{\mathrm{AMC}}^{\star}$",
        zorder=1,
    )
    if "gamma_ci95_low" in primary and primary["gamma_ci95_low"].notna().all():
        gamma_value = primary["gamma_amc_star"].to_numpy(dtype=float)
        gamma_low = primary["gamma_ci95_low"].to_numpy(dtype=float)
        gamma_high = primary["gamma_ci95_high"].to_numpy(dtype=float)
        ax_gamma.errorbar(
            x,
            gamma_value,
            yerr=np.vstack([gamma_value - gamma_low, gamma_high - gamma_value]),
            fmt="none",
            ecolor="#1677C8",
            elinewidth=1.0,
            capsize=2.5,
            capthick=0.9,
            zorder=5,
        )
    limit_line, = ax_error.plot(
        x,
        primary["e_mu_max_hz_per_s"],
        "o-",
        color="#1677C8",
        linewidth=1.35,
        markersize=4.2,
        label=r"Allowable error $e_{\mu,\max}$",
        zorder=4,
    )
    if "e_mu_ci95_low_hz_per_s" in primary and primary["e_mu_ci95_low_hz_per_s"].notna().all():
        ax_error.fill_between(
            x,
            primary["e_mu_ci95_low_hz_per_s"].to_numpy(dtype=float),
            primary["e_mu_ci95_high_hz_per_s"].to_numpy(dtype=float),
            color="#1677C8",
            alpha=0.10,
            linewidth=0.0,
            zorder=2,
        )
    if baseline_error is not None:
        baseline_p90_line, = ax_error.plot(
            x,
            primary["baseline_test_mu_abs_error_p90"],
            "s:",
            color="#777777",
            linewidth=1.0,
            markersize=3.7,
            label="Grid estimator P90",
            zorder=3,
        )
        baseline_p95_line, = ax_error.plot(
            x,
            primary["baseline_test_mu_abs_error_p95"],
            "^:",
            color="#AAAAAA",
            linewidth=1.0,
            markersize=4.0,
            label="Grid estimator P95",
            zorder=3,
        )
    p90_line, = ax_error.plot(
        x,
        primary["test_mu_abs_error_p90"],
        "s--",
        color="#D55E00",
        linewidth=1.2,
        markersize=4.0,
        label=f"{args.estimator_label} P90",
        zorder=4,
    )
    p95_line, = ax_error.plot(
        x,
        primary["test_mu_abs_error_p95"],
        "^--",
        color="#009E73",
        linewidth=1.2,
        markersize=4.4,
        label=f"{args.estimator_label} P95",
        zorder=4,
    )

    ax_error.set_xlabel("SNR (dB)")
    ax_error.set_ylabel("Doppler-rate error (Hz/s)")
    ax_gamma.set_ylabel(r"AMC severity tolerance $\gamma_{\mathrm{AMC}}^{\star}$")
    ax_error.set_xticks(x)
    ax_error.set_ylim(0, 650)
    ax_gamma.set_ylim(0, 1.0)
    ax_gamma.set_yticks(np.arange(0.0, 1.01, 0.2))
    format_ieee_axis(ax_error)
    ax_error.grid(True, linestyle="-.", linewidth=0.4, color="0.75", zorder=0)
    ax_gamma.grid(False)
    legend_handles = [gamma_bars, limit_line]
    legend_labels = [r"Tolerance $\gamma_{\mathrm{AMC}}^{\star}$", r"Allowable error $e_{\mu,\max}$"]
    if baseline_error is not None:
        legend_handles.extend([baseline_p90_line, baseline_p95_line])
        legend_labels.extend(["Grid estimator P90", "Grid estimator P95"])
    legend_handles.extend([p90_line, p95_line])
    legend_labels.extend([f"{args.estimator_label} P90", f"{args.estimator_label} P95"])
    ax_error.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=True,
        ncol=2,
        fontsize=8.2,
        borderpad=0.35,
        labelspacing=0.25,
        handlelength=2.1,
    )
    fig.tight_layout(pad=0.35)
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, bbox_inches="tight")
    plt.close(fig)
    print(primary.to_string(index=False))


if __name__ == "__main__":
    main()
