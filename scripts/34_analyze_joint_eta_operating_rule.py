#!/usr/bin/env python
"""Apply independently calibrated joint-eta thresholds to frozen test errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rate-operating-rule", required=True)
    p.add_argument("--eta-calibration", required=True)
    p.add_argument("--delta", type=float, default=0.02)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rate = pd.read_csv(args.rate_operating_rule)
    rate = rate[np.isclose(rate["delta"], float(args.delta))].copy()
    eta = pd.read_csv(args.eta_calibration)
    eta = eta[np.isclose(eta["delta"], float(args.delta))].copy()
    required_rate = {
        "snr_db",
        "test_joint_phase_bound_p90",
        "test_joint_phase_bound_p95",
        "baseline_test_joint_phase_bound_p90",
        "baseline_test_joint_phase_bound_p95",
    }
    required_eta = {
        "snr_db",
        "eta_amc_star",
        "eta_ci95_low",
        "eta_ci95_high",
        "eta_lcb95_monotone",
        "eta_lcb95_bonferroni_monotone",
        "point_threshold_right_censored",
    }
    if missing := sorted(required_rate.difference(rate.columns)):
        raise ValueError(f"Missing rate-operating-rule columns: {missing}")
    if missing := sorted(required_eta.difference(eta.columns)):
        raise ValueError(f"Missing eta-calibration columns: {missing}")

    keep_rate = [
        "snr_db",
        "test_joint_phase_bound_p90",
        "test_joint_phase_bound_p95",
        "baseline_test_joint_phase_bound_p90",
        "baseline_test_joint_phase_bound_p95",
    ]
    keep_eta = [
        "snr_db",
        "delta",
        "eta_amc_star",
        "eta_ci95_low",
        "eta_ci95_high",
        "eta_lcb95_monotone",
        "eta_lcb95_bonferroni_monotone",
        "point_threshold_right_censored",
        "base_frames",
        "sampled_directions_nonzero",
        "bootstrap_replicates",
    ]
    result = eta[keep_eta].merge(rate[keep_rate], on="snr_db", how="inner", validate="one_to_one")
    if len(result) != len(eta) or len(result) != len(rate):
        raise ValueError("SNR strata do not align between eta calibration and test errors.")
    point = result["eta_amc_star"].to_numpy(dtype=float)
    lcb = result["eta_lcb95_bonferroni_monotone"].to_numpy(dtype=float)
    hybrid_p90 = result["test_joint_phase_bound_p90"].to_numpy(dtype=float)
    hybrid_p95 = result["test_joint_phase_bound_p95"].to_numpy(dtype=float)
    grid_p90 = result["baseline_test_joint_phase_bound_p90"].to_numpy(dtype=float)
    grid_p95 = result["baseline_test_joint_phase_bound_p95"].to_numpy(dtype=float)
    if np.any(point <= 0.0):
        raise ValueError("Joint eta point thresholds must be positive on every reported SNR.")
    if result["point_threshold_right_censored"].astype(bool).any():
        raise ValueError("At least one joint eta point threshold is right-censored by its scan grid.")

    result["hybrid_p90_satisfied_eta"] = hybrid_p90 <= point
    result["hybrid_p95_satisfied_eta"] = hybrid_p95 <= point
    result["hybrid_p90_satisfied_eta_lcb95"] = hybrid_p90 <= lcb
    result["hybrid_p95_satisfied_eta_lcb95"] = hybrid_p95 <= lcb
    result["grid_p90_satisfied_eta"] = grid_p90 <= point
    result["grid_p95_satisfied_eta"] = grid_p95 <= point
    result["grid_p90_satisfied_eta_lcb95"] = grid_p90 <= lcb
    result["grid_p95_satisfied_eta_lcb95"] = grid_p95 <= lcb
    result["hybrid_p90_satisfied_eta_simultaneous_lcb95"] = hybrid_p90 <= lcb
    result["hybrid_p95_satisfied_eta_simultaneous_lcb95"] = hybrid_p95 <= lcb
    result["grid_p90_satisfied_eta_simultaneous_lcb95"] = grid_p90 <= lcb
    result["grid_p95_satisfied_eta_simultaneous_lcb95"] = grid_p95 <= lcb
    result["eta_lcb_to_point_ratio"] = lcb / point
    result["hybrid_p90_to_eta_star_ratio"] = hybrid_p90 / point
    result["hybrid_p95_to_eta_star_ratio"] = hybrid_p95 / point
    result["grid_p90_to_eta_star_ratio"] = grid_p90 / point
    result["grid_p95_to_eta_star_ratio"] = grid_p95 / point
    result = result.sort_values("snr_db").reset_index(drop=True)

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    payload = {
        "rate_operating_rule_source": args.rate_operating_rule,
        "eta_calibration_source": args.eta_calibration,
        "delta": float(args.delta),
        "test_split_role": "final closure only; no threshold, direction, model, or gate selection",
        "eta_definition": "2*pi*abs(e_f)*T_f + pi*abs(e_mu)*T_f^2",
        "closure_definition": "Q_(1-alpha)(eta|rho) <= eta_AMC_star(rho,delta)",
        "confidence_boundary": "Bonferroni-adjusted simultaneous 95% monotone lower envelope over five SNR bins",
        "results": result.to_dict(orient="records"),
    }
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        result[
            [
                "snr_db",
                "eta_amc_star",
                "eta_ci95_low",
                "eta_ci95_high",
                "test_joint_phase_bound_p90",
                "test_joint_phase_bound_p95",
                "hybrid_p90_satisfied_eta",
                "hybrid_p95_satisfied_eta",
                "hybrid_p90_satisfied_eta_lcb95",
                "hybrid_p95_satisfied_eta_lcb95",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
