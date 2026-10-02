#!/usr/bin/env python
"""Refine joint CFO--rate estimates by residual pilot-phase regression.

The existing coherent grid estimate supplies an ambiguity-safe initialization.
After de-rotation, a robust quadratic phase fit estimates small CFO and
Doppler-rate corrections.  A receiver-SNR gate, selected on validation only,
prevents phase-unwrapping failures at low SNR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import pandas as pd
import torch

from src.signal.pulse_shape import rrc_filter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-data", required=True)
    p.add_argument("--initial-estimates", required=True)
    p.add_argument("--receiver-features", required=True, help="HDF5 containing pilot_snr_est_db.")
    p.add_argument("--splits", required=True)
    p.add_argument("--operating-rule", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-csv", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gate-candidates", type=float, nargs="+", default=[-2, -1, 0, 1, 2, 3, 4, 5, 6])
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--regression",
        choices=["ols", "weighted_huber"],
        default="weighted_huber",
        help="Residual pilot-phase regression used after grid acquisition.",
    )
    p.add_argument("--huber-iterations", type=int, default=5)
    p.add_argument("--huber-delta-rad", type=float, default=0.70)
    p.add_argument(
        "--gate-objective",
        choices=["rate_tail", "joint_phase_tail"],
        default="joint_phase_tail",
        help=(
            "Validation objective. joint_phase_tail includes both residual CFO "
            "and residual Doppler rate through their conservative accumulated "
            "phase severity."
        ),
    )
    return p.parse_args()


def _weighted_lstsq(design: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    root = np.sqrt(np.maximum(np.asarray(weights, dtype=np.float64), 1e-8))
    return np.linalg.lstsq(design * root[:, None], target * root, rcond=None)[0]


def _fit_residual_phase(
    residual: np.ndarray,
    u: np.ndarray,
    method: str,
    huber_iterations: int,
    huber_delta_rad: float,
) -> tuple[np.ndarray, float]:
    phase = np.unwrap(np.angle(residual))
    design = np.column_stack([np.ones_like(u), 2.0 * np.pi * u, np.pi * u**2])
    if method == "ols":
        coefficient = np.linalg.lstsq(design, phase, rcond=None)[0]
    else:
        magnitude = np.abs(residual)
        reference = max(float(np.median(magnitude)), 1e-12)
        amplitude_weights = np.clip(magnitude / reference, 0.1, 2.0)
        coefficient = _weighted_lstsq(design, phase, amplitude_weights)
        delta = max(float(huber_delta_rad), 1e-6)
        for _ in range(max(int(huber_iterations), 1)):
            error = phase - design @ coefficient
            robust_weights = np.minimum(1.0, delta / (np.abs(error) + 1e-12))
            coefficient = _weighted_lstsq(
                design, phase, amplitude_weights * robust_weights
            )
    error = phase - design @ coefficient
    return coefficient, float(np.sqrt(np.mean(error**2)))


def _tail_stats(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.90)), float(np.quantile(values, 0.95))


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with np.load(args.splits) as split_file:
        val_idx = np.asarray(split_file["val"], dtype=np.int64)
        test_idx = np.asarray(split_file["test"], dtype=np.int64)

    with h5py.File(args.raw_data, "r") as raw, h5py.File(args.initial_estimates, "r") as initial, h5py.File(args.receiver_features, "r") as receiver:
        n = int(raw["iq"].shape[0])
        fs = float(raw["fs"][0])
        sps = 8
        mu_initial = np.asarray(initial["pilot_mu_hat"][:], dtype=np.float64)
        fd_initial = np.asarray(initial["pilot_fd0_hat"][:], dtype=np.float64)
        receiver_snr = np.asarray(receiver["pilot_snr_est_db"][:], dtype=np.float64)
        mu_candidate = mu_initial.copy()
        fd_candidate = fd_initial.copy()
        correction_valid = np.zeros(n, dtype=bool)
        correction_residual_rmse = np.full(n, np.nan, dtype=np.float64)

        taps = rrc_filter(beta=0.35, span=8, sps=sps)
        requested_device = str(args.device)
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        device = torch.device(requested_device)
        kernel = torch.as_tensor(
            taps[::-1].copy(), dtype=torch.float32, device=device
        ).reshape(1, 1, -1)
        padding = int(kernel.shape[-1] // 2)

        for start in range(0, n, int(args.batch_size)):
            stop = min(start + int(args.batch_size), n)
            frame = torch.as_tensor(
                np.asarray(raw["iq"][start:stop], dtype=np.float32), device=device
            )
            real = torch.nn.functional.conv1d(frame[:, 0:1], kernel, padding=padding).squeeze(1)
            imag = torch.nn.functional.conv1d(frame[:, 1:2], kernel, padding=padding).squeeze(1)
            matched = torch.complex(real, imag)
            pilot_indices = np.asarray(raw["pilot_indices"][start:stop], dtype=np.int64)
            pilot_symbols = np.asarray(raw["pilot_symbols"][start:stop], dtype=np.complex64)
            sample_indices = pilot_indices * sps
            sample_t = torch.as_tensor(sample_indices, dtype=torch.long)
            observed = torch.gather(matched, 1, sample_t.to(device)).cpu().numpy()
            z = observed * np.conj(pilot_symbols) / (np.abs(pilot_symbols) ** 2 + 1e-12)

            for local, global_index in enumerate(range(start, stop)):
                t = sample_indices[local].astype(np.float64) / fs
                center = float(np.mean(t))
                u = t - center
                residual = z[local] * np.exp(
                    -1j * (2.0 * np.pi * fd_initial[global_index] * t + np.pi * mu_initial[global_index] * t**2)
                )
                coefficient, correction_residual_rmse[global_index] = _fit_residual_phase(
                    residual,
                    u,
                    method=str(args.regression),
                    huber_iterations=int(args.huber_iterations),
                    huber_delta_rad=float(args.huber_delta_rad),
                )
                rate_correction = float(coefficient[2])
                centered_cfo_correction = float(coefficient[1])
                if np.isfinite(rate_correction) and np.isfinite(centered_cfo_correction):
                    mu_candidate[global_index] = mu_initial[global_index] + rate_correction
                    fd_candidate[global_index] = (
                        fd_initial[global_index] + centered_cfo_correction - rate_correction * center
                    )
                    correction_valid[global_index] = True

        mu_true = np.asarray(raw["mu"][:], dtype=np.float64)
        snr_true = np.asarray(raw["snr_db"][:], dtype=np.float64)
        duration = np.asarray(raw["T"][:], dtype=np.float64)
        fd_true = np.asarray(raw["fd0"][:], dtype=np.float64)

    rule = pd.read_csv(args.operating_rule)
    rule = rule[np.isclose(rule["delta"], 0.02)].copy()
    limits = {float(row.snr_db): float(row.e_mu_max_hz_per_s) for row in rule.itertuples(index=False)}

    records: list[dict[str, float | str]] = []
    best_gate = None
    best_objective = np.inf
    for gate in args.gate_candidates:
        use = correction_valid & (receiver_snr >= float(gate))
        mu_estimate = np.where(use, mu_candidate, mu_initial)
        fd_estimate = np.where(use, fd_candidate, fd_initial)
        ratios = []
        p95_ratios = []
        for snr, limit in limits.items():
            mask = val_idx[np.isclose(snr_true[val_idx], snr)]
            mu_error = np.abs(mu_estimate[mask] - mu_true[mask])
            fd_error = np.abs(fd_estimate[mask] - fd_true[mask])
            mu_p90, mu_p95 = _tail_stats(mu_error)
            fd_p90, fd_p95 = _tail_stats(fd_error)
            gamma_limit = float(np.pi * limit * np.median(duration[mask]) ** 2)
            joint_phase = (
                2.0 * np.pi * fd_error * duration[mask]
                + np.pi * mu_error * duration[mask] ** 2
            )
            phase_p90, phase_p95 = _tail_stats(joint_phase)
            if args.gate_objective == "joint_phase_tail":
                ratios.append(phase_p90 / gamma_limit)
                p95_ratios.append(phase_p95 / gamma_limit)
            else:
                ratios.append(mu_p90 / limit)
                p95_ratios.append(mu_p95 / limit)
            records.append(
                {
                    "gate_db": float(gate), "split": "val", "snr_db": snr,
                    "mu_p90": mu_p90, "mu_p95": mu_p95,
                    "fd0_p90": fd_p90, "fd0_p95": fd_p95,
                    "joint_phase_p90": phase_p90,
                    "joint_phase_p95": phase_p95,
                    "mu_limit": limit, "phase_limit": gamma_limit,
                    "objective": str(args.gate_objective),
                }
            )
        objective = float(np.mean(ratios) + 0.1 * np.mean(p95_ratios))
        if objective < best_objective:
            best_objective = objective
            best_gate = float(gate)

    assert best_gate is not None
    selected = correction_valid & (receiver_snr >= best_gate)
    mu_refined = np.where(selected, mu_candidate, mu_initial)
    fd_refined = np.where(selected, fd_candidate, fd_initial)
    for snr, limit in limits.items():
        mask = test_idx[np.isclose(snr_true[test_idx], snr)]
        mu_error = np.abs(mu_refined[mask] - mu_true[mask])
        fd_error = np.abs(fd_refined[mask] - fd_true[mask])
        mu_p90, mu_p95 = _tail_stats(mu_error)
        fd_p90, fd_p95 = _tail_stats(fd_error)
        gamma_limit = float(np.pi * limit * np.median(duration[mask]) ** 2)
        joint_phase = (
            2.0 * np.pi * fd_error * duration[mask]
            + np.pi * mu_error * duration[mask] ** 2
        )
        phase_p90, phase_p95 = _tail_stats(joint_phase)
        records.append(
            {
                "gate_db": best_gate,
                "split": "test",
                "snr_db": snr,
                "mu_p90": mu_p90, "mu_p95": mu_p95,
                "fd0_p90": fd_p90, "fd0_p95": fd_p95,
                "joint_phase_p90": phase_p90,
                "joint_phase_p95": phase_p95,
                "mu_limit": limit, "phase_limit": gamma_limit,
                "objective": str(args.gate_objective),
            }
        )
    pd.DataFrame(records).to_csv(summary_csv, index=False)

    with h5py.File(output, "w") as out:
        out.create_dataset("pilot_mu_hat", data=mu_refined.astype(np.float32), compression="gzip")
        out.create_dataset("mu_hat", data=mu_refined.astype(np.float32), compression="gzip")
        out.create_dataset("pilot_fd0_hat", data=fd_refined.astype(np.float32), compression="gzip")
        out.create_dataset("pilot_valid", data=np.ones(n, dtype=bool), compression="gzip")
        out.create_dataset("pilot_refinement_candidate_mu", data=mu_candidate.astype(np.float32), compression="gzip")
        out.create_dataset("pilot_refinement_candidate_fd0", data=fd_candidate.astype(np.float32), compression="gzip")
        out.create_dataset("pilot_refinement_selected", data=selected, compression="gzip")
        out.create_dataset("pilot_refinement_residual_rmse", data=correction_residual_rmse.astype(np.float32), compression="gzip")
        out.create_dataset("pilot_receiver_snr_db", data=receiver_snr.astype(np.float32), compression="gzip")
        out.create_dataset("mu_error", data=(mu_refined - mu_true).astype(np.float32), compression="gzip")
        out.create_dataset("fd0_error", data=(fd_refined - fd_true).astype(np.float32), compression="gzip")
        out.create_dataset("gamma_res", data=(np.pi * np.abs(mu_refined - mu_true) * duration**2).astype(np.float32), compression="gzip")
        out.create_dataset("snr_db", data=snr_true.astype(np.float32), compression="gzip")
        out.create_dataset("T", data=duration.astype(np.float32), compression="gzip")
        out.attrs["initial_estimates"] = args.initial_estimates
        out.attrs["validation_selected_gate_db"] = best_gate
        out.attrs["method"] = f"grid initialization plus gated residual-phase {args.regression}"
        out.attrs["gate_objective"] = str(args.gate_objective)

    summary = {
        "validation_selected_gate_db": best_gate,
        "validation_objective": best_objective,
        "validation_objective_definition": (
            "mean_over_transition_snr[p90_normalized_tail + "
            "0.1*p95_normalized_tail]"
        ),
        "gate_selection_target": str(args.gate_objective),
        "regression": str(args.regression),
        "device": str(device),
        "selected_fraction_all": float(np.mean(selected)),
        "selected_fraction_val": float(np.mean(selected[val_idx])),
        "selected_fraction_test": float(np.mean(selected[test_idx])),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
