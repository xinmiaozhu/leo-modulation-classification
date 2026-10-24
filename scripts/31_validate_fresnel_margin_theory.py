#!/usr/bin/env python
"""Validate the Fresnel attenuation formula and its finite-sample bound.

This script is a deterministic theory check; it does not use train, validation,
calibration, or test frames.  It compares the closed-form continuous response
with the exact finite-N sum used by the manuscript and records the induced
constant-energy I/Q perturbation radius.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.physics.gamma_metric import (
    attenuation_near_zero_lower_bound,
    constant_energy_iq_perturbation,
    finite_sample_attenuation_error_bound,
    fresnel_attenuation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=8192)
    parser.add_argument("--gamma-max", type=float, default=8.0)
    parser.add_argument("--gamma-step", type=float, default=0.01)
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument(
        "--output-dir",
        default="outputs/results/theory/fresnel_margin",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 2:
        raise ValueError("--num-samples must be at least 2.")
    if args.gamma_max <= 0.0 or args.gamma_step <= 0.0:
        raise ValueError("The gamma range must be positive.")

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    gamma = np.arange(0.0, args.gamma_max + 0.5 * args.gamma_step, args.gamma_step)
    u = np.arange(args.num_samples, dtype=np.float64) / float(args.num_samples)
    rows: list[dict[str, float | int]] = []
    maximum_ratio = 0.0

    for k in args.orders:
        continuous = np.asarray(fresnel_attenuation(gamma, k=k))
        finite = np.asarray(
            [np.mean(np.exp(1j * float(k) * value * u**2)) for value in gamma]
        )
        error = np.abs(finite - continuous)
        bound = np.asarray(
            finite_sample_attenuation_error_bound(
                gamma,
                k=k,
                num_samples=args.num_samples,
            )
        )
        positive = bound > 0.0
        if np.any(positive):
            maximum_ratio = max(maximum_ratio, float(np.max(error[positive] / bound[positive])))
        lower = np.asarray(attenuation_near_zero_lower_bound(gamma, k=k))
        for index, value in enumerate(gamma):
            rows.append(
                {
                    "gamma": float(value),
                    "k": int(k),
                    "fresnel_real": float(np.real(continuous[index])),
                    "fresnel_imag": float(np.imag(continuous[index])),
                    "fresnel_magnitude": float(np.abs(continuous[index])),
                    "finite_real": float(np.real(finite[index])),
                    "finite_imag": float(np.imag(finite[index])),
                    "finite_error": float(error[index]),
                    "finite_error_bound": float(bound[index]),
                    "near_zero_lower_bound": float(lower[index]),
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "fresnel_finite_n_validation.csv", index=False)
    iq_table = pd.DataFrame(
        {
            "gamma": gamma,
            "constant_energy_iq_perturbation": constant_energy_iq_perturbation(gamma),
        }
    )
    iq_table.to_csv(output_dir / "iq_perturbation_radius.csv", index=False)

    summary = {
        "num_samples": int(args.num_samples),
        "orders": [int(value) for value in args.orders],
        "gamma_min": 0.0,
        "gamma_max": float(args.gamma_max),
        "gamma_step": float(args.gamma_step),
        "maximum_finite_error": float(table["finite_error"].max()),
        "maximum_error_to_bound_ratio": float(maximum_ratio),
        "all_finite_errors_within_bound": bool(
            np.all(table["finite_error"] <= table["finite_error_bound"] + 1e-12)
        ),
        "outputs": {
            "attenuation": str((output_dir / "fresnel_finite_n_validation.csv").relative_to(ROOT)),
            "iq_radius": str((output_dir / "iq_perturbation_radius.csv").relative_to(ROOT)),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
