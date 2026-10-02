#!/usr/bin/env python
"""Assemble the Protocol-B front-end attribution table from frozen artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/processed/leo_7mods_joint_practical.h5"
SPLITS = ROOT / "data/splits/leo_7mods_joint_practical_splits.npz"
RESULT_ROOT = ROOT / "outputs/results/frontend_attribution"

ESTIMATORS = {
    "No compensation": (None, "", ""),
    "Coherent, equal budget": (
        ROOT / "data/features/frontend_attribution/protocol_b_pilot_coherent_equal_budget.h5",
        "pilot_mu_hat",
        "pilot_fd0_hat",
    ),
    "Coherent coarse-to-fine": (
        ROOT / "data/features/leo_7mods_joint_practical_pilot_joint.h5",
        "pilot_mu_hat",
        "pilot_fd0_hat",
    ),
    "DFRFT-equivalent acquisition": (
        ROOT / "data/features/frontend_attribution/protocol_b_pilot_dfrft_only.h5",
        "pilot_mu_hat",
        "pilot_fd0_hat",
    ),
    "DFRFT + coherent refinement": (
        ROOT / "data/features/hybrid_dfrft/protocol_b_pilot_hybrid.h5",
        "pilot_mu_hat",
        "pilot_fd0_hat",
    ),
    "Oracle compensation": (
        ROOT / "data/features/joint_practical_gamma_oracle_mu.h5",
        "mu_hat",
        "fd0_hat",
    ),
}

FROZEN_SUMMARIES = {
    "No compensation": "no_comp_seed41.summary.json",
    "Coherent, equal budget": "coherent_equal_budget_seed41.summary.json",
    "Coherent coarse-to-fine": "coherent_grid_seed41.summary.json",
    "DFRFT-equivalent acquisition": "dfrft_only_seed41.summary.json",
    "DFRFT + coherent refinement": None,
    "Oracle compensation": "oracle_seed41.summary.json",
}

CLASSIFIER_INPUTS = {
    "No compensation": (
        "data/features/frontend_attribution/protocol_b_no_comp_iqcomp.h5",
        "data/features/frontend_attribution/protocol_b_no_comp_symbol_exact.h5",
    ),
    "Coherent, equal budget": (
        "data/features/frontend_attribution/protocol_b_coherent_equal_budget_iqcomp.h5",
        "data/features/frontend_attribution/protocol_b_coherent_equal_budget_symbol_exact.h5",
    ),
    "Coherent coarse-to-fine": (
        "data/features/leo_7mods_joint_practical_iqcomp.h5",
        "data/features/exact_mixture/protocol_b_symbol_exact.h5",
    ),
    "DFRFT-equivalent acquisition": (
        "data/features/frontend_attribution/protocol_b_dfrft_only_iqcomp.h5",
        "data/features/frontend_attribution/protocol_b_dfrft_only_symbol_exact.h5",
    ),
    "DFRFT + coherent refinement": (
        "data/features/hybrid_dfrft/protocol_b_iqcomp.h5",
        "data/features/hybrid_dfrft/protocol_b_symbol_exact.h5",
    ),
    "Oracle compensation": (
        "data/features/frontend_attribution/protocol_b_oracle_iqcomp.h5",
        "data/features/frontend_attribution/protocol_b_oracle_symbol_exact.h5",
    ),
}

LATENCY_NAMES = {
    "Coherent, equal budget": "coherent_equal_budget",
    "Coherent coarse-to-fine": "coherent_current",
    "DFRFT-equivalent acquisition": "dfrft_only",
    "DFRFT + coherent refinement": "hybrid_dfrft_coherent",
}

VALIDATION_LATENCY_NAMES = {
    "Coherent, equal budget": "coherent_80x25",
    "Coherent coarse-to-fine": "coherent_40x25",
    "DFRFT-equivalent acquisition": "dfrft_only",
    "DFRFT + coherent refinement": "hybrid_dfrft_coherent",
}

CANDIDATES = {
    "No compensation": 0,
    "Coherent, equal budget": 7534,
    "Coherent coarse-to-fine": 12034,
    "DFRFT-equivalent acquisition": 51 * 2048,
    "DFRFT + coherent refinement": 51 * 2048 + 4941,
    "Oracle compensation": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(RESULT_ROOT.relative_to(ROOT)))
    parser.add_argument(
        "--timing-source",
        choices=("validation", "test"),
        default="validation",
        help="Use the isolated validation timing screen by default; the long test run may overlap other jobs.",
    )
    return parser.parse_args()


def load_accuracy(name: str) -> float | None:
    filename = FROZEN_SUMMARIES[name]
    if filename is None:
        path = ROOT / "outputs/results/hybrid_dfrft/protocol_b/proposed/seed_41_test.summary.json"
    else:
        path = RESULT_ROOT / "frozen_classifier" / filename
    if not path.exists():
        return None
    return 100.0 * float(json.loads(path.read_text(encoding="utf-8"))["accuracy"])


def main() -> None:
    args = parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    test = np.asarray(np.load(SPLITS)["test"], dtype=np.int64)
    timing_path = RESULT_ROOT / "independent_test/summary.json"
    timings = {}
    timing_scope = "independent-test subset"
    if args.timing_source == "test" and timing_path.exists():
        timings = json.loads(timing_path.read_text(encoding="utf-8"))["summary"]
    else:
        timing_scope = "validation subset"
        validation_timing_path = RESULT_ROOT / "validation/summary.json"
        timings = json.loads(validation_timing_path.read_text(encoding="utf-8"))["summary"]

    rows: list[dict[str, object]] = []
    with h5py.File(RAW, "r") as raw:
        truth_mu = np.asarray(raw["mu"][test], dtype=np.float64)
        truth_cfo = np.asarray(raw["fd0"][test], dtype=np.float64)
        duration = np.asarray(raw["T"][test], dtype=np.float64)
        for name, (path, mu_key, cfo_key) in ESTIMATORS.items():
            if name == "No compensation":
                mu_hat = np.zeros_like(truth_mu)
                cfo_hat = np.zeros_like(truth_cfo)
            else:
                if path is None or not path.exists():
                    raise FileNotFoundError(path)
                with h5py.File(path, "r") as source:
                    mu_hat = np.asarray(source[mu_key][test], dtype=np.float64)
                    cfo_hat = np.asarray(source[cfo_key][test], dtype=np.float64)
            em = np.abs(truth_mu - mu_hat)
            ef = np.abs(truth_cfo - cfo_hat)
            eta = 2.0 * np.pi * ef * duration + np.pi * em * duration**2
            latency_name = (
                LATENCY_NAMES.get(name)
                if timing_scope == "independent-test subset"
                else VALIDATION_LATENCY_NAMES.get(name)
            )
            latency = None
            if latency_name in timings:
                latency = float(timings[latency_name]["median_cpu_ms"])
            elif name == "No compensation":
                latency = 0.0
            rows.append(
                {
                    "front_end": name,
                    "scope": "full independent test (7700 frames)",
                    "search_scores_per_frame": CANDIDATES[name],
                    "rate_p95_hz_per_s": float(np.quantile(em, 0.95)),
                    "cfo_p95_hz": float(np.quantile(ef, 0.95)),
                    "eta_p95": float(np.quantile(eta, 0.95)),
                    "frozen_seed41_accuracy_percent": load_accuracy(name),
                    "median_cpu_ms": latency,
                    "median_gpu_ms": None,
                }
            )

    exhaustive_path = RESULT_ROOT / "exhaustive_subset/summary.json"
    subset_comparison: dict[str, object] = {}
    if exhaustive_path.exists():
        exact = json.loads(exhaustive_path.read_text(encoding="utf-8"))
        rows.insert(
            3,
            {
                "front_end": "Global coherent 5 Hz/s x 1 Hz",
                "scope": "fixed stratified test subset (77 frames)",
                "search_scores_per_frame": int(exact["joint_candidates_per_frame"]),
                "rate_p95_hz_per_s": float(exact["rate_p95_hz_per_s"]),
                "cfo_p95_hz": float(exact["cfo_p95_hz"]),
                "eta_p95": float(exact["eta_p95"]),
                "frozen_seed41_accuracy_percent": None,
                "median_cpu_ms": None,
                "median_gpu_ms": float(exact["median_gpu_ms_per_frame"]),
            },
        )
        subset_index = pd.read_csv(
            RESULT_ROOT / "exhaustive_subset/frame_results.csv"
        )["index"].to_numpy(np.int64)
        with h5py.File(RAW, "r") as raw:
            subset_mu = np.asarray(raw["mu"][subset_index], dtype=np.float64)
            subset_cfo = np.asarray(raw["fd0"][subset_index], dtype=np.float64)
            subset_duration = np.asarray(raw["T"][subset_index], dtype=np.float64)
            for name, (path, mu_key, cfo_key) in ESTIMATORS.items():
                if name in {"No compensation", "Oracle compensation"}:
                    continue
                assert path is not None
                with h5py.File(path, "r") as source:
                    em = np.abs(subset_mu - np.asarray(source[mu_key][subset_index]))
                    ef = np.abs(subset_cfo - np.asarray(source[cfo_key][subset_index]))
                eta = 2.0 * np.pi * ef * subset_duration + np.pi * em * subset_duration**2
                subset_comparison[name] = {
                    "frames": int(subset_index.size),
                    "rate_p95_hz_per_s": float(np.quantile(em, 0.95)),
                    "cfo_p95_hz": float(np.quantile(ef, 0.95)),
                    "eta_p95": float(np.quantile(eta, 0.95)),
                }
        subset_comparison["Global coherent 5 Hz/s x 1 Hz"] = {
            "frames": int(exact["frames"]),
            "rate_p95_hz_per_s": float(exact["rate_p95_hz_per_s"]),
            "cfo_p95_hz": float(exact["cfo_p95_hz"]),
            "eta_p95": float(exact["eta_p95"]),
        }

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "task_table.csv", index=False)
    hybrid_predictions = pd.read_csv(
        ROOT / "outputs/results/hybrid_dfrft/protocol_b/proposed/seed_41_test.csv"
    ).set_index("index")
    hybrid_correct = hybrid_predictions["correct"].astype(float)
    rng = np.random.default_rng(20260905)
    paired: dict[str, object] = {}
    for name, filename in FROZEN_SUMMARIES.items():
        if filename is None:
            continue
        path = RESULT_ROOT / "frozen_classifier" / filename.replace(".summary.json", ".csv")
        if not path.exists():
            continue
        candidate = pd.read_csv(path).set_index("index").loc[hybrid_correct.index]
        delta = 100.0 * (hybrid_correct.to_numpy() - candidate["correct"].to_numpy(float))
        draw = rng.integers(0, delta.size, size=(5000, delta.size))
        boot = delta[draw].mean(axis=1)
        paired[name] = {
            "hybrid_minus_candidate_pp": float(delta.mean()),
            "paired_bootstrap_95_ci_pp": [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ],
            "hybrid_only_correct": int(np.count_nonzero(delta > 0.0)),
            "candidate_only_correct": int(np.count_nonzero(delta < 0.0)),
            "bootstrap_replicates": 5000,
        }

    payload = {
        "table": rows,
        "paired_accuracy_vs_hybrid": paired,
        "matched_exhaustive_subset": subset_comparison,
        "artifact_lineage": {
            "raw_data": str(RAW.relative_to(ROOT)).replace("\\", "/"),
            "splits": str(SPLITS.relative_to(ROOT)).replace("\\", "/"),
            "frozen_checkpoint": "outputs/checkpoints/hybrid_dfrft/protocol_b/proposed/seed_41/best.pt",
            "classifier_inputs": {
                name: {"iq": paths[0], "descriptor": paths[1]}
                for name, paths in CLASSIFIER_INPUTS.items()
            },
        },
        "accuracy_intervention": (
            "One hybrid-trained seed-41 checkpoint is frozen; only the front-end "
            "I/Q and 48-D descriptor artifacts are replaced."
        ),
        "timing_policy": (
            f"Full-set methods use scalar CPU medians on the {timing_scope}. The exhaustive "
            "row is a CUDA subset timing and is not compared numerically with CPU timing."
        ),
    }
    (output / "task_table.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
