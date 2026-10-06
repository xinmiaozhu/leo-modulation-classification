#!/usr/bin/env python
"""Run the exhaustive 5-Hz/s by 1-Hz coherent grid on a fixed test subset.

The exhaustive grid contains 1,758,297 CFO--rate pairs per frame.  It is
therefore evaluated on a prespecified class--SNR-stratified subset and is not
presented as an equal-cost receiver.
"""

from __future__ import annotations

import argparse
import importlib.util
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
import torch

from src.drc_hoc.pilot_estimator import PilotMuEstimatorConfig, make_grid
from src.signal.pulse_shape import rrc_filter


def load_cuda_function():
    path = ROOT / "scripts/06_precompute_pilot_mu.py"
    spec = importlib.util.spec_from_file_location("precompute_pilot_mu_08", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._estimate_coherent_cuda_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-data", default="data/processed/leo_7mods_joint_practical.h5")
    p.add_argument("--splits", default="data/splits/leo_7mods_joint_practical_splits.npz")
    p.add_argument("--samples-per-class-snr", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--output-dir", default="outputs/results/frontend_attribution/exhaustive_subset")
    return p.parse_args()


def select_indices(candidates: np.ndarray, labels: np.ndarray, snr: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in np.unique(labels[candidates]):
        for level in np.unique(snr[candidates]):
            cell = candidates[(labels[candidates] == label) & np.isclose(snr[candidates], level)]
            selected.extend(rng.choice(cell, count, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise SystemExit("CUDA is required for the exhaustive subset experiment")
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    split = np.load(ROOT / args.splits)
    test = np.asarray(split["test"], dtype=np.int64)
    with h5py.File(ROOT / args.raw_data, "r") as raw:
        labels = np.asarray(raw["label"][:], dtype=np.int64)
        snr = np.asarray(raw["snr_db"][:], dtype=np.float64)
        selected = select_indices(test, labels, snr, args.samples_per_class_snr, args.seed)
        iq = np.asarray(raw["iq"][selected], dtype=np.float32)
        pilot_indices = np.asarray(raw["pilot_indices"][selected], dtype=np.int64)
        pilot_symbols = np.asarray(raw["pilot_symbols"][selected], dtype=np.complex64)
        truth_mu = np.asarray(raw["mu"][selected], dtype=np.float64)
        truth_cfo = np.asarray(raw["fd0"][selected], dtype=np.float64)
        duration = np.asarray(raw["T"][selected], dtype=np.float64)

    cfg = PilotMuEstimatorConfig(
        sample_rate_hz=200_000.0,
        samples_per_symbol=8,
        rrc_beta=0.35,
        rrc_span=8,
        timing_phases=(0,),
        mu_min=-8160.0,
        mu_max=-180.0,
        coarse_step_hz_per_s=5.0,
        fine_radius_hz_per_s=0.0,
        fine_step_hz_per_s=5.0,
        fd0_min_hz=-550.0,
        fd0_max_hz=550.0,
        fd0_step_hz=1.0,
        fine_fd0_radius_hz=0.0,
        fine_fd0_step_hz=1.0,
        pilot_weighting="coherent",
    )
    function = load_cuda_function()
    device = torch.device(args.device)
    taps = rrc_filter(beta=cfg.rrc_beta, span=cfg.rrc_span, sps=cfg.samples_per_symbol)
    results = []
    elapsed = []
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(selected), batch_size):
        stop = min(start + batch_size, len(selected))
        torch.cuda.synchronize(device)
        tic = time.perf_counter()
        mu, cfo, _, _, _ = function(
            iq[start:stop], pilot_indices[start:stop], pilot_symbols[start:stop],
            config=cfg, device=device, rrc_taps=taps,
        )
        torch.cuda.synchronize(device)
        per_frame_ms = 1000.0 * (time.perf_counter() - tic) / (stop - start)
        elapsed.extend([per_frame_ms] * (stop - start))
        for local, position in enumerate(range(start, stop)):
            em = truth_mu[position] - mu[local]
            ef = truth_cfo[position] - cfo[local]
            eta = 2 * np.pi * abs(ef) * duration[position] + np.pi * abs(em) * duration[position] ** 2
            results.append({
                "index": int(selected[position]), "label": int(labels[selected[position]]),
                "snr_db": float(snr[selected[position]]), "mu_hat_hz_per_s": float(mu[local]),
                "cfo_hat_hz": float(cfo[local]), "abs_mu_error_hz_per_s": float(abs(em)),
                "abs_cfo_error_hz": float(abs(ef)), "eta": float(eta),
                "gpu_ms_per_frame": float(per_frame_ms),
            })
        print(f"{stop}/{len(selected)}", flush=True)

    frame = pd.DataFrame(results)
    frame.to_csv(output / "frame_results.csv", index=False)
    em = frame.abs_mu_error_hz_per_s.to_numpy(float)
    ef = frame.abs_cfo_error_hz.to_numpy(float)
    eta = frame.eta.to_numpy(float)
    summary = {
        "scope": "fixed stratified independent-test subset",
        "frames": int(len(frame)),
        "rate_grid_points": int(len(make_grid(cfg.mu_min, cfg.mu_max, cfg.coarse_step_hz_per_s))),
        "cfo_grid_points": int(len(make_grid(cfg.fd0_min_hz, cfg.fd0_max_hz, cfg.fd0_step_hz))),
        "joint_candidates_per_frame": int(
            len(make_grid(cfg.mu_min, cfg.mu_max, cfg.coarse_step_hz_per_s))
            * len(make_grid(cfg.fd0_min_hz, cfg.fd0_max_hz, cfg.fd0_step_hz))
        ),
        "rate_rmse_hz_per_s": float(np.sqrt(np.mean(em**2))),
        "rate_p95_hz_per_s": float(np.quantile(em, .95)),
        "cfo_rmse_hz": float(np.sqrt(np.mean(ef**2))),
        "cfo_p95_hz": float(np.quantile(ef, .95)),
        "eta_p95": float(np.quantile(eta, .95)),
        "median_gpu_ms_per_frame": float(np.median(elapsed)),
        "warning": "GPU subset timing is not compared with the CPU latency column.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
