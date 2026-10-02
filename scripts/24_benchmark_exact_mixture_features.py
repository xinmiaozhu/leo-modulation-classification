#!/usr/bin/env python
"""Benchmark receiver-side exact-mixture descriptor construction on real frames."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import pandas as pd

from src.signal.modulation import get_constellation
from src.signal.pulse_shape import rrc_filter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-data", required=True, help="HDF5 containing compensated I/Q.")
    p.add_argument("--symbol-feature-data", required=True, help="Exact-mixture symbol HDF5.")
    p.add_argument("--splits", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--frames", type=int, default=128)
    p.add_argument("--warmup-passes", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=616161)
    p.add_argument(
        "--output-root",
        default="outputs/results/exact_mixture/latency",
    )
    return p.parse_args()


def load_helpers():
    path = ROOT / "scripts" / "10_precompute_symbol_constellation.py"
    spec = importlib.util.spec_from_file_location("symbol_latency_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode(values: np.ndarray) -> tuple[str, ...]:
    return tuple(x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values)


def statistics(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.quantile(values, 0.90)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "p99_ms": float(np.quantile(values, 0.99)),
        "max_ms": float(np.max(values)),
    }


def main() -> None:
    args = parse_args()
    helpers = load_helpers()
    feature_path = Path(args.feature_data)
    symbol_path = Path(args.symbol_feature_data)
    split_path = Path(args.splits)
    for path in (feature_path, symbol_path, split_path):
        if not path.exists():
            raise FileNotFoundError(path)

    with np.load(split_path) as split_file:
        indices = np.asarray(split_file[args.split], dtype=np.int64)

    with h5py.File(symbol_path, "r") as symbol:
        config = json.loads(symbol.attrs["config_json"])
        if str(config.get("candidate_score_mode")) != "exact_mixture":
            raise ValueError("The benchmark requires an exact-mixture symbol artifact.")
        valid = np.asarray(symbol["symbol_valid"][indices], dtype=bool)
    indices = indices[valid]
    rng = np.random.default_rng(args.seed)
    if indices.size > args.frames:
        indices = np.sort(rng.choice(indices, size=args.frames, replace=False))

    sps = int(config["samples_per_symbol"])
    taps = rrc_filter(
        beta=float(config["rrc_beta"]),
        span=int(config["rrc_span"]),
        sps=sps,
    )
    phase_grid_size = int(config.get("phase_grid_size", 1))
    phase_grid = (
        np.asarray([0.0], dtype=np.float64)
        if phase_grid_size <= 1
        else np.linspace(0.0, 2.0 * np.pi, phase_grid_size, endpoint=False)
    )

    records: list[dict[str, object]] = []
    with h5py.File(feature_path, "r") as feature, h5py.File(symbol_path, "r") as symbol:
        modulations = tuple(name.upper() for name in decode(symbol["candidate_modulations"][:]))
        constellations = {
            name: get_constellation(name).astype(np.complex128) for name in modulations
        }
        for index in indices:
            iq = np.asarray(feature["iq_comp"][index], dtype=np.float64)
            frame = iq[0] + 1j * iq[1]
            matched = np.convolve(frame, taps, mode="same")
            gain = complex(
                float(symbol["complex_gain_real"][index]),
                float(symbol["complex_gain_imag"][index]),
            )
            symbols = helpers._extract_equalized_symbols(
                matched,
                gain,
                int(symbol["timing_phase"][index]),
                sps,
                np.asarray(symbol["pilot_indices"][index]),
                symbol_trim=int(config["symbol_trim"]),
                max_symbols=int(config["max_symbols"]),
            )
            symbols = helpers._prepare_symbols(symbols, center=True, power_normalize=True)
            records.append(
                {
                    "symbols": symbols,
                    "evm": np.asarray(symbol["all_candidate_evm"][index], dtype=np.float64),
                    "phase_best": np.asarray(symbol["phase_best"][index], dtype=np.float64),
                    "pilot_evm": float(symbol["pilot_evm"][index]),
                    "snr_db": float(symbol["penalty_snr_db"][index]),
                }
            )

    common_kwargs = {
        "complexity_penalty_lambda": float(config["complexity_penalty_lambda"]),
        "complexity_penalty_snr_threshold": float(config["complexity_penalty_snr_threshold"]),
        "complexity_penalty_snr_scale": float(config["complexity_penalty_snr_scale"]),
        "apsk_ring_penalty_lambda": float(config["apsk_ring_penalty_lambda"]),
        "apsk_phase_weight": float(config["apsk_phase_weight"]),
        "apsk_occupancy_weight": float(config["apsk_occupancy_weight"]),
    }

    def compute(record: dict[str, object], mode: str, full: bool) -> np.ndarray:
        symbols = np.asarray(record["symbols"])
        if full:
            evm, phase_best = helpers._score_evm_all_candidates(
                symbols, modulations, constellations, phase_grid
            )
        else:
            evm = np.asarray(record["evm"])
            phase_best = np.asarray(record["phase_best"])
        vector, _ = helpers._evm_feature_vector(
            evm,
            modulations,
            float(record["pilot_evm"]),
            symbols,
            constellations,
            phase_best,
            snr_db=float(record["snr_db"]),
            candidate_score_mode=mode,
            **common_kwargs,
        )
        return np.asarray(vector)

    for _ in range(max(args.warmup_passes, 0)):
        for record in records:
            compute(record, "exact_mixture", True)
            compute(record, "logistic", True)

    rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    dimensions: dict[str, int] = {}
    for scope, full in (("vector_only", False), ("candidate_plus_vector", True)):
        for mode in ("logistic", "exact_mixture"):
            elapsed: list[float] = []
            for repeat in range(max(args.repeats, 1)):
                for frame_number, record in enumerate(records):
                    start = time.perf_counter_ns()
                    vector = compute(record, mode, full)
                    duration_ms = (time.perf_counter_ns() - start) / 1.0e6
                    elapsed.append(duration_ms)
                    raw_rows.append(
                        {
                            "scope": scope,
                            "mode": mode,
                            "repeat": repeat,
                            "frame_number": frame_number,
                            "latency_ms": duration_ms,
                        }
                    )
                    dimensions[mode] = int(vector.size)
            row: dict[str, object] = {"scope": scope, "mode": mode}
            row.update(statistics(np.asarray(elapsed, dtype=np.float64)))
            rows.append(row)

    table = pd.DataFrame(rows)
    raw = pd.DataFrame(raw_rows)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "summary.csv", index=False)
    raw.to_csv(output / "measurements.csv", index=False)

    comparisons: dict[str, dict[str, float]] = {}
    for scope in table["scope"].unique():
        group = table[table["scope"] == scope].set_index("mode")
        exact = float(group.loc["exact_mixture", "median_ms"])
        logistic = float(group.loc["logistic", "median_ms"])
        comparisons[str(scope)] = {
            "exact_minus_logistic_median_ms": exact - logistic,
            "exact_over_logistic_median": exact / logistic,
        }
    metadata = {
        "feature_data": str(feature_path),
        "symbol_feature_data": str(symbol_path),
        "split": args.split,
        "frames": len(records),
        "repeats": int(args.repeats),
        "measurements_per_condition": len(records) * int(args.repeats),
        "max_symbols": int(config["max_symbols"]),
        "candidate_modulations": list(modulations),
        "feature_dimensions": dimensions,
        "timing_scope": {
            "vector_only": "geometry plus candidate-score vector from cached nearest-point EVM/phase",
            "candidate_plus_vector": "nearest-point candidate scoring plus complete descriptor vector",
            "excluded": "matched filtering, gain/timing estimation, symbol extraction, HDF5 I/O, and network inference",
        },
        "cpu": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "comparisons": comparisons,
    }
    (output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
