#!/usr/bin/env python
"""Evaluate an idealized symbol-likelihood reference for the controlled AWGN setup.

The reference uses oracle Doppler-rate compensation, pilot-aided timing and
complex-gain estimation, the simulated sample-domain SNR, and the complete
candidate constellation set.  It is an upper reference for the present
receiver assumptions, not a Bayes bound under unknown synchronization or
channel parameters.
"""

from __future__ import annotations

import argparse
import importlib.util
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
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from src.plotting.common import IEEE_TRANS_PALETTE, boxed_legend, format_ieee_axis, set_panel_aspect_10_9, set_ieee_trans_style

from src.drc_hoc.compensation import compensate_full_doppler
from src.signal.modulation import get_constellation
from src.signal.pulse_shape import rrc_filter
from src.utils.math_utils import to_complex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an oracle-compensated symbol-likelihood reference."
    )
    parser.add_argument(
        "--raw-data",
        default="data/processed/leo_7mods_snr_balanced_independent_test_pilot.h5",
    )
    parser.add_argument(
        "--proposed-result",
        default=(
            "outputs/results/paper_baselines_independent_test/"
            "proposed_iq_evm_test.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/results/likelihood_reference_all_symbols",
    )
    parser.add_argument(
        "--figure",
        default=(
            "outputs/figures/paper/"
            "likelihood_reference_all_symbols_accuracy_vs_snr.pdf"
        ),
    )
    parser.add_argument("--samples-per-symbol", type=int, default=8)
    parser.add_argument("--rrc-beta", type=float, default=0.35)
    parser.add_argument("--rrc-span", type=int, default=8)
    parser.add_argument("--timing-phases", type=int, nargs="*", default=[0])
    parser.add_argument("--symbol-trim", type=int, default=8)
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=0,
        help="Maximum payload symbols per frame; use 0 for all available symbols.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_symbol_helpers():
    path = PROJECT_ROOT / "scripts" / "10_precompute_symbol_constellation.py"
    spec = importlib.util.spec_from_file_location("symbol_feature_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load symbol helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decode(values: np.ndarray) -> list[str]:
    return [
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in values
    ]


def _candidate_log_likelihood(
    symbols: np.ndarray,
    constellation: np.ndarray,
    noise_variance: float,
) -> float:
    z = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    points = np.asarray(constellation, dtype=np.complex128).reshape(-1)
    if z.size == 0 or points.size == 0:
        return -np.inf

    variance = max(float(noise_variance), 1e-8)
    scaled = -np.abs(z[:, None] - points[None, :]) ** 2 / variance
    row_max = np.max(scaled, axis=1)
    log_mixture = row_max + np.log(
        np.mean(np.exp(scaled - row_max[:, None]), axis=1) + 1e-300
    )
    return float(np.sum(log_mixture))


def _wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = correct / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return float(center - half), float(center + half)


def _aggregate(df: pd.DataFrame, estimate: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for snr_db, group in df.groupby("snr_db", sort=True):
        total = int(len(group))
        correct = int(group["correct"].sum())
        low, high = _wilson_interval(correct, total)
        rows.append(
            {
                "estimate": estimate,
                "snr_db": float(snr_db),
                "count": total,
                "accuracy": correct / total if total else np.nan,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def _label_names(raw_data: str) -> list[str]:
    with h5py.File(raw_data, "r") as raw:
        return _decode(raw["label_names"][:])


def _plot_reference_gap(
    reference: pd.DataFrame,
    proposed: pd.DataFrame,
    label_names: list[str],
    output: Path,
) -> None:
    """Plot class- and SNR-resolved practical-to-reference accuracy differences."""

    output.parent.mkdir(parents=True, exist_ok=True)
    set_ieee_trans_style()
    fig, ax = plt.subplots(figsize=(3.45, 3.105))

    columns = ["index", "snr_db", "label", "correct"]
    merged = reference[columns].merge(
        proposed[columns],
        on=["index", "snr_db", "label"],
        suffixes=("_reference", "_practical"),
        validate="one_to_one",
    )
    if len(merged) != len(reference) or len(merged) != len(proposed):
        raise ValueError("Likelihood-reference and practical-receiver results do not align.")

    snr_points = np.sort(merged["snr_db"].unique().astype(float))
    labels = np.arange(len(label_names), dtype=int)
    gap = np.full((len(labels), len(snr_points)), np.nan, dtype=np.float64)
    for row, label in enumerate(labels):
        for column, snr_db in enumerate(snr_points):
            subset = merged[
                (merged["label"] == label)
                & np.isclose(merged["snr_db"].to_numpy(dtype=float), snr_db)
            ]
            if not subset.empty:
                gap[row, column] = 100.0 * (
                    float(subset["correct_reference"].mean())
                    - float(subset["correct_practical"].mean())
                )

    span = np.diff(snr_points)
    snr_edges = np.concatenate(
        ([snr_points[0] - span[0] / 2.0], (snr_points[:-1] + snr_points[1:]) / 2.0, [snr_points[-1] + span[-1] / 2.0])
    )
    max_gap = max(5.0, 5.0 * np.ceil(float(np.nanmax(np.abs(gap))) / 5.0))
    mesh = ax.pcolormesh(
        snr_edges,
        np.arange(len(labels) + 1, dtype=float) - 0.5,
        gap,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-max_gap, vcenter=0.0, vmax=max_gap),
        shading="auto",
        rasterized=True,
    )
    tick_indices = np.arange(0, len(snr_points), 2, dtype=int)
    if tick_indices[-1] != len(snr_points) - 1:
        tick_indices = np.append(tick_indices, len(snr_points) - 1)
    ax.set_xticks(snr_points[tick_indices])
    ax.set_xticklabels([f"{snr_points[index]:g}" for index in tick_indices])
    ax.set_yticks(labels)
    ax.set_yticklabels(label_names)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("True modulation")
    ax.set_xlim(float(snr_edges[0]), float(snr_edges[-1]))
    ax.set_ylim(float(len(labels) - 0.5), -0.5)
    format_ieee_axis(ax)
    set_panel_aspect_10_9(ax)
    ax.grid(False)
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.025, fraction=0.052)
    colorbar.set_label(r"$\Delta$ accuracy (pp)", fontsize=8.0)
    colorbar.ax.tick_params(labelsize=7.5, direction="out")
    fig.tight_layout(pad=0.45)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    sample_csv = output_dir / "symbol_likelihood_reference.csv"
    curve_csv = output_dir / "accuracy_vs_snr.csv"
    summary_json = output_dir / "summary.json"
    if sample_csv.exists() and not args.overwrite:
        print(f"Reusing {sample_csv}")
        samples = pd.read_csv(sample_csv)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        helpers = _load_symbol_helpers()
        taps = rrc_filter(
            beta=float(args.rrc_beta),
            span=int(args.rrc_span),
            sps=int(args.samples_per_symbol),
        )
        timing_phases = tuple(int(x) % int(args.samples_per_symbol) for x in args.timing_phases)
        rows: list[dict[str, float | int | str]] = []

        with h5py.File(args.raw_data, "r") as raw:
            label_names = _decode(raw["label_names"][:])
            constellations = {
                name: get_constellation(name).astype(np.complex128) for name in label_names
            }
            n = int(raw["iq"].shape[0])
            for i in range(n):
                fs = float(raw["fs"][i])
                mu = float(raw["mu"][i])
                fd0 = float(raw["fd0"][i]) if "fd0" in raw else 0.0
                received = to_complex(raw["iq"][i])
                compensated = compensate_full_doppler(
                    received,
                    sample_rate_hz=fs,
                    fd0_hat_hz=fd0,
                    mu_hat_hz_per_s=mu,
                    centered=False,
                )
                matched = np.convolve(compensated, taps, mode="same")

                best = (np.inf, 0, 1.0 + 0.0j)
                for timing in timing_phases:
                    gain, pilot_evm, pilot_count = helpers._estimate_gain_and_pilot_evm(
                        matched,
                        raw["pilot_indices"][i],
                        raw["pilot_symbols"][i],
                        timing,
                        int(args.samples_per_symbol),
                    )
                    if pilot_count > 0 and pilot_evm < best[0]:
                        best = (float(pilot_evm), int(timing), complex(gain))

                pilot_evm, timing, gain = best
                symbols = helpers._extract_equalized_symbols(
                    matched,
                    gain,
                    timing,
                    int(args.samples_per_symbol),
                    raw["pilot_indices"][i],
                    symbol_trim=int(args.symbol_trim),
                    max_symbols=int(args.max_symbols),
                )
                snr_db = float(raw["snr_db"][i])
                noise_variance = 10.0 ** (-snr_db / 10.0) / int(args.samples_per_symbol)
                scores = np.asarray(
                    [
                        _candidate_log_likelihood(
                            symbols,
                            constellations[name],
                            noise_variance,
                        )
                        for name in label_names
                    ],
                    dtype=np.float64,
                )
                pred = int(np.argmax(scores))
                label = int(raw["label"][i])
                rows.append(
                    {
                        "index": i,
                        "snr_db": snr_db,
                        "label": label,
                        "pred": pred,
                        "correct": int(pred == label),
                        "symbol_count": int(symbols.size),
                        "pilot_evm": pilot_evm,
                        "noise_variance": noise_variance,
                    }
                )
                if (i + 1) % 500 == 0 or i + 1 == n:
                    print(f"Likelihood reference: {i + 1}/{n}")

        samples = pd.DataFrame(rows)
        samples.to_csv(sample_csv, index=False)

    reference_curve = _aggregate(samples, "Idealized likelihood reference")
    curves = [reference_curve]
    proposed_samples: pd.DataFrame | None = None
    proposed_path = Path(args.proposed_result)
    if proposed_path.exists():
        proposed_samples = pd.read_csv(proposed_path)
        if "correct" not in proposed_samples:
            proposed_samples["correct"] = (
                proposed_samples["label"] == proposed_samples["pred"]
            ).astype(int)
        curves.insert(0, _aggregate(proposed_samples, "Proposed receiver"))
    combined = pd.concat(curves, ignore_index=True)
    combined.to_csv(curve_csv, index=False)
    if proposed_samples is not None:
        _plot_reference_gap(
            samples,
            proposed_samples,
            _label_names(args.raw_data),
            Path(args.figure),
        )

    reference_accuracy = float(samples["correct"].mean())
    proposed_accuracy = np.nan
    if "Proposed receiver" in set(combined["estimate"]):
        proposed_rows = combined[combined["estimate"] == "Proposed receiver"]
        proposed_accuracy = float(
            np.average(proposed_rows["accuracy"], weights=proposed_rows["count"])
        )
    summary = {
        "raw_data": args.raw_data,
        "num_samples": int(len(samples)),
        "reference_accuracy": reference_accuracy,
        "proposed_accuracy": proposed_accuracy,
        "accuracy_gap_percentage_points": 100.0 * (reference_accuracy - proposed_accuracy),
        "assumptions": [
            "oracle Doppler-rate and constant-frequency compensation",
            "pilot-aided timing and complex-gain estimation",
            "true simulated sample-domain SNR",
            "flat AWGN and equiprobable candidate symbols",
        ],
        "interpretation": (
            "Idealized likelihood reference for the controlled simulation only; "
            "not a strict Bayes bound under unknown receiver parameters."
        ),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
