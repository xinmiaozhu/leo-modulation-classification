#!/usr/bin/env python
"""Plot the formal seven-method matched five-seed baseline comparison."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.plotting.common import (
    IEEE_TRANS_PALETTE, boxed_legend, format_ieee_axis,
    set_ieee_trans_style, set_panel_aspect_10_9,
)

METHODS = [
    ("cnn2", "C1 CNN2", "black", "o"),
    ("mcnet", "C2 MCNet", "gray", "s"),
    ("cnn_lstm_dual", "C3 CNN-LSTM", "red", "^"),
    ("satellite_cnn", "C4 IQCNet", "orange", "D"),
    ("iqformer", "C5 IQFormer", "blue", "v"),
    ("starnet", "C6 STARNet", "purple", "X"),
    ("proposed", "Proposed I/Q-Constellation", "green", "P"),
]


def load_runner():
    spec = importlib.util.spec_from_file_location("baseline_runner", ROOT / "scripts/16_run_hybrid_dfrft_seeds.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.ROOT = ROOT
    return runner


def matched_seed_curves(protocol):
    runner = load_runner()
    cfg = runner.protocol_inputs(protocol)
    curves, manifest = [], []
    for method, label, *_ in METHODS:
        for seed in runner.SEEDS:
            path = runner.prediction_path(protocol, method, seed)
            runner.validate_test_predictions(path, cfg)
            df = pd.read_csv(path)
            curve = df.groupby("snr_db").correct.agg(sample_count="size", accuracy="mean").reset_index()
            curve["accuracy_percent"] = 100 * curve.accuracy
            curve["method"], curve["display_label"], curve["seed"] = method, label, seed
            curves.append(curve)
            manifest.append(dict(method=method, seed=seed, eval_csv=path.relative_to(ROOT).as_posix()))
    individual = pd.concat(curves, ignore_index=True)
    aggregate = aggregate_seed_curves(individual, runner.SEEDS)
    return individual, aggregate, manifest


def aggregate_seed_curves(individual, seeds):
    """SD is across independent training seeds, never across pooled test frames."""
    expected_snr = set(individual.snr_db.unique())
    if individual.duplicated(["method", "snr_db", "seed"]).any():
        raise ValueError("Duplicate method/SNR/seed curve rows")
    for method in individual.method.unique():
        for snr in expected_snr:
            group = individual[(individual.method == method) & (individual.snr_db == snr)]
            if set(group.seed) != set(seeds):
                raise ValueError(f"Missing matched seed: {method}, SNR={snr}")
            if group.sample_count.nunique() != 1:
                raise ValueError(f"Test-frame counts differ across seeds: {method}, SNR={snr}")
    return individual.groupby(["method", "display_label", "snr_db"]).accuracy_percent.agg(
        seed_count="count", mean_percent="mean", sample_sd_percent="std").reset_index()


def plot_five_seed(protocol, output_dir=None):
    # Load current publication code even when the training parent predates edits.
    # Require complete results before writing any formal figure or table.
    runner = load_runner()
    runner.five_seed_report(require_complete=True)
    individual, aggregate, manifest = matched_seed_curves(protocol)
    source = ROOT / "outputs/results/baseline_comparison"
    out = Path(output_dir) if output_dir is not None else ROOT / "outputs/figures/paper"
    source.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    name = f"paper_baseline_protocol_{protocol.lower()}_accuracy_vs_snr"
    set_ieee_trans_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.15))
    # Shift the zoom panel slightly down and right for the revised layout.
    inset = ax.inset_axes([0.54, 0.50, 0.44, 0.35])
    for method, label, color_key, marker in METHODS:
        curve = aggregate[aggregate.method == method].sort_values("snr_db")
        color = IEEE_TRANS_PALETTE[color_key]
        for panel in (ax, inset):
            panel.plot(curve.snr_db, curve.mean_percent, label=label, color=color,
                       marker=marker, markersize=3 if panel is ax else 2.2,
                       markerfacecolor="white", markeredgewidth=0.75,
                       linewidth=1.45 if method == "proposed" else 1.15)
    snr = np.sort(aggregate.snr_db.unique())
    ax.set(xlim=(snr[0] - 0.5, snr[-1] + 0.5), ylim=(0, 101.5), xticks=snr,
           yticks=np.arange(0, 101, 20), xlabel="SNR (dB)", ylabel="Classification accuracy (%)")
    format_ieee_axis(ax)
    set_panel_aspect_10_9(ax)
    inset.set(xlim=(-4, 5), ylim=(70, 100), xticks=[-4, -1, 2, 5], yticks=[70, 80, 90, 100])
    inset.grid(True, linestyle=":", linewidth=0.35, alpha=0.5)
    inset.tick_params(direction="out", labelsize=7, length=2, width=0.55, pad=1)
    for spine in inset.spines.values():
        spine.set_linewidth(0.65)
    ax.indicate_inset_zoom(inset, edgecolor="0.35", alpha=0.75, linewidth=0.55)
    boxed_legend(ax, loc="lower right", fontsize=8.5, borderaxespad=0.18)
    fig.tight_layout(pad=0.35)
    for extension in ("pdf", "png"):
        fig.savefig(out / f"{name}.{extension}", dpi=600)
        # Retain the explicit five-seed alias for existing automation.
        alias = f"paper_baseline_protocol_{protocol.lower()}_five_seed_accuracy_vs_snr.{extension}"
        (out / alias).write_bytes((out / f"{name}.{extension}").read_bytes())
    plt.close(fig)
    individual.to_csv(source / f"protocol_{protocol.lower()}_snr_seed_runs.csv", index=False)
    aggregate.to_csv(source / f"protocol_{protocol.lower()}_snr_summary.csv", index=False)
    (source / f"protocol_{protocol.lower()}_figure_manifest.json").write_text(json.dumps(dict(
        protocol=protocol, front_end="coherent" if protocol == "A" else "dcft_refined",
        paper_figure=5 if protocol == "A" else 6,
        figure_pdf=str(out / f"{name}.pdf"),
        seeds=[41, 73, 107, 149, 211], runs=manifest,
        caption=f"Protocol-{protocol} test accuracy versus SNR across five matched seeds. Lines show the mean across seeds."
    ), indent=2) + "\n", encoding="utf-8")
    print(f"Saved five-seed figures: {out / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["A", "B"], default="B")
    parser.add_argument("--five-seed", action="store_true",
                        help="Accepted for compatibility; formal plots always use all five seeds.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Optional destination for exported PDF and PNG figures.")
    args = parser.parse_args()
    plot_five_seed(args.protocol, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
