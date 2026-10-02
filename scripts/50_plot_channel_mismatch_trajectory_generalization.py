#!/usr/bin/env python
"""Plot exact-mixture recovery under frequency-selective channel mismatch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.common import (
    IEEE_TRANS_PALETTE,
    boxed_legend,
    format_ieee_axis,
    set_ieee_trans_style,
    set_panel_aspect_10_9,
)


CONDITIONS = ("unequalized", "pilot_ls", "oracle")
CONDITION_LABELS = ("No equalization", "Pilot-LS", "Oracle")
CONDITION_COLORS = (
    IEEE_TRANS_PALETTE["black"],
    IEEE_TRANS_PALETTE["blue"],
    IEEE_TRANS_PALETTE["red"],
)
CONDITION_MARKERS = ("o", "s", "^")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mismatch-csv",
        default="outputs/results/frequency_selective_equalization_100/exact_mixture_accuracy_by_snr.csv",
    )
    parser.add_argument(
        "--mismatch-output",
        default="outputs/figures/paper/channel_mismatch_equalization_vs_snr_fig6_style.pdf",
    )
    return parser.parse_args()


def validate_mismatch(table: pd.DataFrame) -> pd.DataFrame:
    required = {"condition", "snr_db", "n", "accuracy"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Mismatch table is missing columns: {sorted(missing)}")
    work = table[table["snr_db"].astype(str) != "all"].copy()
    work["snr_db"] = pd.to_numeric(work["snr_db"])
    work["n"] = pd.to_numeric(work["n"]).astype(int)
    work["accuracy"] = pd.to_numeric(work["accuracy"])
    if set(work["condition"]) != set(CONDITIONS):
        raise ValueError(f"Unexpected mismatch conditions: {sorted(work['condition'].unique())}")
    return work


def plot_mismatch(ax, table: pd.DataFrame) -> None:
    for condition, label, color, marker in zip(
        CONDITIONS, CONDITION_LABELS, CONDITION_COLORS, CONDITION_MARKERS
    ):
        group = table.loc[table["condition"] == condition].sort_values("snr_db")
        accuracy = group["accuracy"].to_numpy(float)
        ax.plot(
            group["snr_db"],
            100.0 * accuracy,
            label=label,
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.8,
            markersize=3.0,
            linewidth=1.25,
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Exact-mixture accuracy (%)")
    ax.set_xlim(-11.2, 21.2)
    ax.set_xticks(np.arange(-10, 21, 5))
    ax.set_ylim(20.0, 103.0)
    ax.set_yticks(np.arange(20, 101, 20))
    format_ieee_axis(ax)
    set_panel_aspect_10_9(ax)
    boxed_legend(
        ax,
        loc="lower right",
        fontsize=10,
        handletextpad=0.55,
        numpoints=1,
    )
def main() -> None:
    args = parse_args()
    mismatch = validate_mismatch(pd.read_csv(ROOT / args.mismatch_csv))
    set_ieee_trans_style()

    # Match the manuscript Fig. 6 artifact: 3.45 x 3.105 in single-column
    # canvas, 10:9 plotting region, 10-pt labels, and compact 8.5-pt legend.
    mismatch_output = ROOT / args.mismatch_output
    mismatch_output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.45, 3.105))
    plot_mismatch(ax, mismatch)
    fig.tight_layout(pad=0.35)
    fig.savefig(mismatch_output, dpi=600, bbox_inches="tight")
    fig.savefig(mismatch_output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(mismatch_output)
    print(mismatch_output.with_suffix(".png"))


if __name__ == "__main__":
    main()
