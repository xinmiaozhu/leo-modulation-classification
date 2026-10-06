#!/usr/bin/env python
"""Plot matched five-seed I/Q--constellation representation ablations."""

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
from matplotlib.transforms import Bbox

from src.plotting.common import (
    format_ieee_axis,
    set_ieee_trans_style,
    set_panel_aspect_10_9,
)


METHODS = ["iq_only", "constellation_only", "fusion"]
LABELS = ["I/Q-only", "Constellation-only", "Fusion"]
METHOD_COLORS = ["#2F73C5", "#F28E2B", "#3A9746"]
METHOD_MARKERS = ["o", "^", "D"]
TRAJECTORY_COLOR = "#AFAFAF"
ERROR_COLOR = "#303030"
MEAN_X_OFFSET = 0.14

# Exact single-panel PDF page used by manuscript Fig. 2.
FIG2_PANEL_SIZE_IN = (208.95 / 72.0, 188.892 / 72.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--protocol-a",
        default="outputs/results/exact_mixture/representation_ablation/protocol_a/seed_runs.csv",
    )
    p.add_argument(
        "--protocol-b",
        default="outputs/results/exact_mixture/representation_ablation/protocol_b/seed_runs.csv",
    )
    p.add_argument(
        "--output",
        default="outputs/figures/paper/representation_ablation_five_seed.pdf",
    )
    return p.parse_args()


def validate(table: pd.DataFrame, protocol: str) -> pd.DataFrame:
    required_columns = {"method", "seed", "accuracy"}
    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        raise ValueError(f"Protocol {protocol} missing columns: {sorted(missing_columns)}")
    work = table[table["method"].isin(METHODS)].copy()
    counts = work.groupby("method")["seed"].nunique()
    if any(int(counts.get(method, 0)) != 5 for method in METHODS):
        raise ValueError(f"Protocol {protocol} is not a complete five-seed comparison: {counts.to_dict()}")
    seed_sets = [set(work.loc[work["method"] == method, "seed"].astype(int)) for method in METHODS]
    if not all(seed_set == seed_sets[0] for seed_set in seed_sets[1:]):
        raise ValueError(f"Protocol {protocol} methods do not use matched seeds")
    work["accuracy_percent"] = 100.0 * work["accuracy"].astype(float)
    return work


def plot_panel(ax, table: pd.DataFrame) -> None:
    x = np.arange(len(METHODS), dtype=float)
    pivot = (
        table.pivot(index="seed", columns="method", values="accuracy_percent")
        .loc[:, METHODS]
        .sort_index()
    )
    # Thin paired trajectories expose seed-to-seed consistency without turning
    # the ablation into another accuracy-vs-SNR curve.
    for _, row in pivot.iterrows():
        ax.plot(
            x,
            row.to_numpy(float),
            color=TRAJECTORY_COLOR,
            linestyle="--",
            linewidth=0.55,
            alpha=0.62,
            zorder=1,
        )

    # Method color identifies the representation, while low opacity keeps the
    # five individual seeds subordinate to the summary statistics.
    for index, (method, color, marker) in enumerate(
        zip(METHODS, METHOD_COLORS, METHOD_MARKERS)
    ):
        ax.scatter(
            np.full(len(pivot), x[index]),
            pivot[method].to_numpy(float),
            s=7,
            marker=marker,
            facecolor=color,
            edgecolor=color,
            linewidth=0.35,
            alpha=0.88,
            zorder=3,
        )

    means = pivot.mean(axis=0).to_numpy(float)
    sds = pivot.std(axis=0, ddof=1).to_numpy(float)
    for index, color in enumerate(METHOD_COLORS):
        ax.errorbar(
            x[index] + MEAN_X_OFFSET,
            means[index],
            yerr=sds[index],
            fmt="D",
            color=ERROR_COLOR,
            ecolor=ERROR_COLOR,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=4.4,
            elinewidth=1.0,
            capsize=2.2,
            capthick=0.9,
            zorder=5,
        )

    ax.set_xticks(x, LABELS)
    ax.set_xlim(-0.28, len(METHODS) - 0.64)
    format_ieee_axis(ax, grid=False)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.yaxis.grid(True, color="0.68", linestyle=":", linewidth=0.55, alpha=0.75)
    set_panel_aspect_10_9(ax)


def save_fig2_sized_panels(fig, axes, output: Path) -> list[Path]:
    """Save A/B as true single-column panels matching Fig. 2 exactly."""

    axes = list(axes)
    original_size = fig.get_size_inches().copy()
    original_positions = {ax: ax.get_position().frozen() for ax in axes}
    original_visibility = {ax: ax.get_visible() for ax in axes}
    page_bbox = Bbox.from_bounds(0.0, 0.0, *FIG2_PANEL_SIZE_IN)
    saved: list[Path] = []
    try:
        fig.set_size_inches(*FIG2_PANEL_SIZE_IN, forward=True)
        for suffix, ax in zip(("a", "b"), axes):
            for other in axes:
                other.set_visible(other is ax)
            ax.set_position([0.19, 0.18, 0.79, 0.79])
            ax.set_ylabel("Test accuracy (%)")
            ax.yaxis.set_label_coords(-0.145, 0.5)
            ax.tick_params(labelleft=True)
            fig.canvas.draw()
            panel = output.with_name(f"{output.stem}_{suffix}{output.suffix}")
            fig.savefig(panel, bbox_inches=page_bbox, pad_inches=0)
            fig.savefig(
                panel.with_suffix(".png"), dpi=220,
                bbox_inches=page_bbox, pad_inches=0,
            )
            saved.extend([panel, panel.with_suffix(".png")])
    finally:
        fig.set_size_inches(*original_size, forward=True)
        for ax in axes:
            ax.set_position(original_positions[ax])
            ax.set_visible(original_visibility[ax])
        fig.canvas.draw()
    return saved


def main() -> None:
    args = parse_args()
    tables = [
        validate(pd.read_csv(args.protocol_a), "A"),
        validate(pd.read_csv(args.protocol_b), "B"),
    ]
    set_ieee_trans_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.18), sharey=False)
    plot_panel(axes[0], tables[0])
    plot_panel(axes[1], tables[1])
    axes[0].set_ylabel("Test accuracy (%)")

    axes[0].set_ylim(87.0, 91.0)
    axes[1].set_ylim(87.0, 91.0)
    axes[0].set_yticks(np.arange(87.0, 92.0, 1.0))
    axes[1].set_yticks(np.arange(87.0, 92.0, 1.0))

    fig.subplots_adjust(left=0.086, right=0.993, bottom=0.19, top=0.92, wspace=0.14)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for saved in save_fig2_sized_panels(fig, axes, output):
        print(saved)
    plt.close(fig)


if __name__ == "__main__":
    main()
