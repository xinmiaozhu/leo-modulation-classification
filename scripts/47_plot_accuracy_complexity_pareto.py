#!/usr/bin/env python
"""Plot IEEE-style accuracy/latency and accuracy/parameter Pareto fronts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "C1": "#7f7f7f",
    "C2": "#e41a1c",
    "C3": "#ff7f00",
    "C4": "#377eb8",
    "C5": "#2ca02c",
    "Proposed": "#7b3294",
}
MARKERS = {
    "C1": "s",
    "C2": "^",
    "C3": "D",
    "C4": "v",
    "C5": "P",
    "Proposed": "X",
}
LABEL_OFFSETS = {
    "end_to_end_cpu_median_ms": {
        "C1": (5, 5),
        "C2": (3, 6),
        "C3": (5, 5),
        "C4": (-8, -12),
        "C5": (5, 7),
        "Proposed": (5, -14),
    },
    "parameter_count": {
        "C1": (5, 5),
        "C2": (-17, 6),
        "C3": (5, -13),
        "C4": (5, 4),
        "C5": (5, 6),
        "Proposed": (5, 7),
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot accuracy-complexity Pareto fronts.")
    p.add_argument("--input", required=True)
    p.add_argument("--latency-output", required=True)
    p.add_argument("--parameter-output", required=True)
    return p.parse_args()


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "savefig.dpi": 600,
        }
    )


def _pareto_rows(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    ordered = df.sort_values([x_col, "accuracy_percent"], ascending=[True, False])
    keep: list[int] = []
    best = -np.inf
    for index, row in ordered.iterrows():
        value = float(row["accuracy_percent"])
        if value > best + 1e-12:
            keep.append(index)
            best = value
    return df.loc[keep].sort_values(x_col)


def _plot(df: pd.DataFrame, x_col: str, x_label: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.55, 3.35))
    pareto = _pareto_rows(df, x_col)
    ax.plot(
        pareto[x_col],
        pareto["accuracy_percent"],
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        zorder=1,
        label="Pareto front",
    )
    for _, row in df.iterrows():
        method_id = str(row["method_id"])
        proposed = method_id == "Proposed"
        ax.scatter(
            row[x_col],
            row["accuracy_percent"],
            s=52 if proposed else 34,
            marker=MARKERS[method_id],
            color=COLORS[method_id],
            edgecolor="black",
            linewidth=0.55,
            zorder=3,
            label=str(row["display_label"]),
        )
        ax.annotate(
            method_id,
            (row[x_col], row["accuracy_percent"]),
            xytext=LABEL_OFFSETS[x_col][method_id],
            textcoords="offset points",
            fontsize=7.5,
        )
    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Independent-test accuracy (%)")
    ax.grid(True, which="major", color="#d0d0d0", linewidth=0.55)
    ax.grid(True, which="minor", color="#ececec", linewidth=0.35)
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_ylim(max(0.0, float(df["accuracy_percent"].min()) - 4.0), 100.5)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _set_style()
    df = pd.read_csv(args.input)
    required = {
        "method_id",
        "display_label",
        "accuracy_percent",
        "end_to_end_cpu_median_ms",
        "parameter_count",
    }
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing complexity columns: {sorted(missing)}")
    # Normalize model names from historical benchmark tables.
    df["display_label"] = df["display_label"].str.replace(
        r"\bSat-(?:CNN|NN)\b", "IQCNet", case=False, regex=True
    )
    _plot(
        df,
        "end_to_end_cpu_median_ms",
        "End-to-end latency (ms/frame, one CPU thread)",
        Path(args.latency_output),
    )
    _plot(
        df,
        "parameter_count",
        "Effective model parameter count",
        Path(args.parameter_output),
    )
    print(f"Saved: {args.latency_output}")
    print(f"Saved: {args.parameter_output}")


if __name__ == "__main__":
    main()
