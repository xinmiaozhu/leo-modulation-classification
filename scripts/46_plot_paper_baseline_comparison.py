#!/usr/bin/env python
"""Plot the external paper-baseline comparison in an IEEE-style format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plotting.common import IEEE_TRANS_PALETTE, boxed_legend, format_ieee_axis, set_panel_aspect_10_9, set_ieee_trans_style


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot paper-baseline AMC comparison.")
    p.add_argument("--summary", type=str, default="outputs/results/paper_baselines/summary.csv")
    p.add_argument("--output-dir", type=str, default="outputs/figures/paper")
    p.add_argument(
        "--accuracy-output",
        type=str,
        default="paper_baseline_accuracy_vs_snr.pdf",
    )
    p.add_argument(
        "--overall-output",
        type=str,
        default="paper_baseline_overall_accuracy.pdf",
    )
    p.add_argument(
        "--table-output",
        type=str,
        default="outputs/results/paper_baselines/snr_curve.csv",
    )
    p.add_argument("--fig-width", type=float, default=3.0)
    p.add_argument("--fig-height", type=float, default=2.7)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument(
        "--exclude-methods",
        nargs="*",
        default=[],
        help="Method IDs to omit from the rendered figure while retaining source CSV records.",
    )
    return p.parse_args()


def _set_style() -> None:
    set_ieee_trans_style()


def _load(summary_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(summary_path)
    required = {"method_id", "display_label", "eval_csv"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Summary is missing columns: {missing}")

    # Keep historical result manifests compatible with the current model name.
    summary["display_label"] = summary["display_label"].str.replace(
        r"\bSat-(?:CNN|NN)\b", "IQCNet", case=False, regex=True
    )

    rows: list[dict[str, object]] = []
    for _, method in summary.iterrows():
        path = Path(str(method["eval_csv"]))
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluation CSV: {path}")
        df = pd.read_csv(path, usecols=["snr_db", "correct"])
        grouped = df.groupby("snr_db", sort=True)["correct"].agg(["size", "mean"]).reset_index()
        for _, point in grouped.iterrows():
            rows.append(
                {
                    "method_id": str(method["method_id"]),
                    "display_label": str(method["display_label"]),
                    "snr_db": float(point["snr_db"]),
                    "count": int(point["size"]),
                    "accuracy": float(point["mean"]),
                    "accuracy_percent": 100.0 * float(point["mean"]),
                }
            )
    return summary, pd.DataFrame(rows)


def _styles() -> list[dict[str, object]]:
    return [
        {"color": IEEE_TRANS_PALETTE["black"], "marker": "o", "linestyle": "-", "mfc": "black"},
        {"color": IEEE_TRANS_PALETTE["gray"], "marker": "s", "linestyle": "-", "mfc": "white"},
        {"color": IEEE_TRANS_PALETTE["red"], "marker": "^", "linestyle": "-", "mfc": IEEE_TRANS_PALETTE["red"]},
        {"color": IEEE_TRANS_PALETTE["orange"], "marker": "D", "linestyle": "-", "mfc": "white"},
        {"color": IEEE_TRANS_PALETTE["blue"], "marker": "v", "linestyle": "-", "mfc": IEEE_TRANS_PALETTE["blue"]},
        {"color": IEEE_TRANS_PALETTE["purple"], "marker": "X", "linestyle": "-", "mfc": IEEE_TRANS_PALETTE["purple"]},
        {"color": IEEE_TRANS_PALETTE["green"], "marker": "P", "linestyle": "-", "mfc": "white"},
    ]


def _finish_axis(ax: plt.Axes) -> None:
    format_ieee_axis(ax)


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_snr(summary: pd.DataFrame, curve: pd.DataFrame, path: Path, args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    styles = _styles()
    plotted: list[tuple[pd.DataFrame, dict[str, object], str, str, float]] = []
    for idx, method in summary.reset_index(drop=True).iterrows():
        method_id = str(method["method_id"])
        g = curve[curve["method_id"] == method_id].sort_values("snr_db")
        if g.empty:
            continue
        style = styles[idx % len(styles)]
        linewidth = 1.45 if method_id == "Proposed" else 1.15
        plotted.append((g, style, method_id, str(method["display_label"]), linewidth))

    for g, style, _, label, linewidth in plotted:
        ax.plot(
            g["snr_db"],
            g["accuracy_percent"],
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=linewidth,
            marker=style["marker"],
            markersize=3.0,
            markerfacecolor=style["mfc"],
            markeredgecolor=style["color"],
            markeredgewidth=0.75,
        )
    snr = np.sort(curve["snr_db"].unique())
    if snr.size:
        ax.set_xticks(snr)
        ax.set_xlim(float(snr[0]) - 0.5, float(snr[-1]) + 0.5)
    ax.set_ylim(0.0, 101.5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Classification accuracy (%)")
    _finish_axis(ax)
    set_panel_aspect_10_9(ax)

    # Enlarge the region where the neural baselines and proposed receiver differ.
    inset = ax.inset_axes([0.50, 0.56, 0.46, 0.35])
    for g, style, method_id, _, linewidth in plotted:
        # STARNet remains in the main panel and legend, but its separated curve
        # is omitted from this local comparison of the clustered methods.
        if method_id == "C6":
            continue
        inset.plot(
            g["snr_db"],
            g["accuracy_percent"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=max(0.75, 0.85 * linewidth),
            marker=style["marker"],
            markersize=2.2,
            markerfacecolor=style["mfc"],
            markeredgecolor=style["color"],
            markeredgewidth=0.45,
        )
    inset.set_xlim(-4.4, 8.4)
    inset.set_ylim(70.0, 100.0)
    inset.set_xticks([-4, -1, 2, 5, 8])
    inset.set_yticks([70, 80, 90, 100])
    inset.grid(True, linestyle=":", linewidth=0.35, alpha=0.5)
    inset.tick_params(direction="out", labelsize=7.0, length=2.0, width=0.55, pad=1.0)
    for spine in inset.spines.values():
        spine.set_linewidth(0.65)
    ax.indicate_inset_zoom(inset, edgecolor="0.35", alpha=0.75, linewidth=0.55)

    boxed_legend(
        ax,
        loc="lower right",
        ncol=1,
        fontsize=8.5,
        borderaxespad=0.12,
    )
    _save(fig, path, args.dpi)


def _plot_overall(summary: pd.DataFrame, path: Path, args: argparse.Namespace) -> None:
    values: list[float] = []
    for _, method in summary.iterrows():
        df = pd.read_csv(str(method["eval_csv"]), usecols=["correct"])
        p = float(df["correct"].mean())
        values.append(100.0 * p)

    labels = summary["display_label"].astype(str).tolist()
    colors = [style["color"] for style in _styles()[: len(labels)]]
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.62, color=colors, edgecolor="black", linewidth=0.55)
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 101.5)
    ax.set_xlabel("Overall classification accuracy (%)")
    for yi, value in zip(y, values):
        ax.text(min(value + 1.0, 98.5), yi, f"{value:.1f}", va="center", ha="left", fontsize=6.8)
    _finish_axis(ax)
    set_panel_aspect_10_9(ax)
    _save(fig, path, args.dpi)


def main() -> None:
    args = parse_args()
    _set_style()
    summary, curve = _load(Path(args.summary))
    excluded = {str(method) for method in args.exclude_methods}
    if excluded:
        summary = summary[~summary["method_id"].astype(str).isin(excluded)].reset_index(drop=True)
        curve = curve[~curve["method_id"].astype(str).isin(excluded)].reset_index(drop=True)
        if summary.empty or curve.empty:
            raise ValueError("--exclude-methods removed every plotted method.")
    table_path = Path(args.table_output)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(table_path, index=False)
    print(f"Saved: {table_path}")

    out = Path(args.output_dir)
    _plot_snr(summary, curve, out / args.accuracy_output, args)
    _plot_overall(summary, out / args.overall_output, args)


if __name__ == "__main__":
    main()
