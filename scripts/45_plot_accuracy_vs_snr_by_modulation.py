#!/usr/bin/env python
"""Plot per-modulation classification probability versus SNR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.common import IEEE_TRANS_PALETTE, boxed_legend, format_ieee_axis, set_panel_aspect_10_9, set_ieee_trans_style


def _wilson_interval(correct: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count <= 0:
        return np.nan, np.nan
    p = float(correct) / float(count)
    z2 = z * z
    denominator = 1.0 + z2 / count
    center = (p + z2 / (2.0 * count)) / denominator
    radius = (
        z
        * np.sqrt(p * (1.0 - p) / count + z2 / (4.0 * count * count))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot correct classification probability versus SNR.")
    p.add_argument("--eval-csv", type=str, required=True, help="Per-sample evaluation CSV from 14_evaluate_model.py.")
    p.add_argument("--raw-data", type=str, default=None, help="Optional HDF5 used to read label_names.")
    p.add_argument("--output", type=str, required=True, help="Output PDF path. A PNG is also written.")
    p.add_argument("--csv-output", type=str, default=None, help="Optional CSV table for plotted points.")
    p.add_argument("--snr-min", type=float, default=None)
    p.add_argument("--snr-max", type=float, default=None)
    p.add_argument("--bin-width", type=float, default=3.0)
    p.add_argument(
        "--exact-snr",
        action="store_true",
        help="Group by exact SNR values instead of fixed-width bins. Use for SNR-balanced validation sets.",
    )
    p.add_argument("--min-count", type=int, default=3, help="Minimum samples per class/bin to plot a point.")
    p.add_argument("--fig-width", type=float, default=3.0)
    p.add_argument("--fig-height", type=float, default=2.7)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--title", type=str, default=None)
    return p.parse_args()


def _read_label_names(raw_data: str | None, labels: np.ndarray) -> list[str]:
    if raw_data:
        with h5py.File(raw_data, "r") as f:
            if "label_names" in f:
                return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["label_names"][:]]
    max_label = int(np.max(labels)) if labels.size else -1
    return [str(i) for i in range(max_label + 1)]


def _make_edges(snr: np.ndarray, snr_min: float | None, snr_max: float | None, bin_width: float) -> np.ndarray:
    lo = float(np.nanmin(snr)) if snr_min is None else float(snr_min)
    hi = float(np.nanmax(snr)) if snr_max is None else float(snr_max)
    width = float(bin_width)
    if width <= 0:
        raise ValueError("--bin-width must be positive.")
    lo = np.floor(lo / width) * width
    hi = np.ceil(hi / width) * width
    edges = np.arange(lo, hi + 0.5 * width, width, dtype=np.float64)
    if edges.size < 2:
        edges = np.asarray([lo, lo + width], dtype=np.float64)
    return edges


def _build_table(df: pd.DataFrame, label_names: list[str], edges: np.ndarray, min_count: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    labels = sorted(int(x) for x in df["label"].dropna().unique())
    for label in labels:
        g_label = df[df["label"].astype(int) == label]
        name = label_names[label] if label < len(label_names) else str(label)
        for left, right in zip(edges[:-1], edges[1:]):
            if right == edges[-1]:
                mask = (g_label["snr_db"] >= left) & (g_label["snr_db"] <= right)
            else:
                mask = (g_label["snr_db"] >= left) & (g_label["snr_db"] < right)
            g = g_label[mask]
            count = int(len(g))
            acc = float(g["correct"].mean()) if count >= int(min_count) else np.nan
            correct = int(np.rint(g["correct"].sum())) if count else 0
            ci_low, ci_high = (
                _wilson_interval(correct, count)
                if count >= int(min_count)
                else (np.nan, np.nan)
            )
            rows.append(
                {
                    "label": int(label),
                    "modulation": name,
                    "snr_left": float(left),
                    "snr_right": float(right),
                    "snr_center": float((left + right) / 2.0),
                    "count": count,
                    "accuracy": acc,
                    "wilson95_low": ci_low,
                    "wilson95_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def _build_exact_snr_table(
    df: pd.DataFrame,
    label_names: list[str],
    snr_min: float | None,
    snr_max: float | None,
    min_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    work = df.copy()
    if snr_min is not None:
        work = work[work["snr_db"] >= float(snr_min)]
    if snr_max is not None:
        work = work[work["snr_db"] <= float(snr_max)]

    labels = sorted(int(x) for x in work["label"].dropna().unique())
    snr_values = sorted(float(x) for x in work["snr_db"].dropna().unique())
    for label in labels:
        g_label = work[work["label"].astype(int) == label]
        name = label_names[label] if label < len(label_names) else str(label)
        for snr in snr_values:
            g = g_label[np.isclose(g_label["snr_db"].to_numpy(dtype=float), snr, atol=1e-9, rtol=0.0)]
            count = int(len(g))
            acc = float(g["correct"].mean()) if count >= int(min_count) else np.nan
            correct = int(np.rint(g["correct"].sum())) if count else 0
            ci_low, ci_high = (
                _wilson_interval(correct, count)
                if count >= int(min_count)
                else (np.nan, np.nan)
            )
            rows.append(
                {
                    "label": int(label),
                    "modulation": name,
                    "snr_left": snr,
                    "snr_right": snr,
                    "snr_center": snr,
                    "count": count,
                    "accuracy": acc,
                    "wilson95_low": ci_low,
                    "wilson95_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def _set_ieee_style() -> None:
    set_ieee_trans_style()


def _plot(table: pd.DataFrame, output: Path, fig_width: float, fig_height: float, dpi: int, title: str | None) -> None:
    _set_ieee_style()
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Keep legend samples visually clean.  Mixed dashed/dash-dot patterns with
    # large markers look cluttered in a compact IEEE single-column legend, so
    # use solid lines and let color + marker shape carry the class identity.
    styles = [
        {"color": IEEE_TRANS_PALETTE["black"], "marker": "o", "markerfacecolor": "black"},
        {"color": IEEE_TRANS_PALETTE["gray"], "marker": "s", "markerfacecolor": "white"},
        {"color": IEEE_TRANS_PALETTE["red"], "marker": "^", "markerfacecolor": IEEE_TRANS_PALETTE["red"]},
        {"color": IEEE_TRANS_PALETTE["orange"], "marker": "D", "markerfacecolor": "white"},
        {"color": IEEE_TRANS_PALETTE["blue"], "marker": "v", "markerfacecolor": IEEE_TRANS_PALETTE["blue"]},
        {"color": IEEE_TRANS_PALETTE["green"], "marker": "P", "markerfacecolor": "white"},
        {"color": IEEE_TRANS_PALETTE["purple"], "marker": "X", "markerfacecolor": IEEE_TRANS_PALETTE["purple"]},
        {"color": IEEE_TRANS_PALETTE["cyan"], "marker": "h", "markerfacecolor": "white"},
    ]

    for idx, (modulation, g) in enumerate(table.groupby("modulation", sort=False)):
        style = styles[idx % len(styles)]
        g = g.sort_values("snr_center")
        accuracy = g["accuracy"].to_numpy(dtype=float)
        ax.plot(
            g["snr_center"],
            accuracy,
            label=str(modulation),
            color=style["color"],
            linestyle="-",
            linewidth=1.25,
            marker=style["marker"],
            markersize=3.0,
            markerfacecolor=style["markerfacecolor"],
            markeredgewidth=0.8,
            markeredgecolor=style["color"],
        )

    ax.set_xlabel("SNR (dB)", fontsize=10.0)
    ax.set_ylabel("Correct classification probability", fontsize=10.0)
    ax.set_ylim(-0.02, 1.02)
    x_vals = table["snr_center"].to_numpy(dtype=float)
    finite_x = x_vals[np.isfinite(x_vals)]
    if finite_x.size:
        ax.set_xlim(float(np.min(finite_x)) - 0.5, float(np.max(finite_x)) + 0.5)
    format_ieee_axis(ax)
    set_panel_aspect_10_9(ax)
    if title:
        ax.set_title(title, fontsize=8.5, pad=2)
    boxed_legend(
        ax,
        loc="lower right",
        fontsize=8.5,
        handletextpad=0.55,
        numpoints=1,
    )
    fig.tight_layout(pad=0.35)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output}")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.eval_csv)
    required = {"label", "correct", "snr_db"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns in {args.eval_csv}: {missing}")

    df = df.copy()
    df["label"] = df["label"].astype(int)
    df["correct"] = df["correct"].astype(float)
    df["snr_db"] = df["snr_db"].astype(float)
    df = df[np.isfinite(df["snr_db"])]
    label_names = _read_label_names(args.raw_data, df["label"].to_numpy())
    if args.exact_snr:
        table = _build_exact_snr_table(df, label_names, args.snr_min, args.snr_max, args.min_count)
    else:
        edges = _make_edges(df["snr_db"].to_numpy(), args.snr_min, args.snr_max, args.bin_width)
        table = _build_table(df, label_names, edges, args.min_count)

    out = Path(args.output)
    csv_out = Path(args.csv_output) if args.csv_output else out.with_suffix(".csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_out, index=False)
    print(f"Saved table: {csv_out}")
    _plot(table, out, args.fig_width, args.fig_height, args.dpi, args.title)


if __name__ == "__main__":
    main()
