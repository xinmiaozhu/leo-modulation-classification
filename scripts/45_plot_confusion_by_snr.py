#!/usr/bin/env python
"""Plot modulation-classification confusion matrices for selected SNR ranges."""

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot confusion matrices for selected SNR ranges.")
    p.add_argument("--eval-csv", type=str, required=True, help="Per-sample evaluation CSV.")
    p.add_argument("--raw-data", type=str, default=None, help="Raw HDF5 file used to read label_names.")
    p.add_argument("--output", type=str, default="outputs/figures/paper/confusion_snr_0_plus5.pdf")
    p.add_argument("--csv-output", type=str, default=None)
    p.add_argument(
        "--snr-ranges",
        type=float,
        nargs="+",
        default=[-2.5, 2.5, 2.5, 7.5],
        help="Pairs of SNR range edges, e.g. -10 -5 0 5 for two ranges.",
    )
    p.add_argument(
        "--snr-points",
        type=float,
        nargs="+",
        default=None,
        help="Target SNR values. If set, draw one matrix per target SNR instead of using --snr-ranges.",
    )
    p.add_argument(
        "--snr-point-mode",
        choices=("exact", "nearest"),
        default="exact",
        help="How to match --snr-points to rows in --eval-csv.",
    )
    p.add_argument(
        "--snr-tolerance",
        type=float,
        default=1e-6,
        help="Tolerance used for exact SNR matching.",
    )
    p.add_argument(
        "--range-labels",
        type=str,
        nargs="*",
        default=None,
        help="Optional subplot labels, one per SNR range.",
    )
    p.add_argument("--fig-width", type=float, default=3.45, help="Figure width in inches; 3.45 is IEEE single-column.")
    p.add_argument("--fig-height", type=float, default=3.0)
    p.add_argument("--cell-fontsize", type=float, default=7.0, help="Font size for confusion-matrix cell numbers.")
    p.add_argument(
        "--axis-label-fontsize",
        type=float,
        default=10.0,
        help="Font size for the true/predicted-class and colorbar labels.",
    )
    p.add_argument(
        "--class-label-fontsize",
        type=float,
        default=7.0,
        help="Font size for modulation-class tick labels.",
    )
    p.add_argument("--show-title", action="store_true", help="Show SNR range title above each matrix.")
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def set_trans_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.45,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def read_label_names(raw_data: str | None, labels: np.ndarray) -> list[str]:
    if raw_data is not None:
        with h5py.File(raw_data, "r") as f:
            if "label_names" in f:
                return [x.decode("utf-8") if hasattr(x, "decode") else str(x) for x in f["label_names"][:]]
    num_classes = int(max(np.max(labels), 0) + 1)
    return [str(i) for i in range(num_classes)]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def normalize_rows(cm: np.ndarray) -> np.ndarray:
    denom = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, denom, out=np.full_like(cm, np.nan, dtype=np.float64), where=denom > 0)


def default_range_labels(edges: np.ndarray) -> list[str]:
    labels = []
    for lo, hi in zip(edges[0::2], edges[1::2]):
        center = 0.5 * (lo + hi)
        labels.append(rf"SNR = {center:g} dB" + "\n" + rf"$[{lo:g},{hi:g}]$ dB")
    return labels


def snr_suffix(lo: float, hi: float) -> str:
    center = 0.5 * (lo + hi)
    text = f"snr{center:g}db".replace("-", "minus").replace(".", "p")
    return text


def snr_point_suffix(target: float) -> str:
    return f"snr{target:g}db".replace("-", "minus").replace(".", "p")


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_output) if args.csv_output else out_path.with_suffix(".csv")

    df = pd.read_csv(args.eval_csv)
    required = {"label", "pred", "snr_db"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Evaluation CSV is missing columns: {sorted(missing)}")

    labels = np.asarray(df["label"], dtype=int)
    class_names = read_label_names(args.raw_data, labels)
    num_classes = len(class_names)

    records = []
    matrices: list[dict[str, object]] = []
    snr_values = df["snr_db"].to_numpy(dtype=np.float64)

    if args.snr_points is not None:
        targets = np.asarray(args.snr_points, dtype=np.float64)
        if args.range_labels and len(args.range_labels) != len(targets):
            raise SystemExit("--range-labels must have the same length as --snr-points.")
        unique_snr = np.sort(np.unique(snr_values))
        for point_idx, target in enumerate(targets):
            if args.snr_point_mode == "nearest":
                actual = float(unique_snr[np.argmin(np.abs(unique_snr - target))])
            else:
                matches = unique_snr[np.isclose(unique_snr, target, atol=args.snr_tolerance, rtol=0.0)]
                if len(matches) == 0:
                    available = ", ".join(f"{x:g}" for x in unique_snr)
                    raise SystemExit(f"SNR {target:g} dB not found. Available SNR values: {available}")
                actual = float(matches[0])

            mask = np.isclose(snr_values, actual, atol=args.snr_tolerance, rtol=0.0)
            if args.range_labels:
                title = args.range_labels[point_idx]
            elif np.isclose(actual, target, atol=args.snr_tolerance, rtol=0.0):
                title = rf"SNR = {target:g} dB"
            else:
                title = rf"Target SNR = {target:g} dB" + "\n" + rf"Actual SNR = {actual:g} dB"
            part = df.loc[mask]
            cm = confusion_matrix(part["label"].to_numpy(), part["pred"].to_numpy(), num_classes)
            cm_norm = normalize_rows(cm)
            matrices.append(
                {
                    "cm": cm,
                    "cm_norm": cm_norm,
                    "count": int(len(part)),
                    "title": title,
                    "suffix": snr_point_suffix(float(target)),
                    "snr_target_db": float(target),
                    "snr_actual_db": actual,
                    "snr_low_db": actual,
                    "snr_high_db": actual,
                }
            )
    else:
        edges = np.asarray(args.snr_ranges, dtype=np.float64)
        if len(edges) % 2 != 0 or len(edges) < 2:
            raise SystemExit("--snr-ranges must contain low/high pairs.")
        ranges = list(zip(edges[0::2], edges[1::2]))
        if args.range_labels:
            if len(args.range_labels) != len(ranges):
                raise SystemExit("--range-labels must have the same length as SNR ranges.")
            range_labels = args.range_labels
        else:
            range_labels = default_range_labels(edges)

        for (lo, hi), title in zip(ranges, range_labels):
            if lo == min(edges[0::2]):
                mask = (snr_values >= lo) & (snr_values <= hi)
            else:
                mask = (snr_values > lo) & (snr_values <= hi)
            part = df.loc[mask]
            cm = confusion_matrix(part["label"].to_numpy(), part["pred"].to_numpy(), num_classes)
            cm_norm = normalize_rows(cm)
            matrices.append(
                {
                    "cm": cm,
                    "cm_norm": cm_norm,
                    "count": int(len(part)),
                    "title": title,
                    "suffix": snr_suffix(float(lo), float(hi)),
                    "snr_target_db": float("nan"),
                    "snr_actual_db": float("nan"),
                    "snr_low_db": float(lo),
                    "snr_high_db": float(hi),
                }
            )

    for item in matrices:
        cm = item["cm"]
        cm_norm = item["cm_norm"]

        for i, true_name in enumerate(class_names):
            for j, pred_name in enumerate(class_names):
                records.append(
                    {
                        "snr_target_db": item["snr_target_db"],
                        "snr_actual_db": item["snr_actual_db"],
                        "snr_low_db": item["snr_low_db"],
                        "snr_high_db": item["snr_high_db"],
                        "range_label": str(item["title"]).replace("\n", " "),
                        "true_label": true_name,
                        "pred_label": pred_name,
                        "count": int(cm[i, j]),
                        "row_percent": float(100.0 * cm_norm[i, j]),
                    }
                )

    pd.DataFrame(records).to_csv(csv_path, index=False)

    set_trans_style()

    def draw_matrix(ax, cm: np.ndarray, cm_norm: np.ndarray, count: int, title: str):
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad(color="white")
        im = ax.imshow(100.0 * cm_norm, cmap=cmap, vmin=0.0, vmax=100.0)
        if args.show_title:
            ax.set_title(f"{title}, N={count}")
        ax.set_xlabel("Predicted class", fontsize=args.axis_label_fontsize)
        ax.set_ylabel("True class", fontsize=args.axis_label_fontsize)
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(
            class_names,
            rotation=35,
            ha="right",
            fontsize=args.class_label_fontsize,
        )
        ax.set_yticklabels(class_names, fontsize=args.class_label_fontsize)

        row_counts = cm.sum(axis=1)
        for i in range(num_classes):
            if row_counts[i] == 0:
                ax.text(
                    (num_classes - 1) / 2,
                    i,
                    "n=0",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=args.cell_fontsize,
                    fontstyle="italic",
                )
                continue
            for j in range(num_classes):
                pct = 100.0 * cm_norm[i, j]
                text_color = "white" if pct >= 55.0 else "black"
                ax.text(j, i, f"{pct:.0f}", ha="center", va="center", color=text_color, fontsize=args.cell_fontsize)

        ax.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
        ax.grid(which="minor", color="0.85", linestyle="-", linewidth=0.35)
        ax.tick_params(which="minor", bottom=False, left=False)
        return im

    saved_paths: list[Path] = []
    for item in matrices:
        fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
        im = draw_matrix(ax, item["cm"], item["cm_norm"], int(item["count"]), str(item["title"]))
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label("Row-normalized accuracy (%)", fontsize=args.axis_label_fontsize)
        pdf_path = out_path.with_name(f"{out_path.stem}_{item['suffix']}{out_path.suffix}")
        fig.savefig(pdf_path)
        plt.close(fig)
        saved_paths.append(pdf_path)

    for item in matrices:
        cm = item["cm"]
        cm_norm = item["cm_norm"]
        count = int(item["count"])
        overall_acc = float(np.trace(cm) / np.sum(cm)) if np.sum(cm) > 0 else float("nan")
        macro_recall = float(np.nanmean(np.diag(cm_norm)))
        target = float(item["snr_target_db"])
        actual = float(item["snr_actual_db"])
        if np.isfinite(target):
            print(
                f"SNR target {target:g} dB, actual {actual:g} dB, count={count}, "
                f"accuracy={overall_acc:.4f}, macro_recall={macro_recall:.4f}"
            )
        else:
            lo = float(item["snr_low_db"])
            hi = float(item["snr_high_db"])
            print(
                f"SNR range [{lo:g}, {hi:g}] dB, count={count}, "
                f"accuracy={overall_acc:.4f}, macro_recall={macro_recall:.4f}"
            )
    for path in saved_paths:
        print(f"Saved figure: {path}")
    print(f"Saved table: {csv_path}")


if __name__ == "__main__":
    main()
