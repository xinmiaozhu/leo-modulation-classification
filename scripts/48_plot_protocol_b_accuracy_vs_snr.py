#!/usr/bin/env python
"""Plot all seven Protocol-B hybrid-DFRFT test results at a fixed seed."""

from __future__ import annotations

import argparse
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
    ("nasa_hoc_nn", "C5 NASA HOC-NN", "blue", "v"),
    ("starnet", "C6 STARNet", "purple", "X"),
    ("proposed", "Proposed I/Q-Constellation", "green", "P"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    source = ROOT / "outputs/results/hybrid_dfrft/protocol_b"
    name = f"paper_baseline_protocol_b_seed{args.seed}_accuracy_vs_snr"
    out = ROOT / "outputs/figures/paper"
    out.mkdir(parents=True, exist_ok=True)
    set_ieee_trans_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.15))
    inset = ax.inset_axes([0.50, 0.56, 0.46, 0.35])
    curves, manifest = [], []
    reference = None
    for method, label, color_key, marker in METHODS:
        path = source / method / f"seed_{args.seed}_test.csv"
        metadata = json.loads(path.with_suffix(".summary.json").read_text())
        if metadata["split"] != "test":
            raise ValueError(f"Not a test result: {path}")
        df = pd.read_csv(path)
        if df["index"].duplicated().any() or not df["correct"].isin([0, 1]).all():
            raise ValueError(f"Invalid predictions: {path}")
        samples = df.sort_values("index")[["index", "label", "snr_db"]].reset_index(drop=True)
        if reference is not None:
            pd.testing.assert_frame_equal(reference, samples)
        reference = samples
        if not np.isclose(df["correct"].mean(), metadata["accuracy"]):
            raise ValueError(f"Accuracy disagrees with metadata: {path}")
        curve = df.groupby("snr_db")["correct"].agg(count="size", accuracy="mean").reset_index()
        curve["accuracy_percent"] = 100 * curve["accuracy"]
        curve["method"] = method
        curve["display_label"] = label
        curve["seed"] = args.seed
        curves.append(curve)
        manifest.append({"method": method, "eval_csv": str(path.relative_to(ROOT)),
                         "checkpoint": metadata["checkpoint"], "accuracy": metadata["accuracy"]})
        color = IEEE_TRANS_PALETTE[color_key]
        for panel in (ax, inset):
            panel.plot(curve.snr_db, curve.accuracy_percent, label=label,
                       color=color, marker=marker, markersize=3 if panel is ax else 2.2,
                       markerfacecolor=color if method == "starnet" else "white",
                       markeredgewidth=0.75, linewidth=1.45 if method == "proposed" else 1.15)
    snr = np.sort(reference.snr_db.unique())
    ax.set(xlim=(snr[0] - 0.5, snr[-1] + 0.5), ylim=(0, 101.5),
           xticks=snr, yticks=np.arange(0, 101, 20),
           xlabel="SNR (dB)", ylabel="Classification accuracy (%)")
    format_ieee_axis(ax)
    set_panel_aspect_10_9(ax)
    inset.set(xlim=(-4.4, 8.4), ylim=(70, 100), xticks=[-4, -1, 2, 5, 8], yticks=[70, 80, 90, 100])
    inset.grid(True, linestyle=":", linewidth=0.35, alpha=0.5)
    inset.tick_params(direction="out", labelsize=7, length=2, width=0.55, pad=1)
    for spine in inset.spines.values():
        spine.set_linewidth(0.65)
    ax.indicate_inset_zoom(inset, edgecolor="0.35", alpha=0.75, linewidth=0.55)
    boxed_legend(ax, loc="lower right", fontsize=8.5, borderaxespad=0.18)
    fig.tight_layout(pad=0.35)
    for extension in ("pdf", "png", "svg"):
        path = out / f"{name}.{extension}"
        fig.savefig(path, dpi=600)
        print(path)
    plt.close(fig)
    pd.concat(curves, ignore_index=True).to_csv(source / f"{name}.csv", index=False)
    (source / f"{name}.json").write_text(json.dumps({
        "protocol": "B", "front_end": "hybrid_dfrft", "seed": args.seed,
        "runs": manifest,
        "missing_reference_methods": [],
        "caption": "Protocol-B test accuracy versus SNR at the selected checkpoints (seed 41).".replace("41", str(args.seed)),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
