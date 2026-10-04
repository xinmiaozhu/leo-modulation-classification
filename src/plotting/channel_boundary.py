"""Single-panel channel-boundary figure in the paper-baseline visual style."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

from src.plotting.common import (
    IEEE_TRANS_PALETTE, boxed_legend, format_ieee_axis,
    set_ieee_trans_style, set_panel_aspect_10_9,
)

# Labels read as training channel / test channel. All use the same
# unequalized, carrier-compensated neural receiver; no descriptor-only scores.
CURVES = (
    ("flat", "flat", "Flat / Flat", "black", "o", True),
    ("flat", "reference", "Flat / Static", "gray", "s", False),
    ("reference", "reference", "Static / Static", "red", "^", True),
    ("flat", "long", "Flat / Long", "orange", "D", False),
    ("reference", "long", "Static / Long", "blue", "v", True),
    ("flat", "tv_fast", "Flat / Fast", "purple", "X", True),
    ("tv_slow", "tv_fast", "Slow / Fast", "green", "P", False),
)


def aggregate_curves(neural: pd.DataFrame) -> pd.DataFrame:
    required = {"train_condition", "test_condition", "receiver", "snr_db", "seed", "n", "accuracy"}
    missing = required.difference(neural.columns)
    if missing:
        raise ValueError(f"Missing neural result columns: {sorted(missing)}")
    data = neural.loc[neural.snr_db.astype(str) != "all"].copy()
    for key in ("snr_db", "seed", "n", "accuracy"):
        data[key] = pd.to_numeric(data[key], errors="raise")
    if not np.isfinite(data[["snr_db", "seed", "n", "accuracy"]].to_numpy()).all():
        raise ValueError("Non-finite result values cannot be plotted.")
    if not data.accuracy.between(0, 1).all() or (data.n <= 0).any():
        raise ValueError("Accuracy must be in [0,1] and test counts must be positive.")
    seeds, snrs = sorted(data.seed.unique()), sorted(data.snr_db.unique())
    if not seeds or not snrs:
        raise ValueError("No per-SNR neural results found.")
    rows = []
    counts = {}
    for train, test, label, _, _, _ in CURVES:
        selected = data.loc[(data.train_condition == train) & (data.test_condition == test)
                            & (data.receiver == "unequalized")]
        if selected.duplicated(["seed", "snr_db"]).any():
            raise ValueError(f"Duplicate seed/SNR rows for {label}")
        if sorted(selected.snr_db.unique()) != snrs:
            raise ValueError(f"Missing SNR points for {label}")
        for snr in snrs:
            group = selected.loc[selected.snr_db == snr]
            if sorted(group.seed.unique()) != seeds or group.n.nunique() != 1:
                raise ValueError(f"Unbalanced or missing seed results for {label} at SNR={snr}")
            count = int(group.n.iloc[0])
            if snr in counts and counts[snr] != count:
                raise ValueError("Curves must use equal test counts at each SNR.")
            counts[snr] = count
            rows.append({"train_condition": train, "test_condition": test,
                         "receiver": "unequalized", "display_label": label,
                         "snr_db": float(snr), "model_seed_count": len(seeds),
                         "model_seeds": ",".join(str(int(seed)) for seed in seeds),
                         "test_frames_per_seed": count,
                         "accuracy_percent": 100 * group.accuracy.mean(),
                         "seed_std_pp": 100 * group.accuracy.std(ddof=1) if len(seeds) > 1 else 0.})
    return pd.DataFrame(rows)


def plot_channel_boundary(neural: pd.DataFrame, output: str | Path, *, dpi: int = 600) -> Path:
    """Use the exact main-axis typography/geometry of script 46's reference.

    A two-column inset legend fits below the lowest curve without hiding the
    channel-mismatch plateaus. No inset zoom is needed for separated curves.
    """
    table = aggregate_curves(neural)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context():
        set_ieee_trans_style()
        fig, ax = plt.subplots(figsize=(3.0, 2.7))
        for train, test, label, color_name, marker, filled in CURVES:
            group = table.loc[(table.train_condition == train) & (table.test_condition == test)].sort_values("snr_db")
            color = IEEE_TRANS_PALETTE[color_name]
            ax.plot(group.snr_db, group.accuracy_percent, label=label,
                    color=color, linestyle="-", linewidth=1.45 if color_name == "green" else 1.15,
                    marker=marker, markersize=3.0,
                    markerfacecolor=color if filled else "white",
                    markeredgecolor=color, markeredgewidth=0.75)
        snr = np.sort(table.snr_db.unique())
        ax.set_xticks(snr)
        ax.set_xlim(snr[0] - .5, snr[-1] + .5)
        ax.set_ylim(0., 101.5)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Classification accuracy (%)")
        format_ieee_axis(ax)
        set_panel_aspect_10_9(ax)
        boxed_legend(ax, loc="lower right", ncol=2, fontsize=8.5,
                     borderaxespad=.12, columnspacing=.65,
                     title="Training / Test", title_fontsize=8.5)
        # The supplied reference additionally crops its 3 x 2.7-inch canvas:
        # CropBox [0, 2.18616, 210.088, 186.769] points. Match the visible page
        # dimensions, laying out labels inside the page instead of clipping
        # a renderer-dependent tight bounding box. Font sizes stay in points.
        page_width, page_height = 210.088 / 72, (186.769 - 2.18616) / 72
        fig.set_size_inches(page_width, page_height)
        fig.tight_layout(pad=.35)
        crop = Bbox.from_bounds(0, 0, page_width, page_height)
        fig.savefig(output, dpi=dpi, bbox_inches=crop, pad_inches=0)
        fig.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches=crop, pad_inches=0)
        plt.close(fig)
    table.to_csv(output.with_suffix(".csv"), index=False)
    metadata = {
        "metric": "Neural classification accuracy; arithmetic mean across model seeds",
        "receiver": "carrier-compensated I/Q + exact-mixture descriptor, without channel equalization",
        "legend_convention": "training channel / test channel",
        "channel_names": {"Flat": "flat", "Static": "reference static multipath",
                          "Long": "unseen long-delay multipath", "Slow": "slow time-varying multipath",
                          "Fast": "fast time-varying multipath"},
        "model_seeds": table.model_seeds.iloc[0],
        "canvas_inches": [210.088 / 72, 184.58284 / 72], "axes_width_to_height": "10:9",
        "export_page_points": [210.088, 184.58284],
        "font": "Times New Roman", "axis_label_pt": 10.,
        "x_tick_pt": 8.1, "y_tick_pt": 9., "legend_pt": 8.5,
        "style_reference": "paper_baseline_protocol_a_seed41_accuracy_vs_snr.pdf",
        "display": "Seven representative curves; seed standard deviations retained in CSV, not drawn as bands",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output
