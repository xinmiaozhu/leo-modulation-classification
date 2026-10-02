#!/usr/bin/env python
"""Plot calibration loss and independent-test operating-rule closure."""

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


# Match the physical PDF page used by manuscript Fig. 2 after tight export.
FIG2_PANEL_SIZE_IN = (208.95 / 72.0, 188.892 / 72.0)


def save_fig2_sized_panels(fig, axes, output: Path) -> list[Path]:
    """Export true single-column panels with Fig. 2's size and typography."""

    axes = list(axes)
    original_size = fig.get_size_inches().copy()
    original_positions = {ax: ax.get_position().frozen() for ax in axes}
    original_visibility = {ax: ax.get_visible() for ax in axes}
    saved: list[Path] = []
    page_bbox = Bbox.from_bounds(0.0, 0.0, *FIG2_PANEL_SIZE_IN)
    try:
        fig.set_size_inches(*FIG2_PANEL_SIZE_IN, forward=True)
        for suffix, ax in zip(("a", "b"), axes):
            original_ylim = ax.get_ylim()
            for other in axes:
                other.set_visible(other is ax)
            # Equal normalized width/height gives essentially the same 10:9
            # physical plotting box as Fig. 2 on its 10:9 PDF page.
            ax.set_position([0.17, 0.17, 0.81, 0.81])
            # The label location inherited from the former two-panel canvas
            # can otherwise remain too far left after reflowing the axes.
            ax.yaxis.set_label_coords(-0.12, 0.5)
            if suffix == "b":
                # Reserve a small log-scale band below the first reported
                # tail point so the requested lower-right legend does not
                # obscure the Bonferroni curve in the narrower panel.
                ax.set_ylim(0.04, original_ylim[1])
            fig.canvas.draw()
            panel_path = output.with_name(f"{output.stem}_{suffix}{output.suffix}")
            fig.savefig(panel_path, bbox_inches=page_bbox, pad_inches=0)
            fig.savefig(
                panel_path.with_suffix(".png"), dpi=220,
                bbox_inches=page_bbox, pad_inches=0,
            )
            saved.extend([panel_path, panel_path.with_suffix(".png")])
            ax.set_ylim(original_ylim)
    finally:
        fig.set_size_inches(*original_size, forward=True)
        for ax in axes:
            ax.set_position(original_positions[ax])
            ax.set_visible(original_visibility[ax])
        fig.canvas.draw()
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classification",
        default="outputs/results/exact_mixture/gamma_amc_calibration_scan.csv",
    )
    parser.add_argument(
        "--operating-rule",
        default="outputs/results/exact_mixture/gamma_amc_operating_rule.csv",
    )
    parser.add_argument(
        "--joint-operating-rule",
        default="outputs/results/exact_mixture/joint_eta_operating_rule.csv",
        help="Joint-eta calibration/closure CSV for the right panel.",
    )
    parser.add_argument(
        "--output",
        default="outputs/figures/paper/gamma_operating_rule_integrated_ieee.pdf",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classification = pd.read_csv(args.classification)
    if "model" in classification.columns:
        classification = classification[classification["model"] == "iq_evm"].copy()
    classification = classification[classification["gamma_res"] <= 1.2].copy()
    operating = pd.read_csv(args.operating_rule)
    operating = operating[np.isclose(operating["delta"], 0.02)].sort_values("snr_db")

    set_ieee_trans_style()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L"],
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.5,
        }
    )
    fig, (ax_response, ax_closure) = plt.subplots(
        1, 2, figsize=(7.05, 3.32), gridspec_kw={"width_ratios": [1.0, 1.0]}
    )

    colors = ["#222222", "#D62728", "#1677C8", "#2E8B57", "#9467BD"]
    markers = ["o", "s", "^", "D", "v"]
    marker_gammas = np.arange(0.0, 1.21, 0.2)
    for index, (snr, group) in enumerate(classification.groupby("snr_db", sort=True)):
        group = group.sort_values("gamma_res")
        zero_rows = group[np.isclose(group["gamma_res"], 0.0)]
        if zero_rows.empty:
            raise ValueError(f"No gamma_res=0 reference for SNR={snr:g} dB")
        zero_accuracy = float(zero_rows.iloc[0]["accuracy"])
        loss_pp = 100.0 * (zero_accuracy - group["accuracy"].to_numpy(float))
        gamma_values = group["gamma_res"].to_numpy(float)
        marker_indices = [
            idx for idx, gamma in enumerate(gamma_values)
            if np.any(np.isclose(gamma, marker_gammas))
        ]
        ax_response.plot(
            group["gamma_res"], loss_pp,
            color=colors[index], marker=markers[index], markerfacecolor="white",
            markeredgewidth=0.65, markersize=2.6, linewidth=1.0,
            markevery=marker_indices,
            label=fr"{snr:g} dB",
        )
        operating_row = operating[np.isclose(operating["snr_db"], float(snr))]
        if not operating_row.empty:
            gamma_star = float(operating_row.iloc[0]["gamma_amc_star"])
            star_row = group[np.isclose(group["gamma_res"], gamma_star)]
            if not star_row.empty:
                star_loss = 100.0 * (zero_accuracy - float(star_row.iloc[0]["accuracy"]))
                ax_response.scatter(
                    gamma_star, star_loss, marker="*", s=43,
                    facecolor=colors[index], edgecolor="white", linewidth=0.45,
                    zorder=5,
                )
    ax_response.axhline(
        2.0, color="#555555", linestyle="--", linewidth=1.0,
        label="2-pp budget",
    )
    ax_response.scatter(
        [], [], marker="*", s=43, facecolor="#555555", edgecolor="white",
        linewidth=0.45, label=r"$\gamma^{\star}_{\rm AMC}$",
    )
    ax_response.set_xlim(0.0, 1.2)
    ax_response.set_ylim(-1.0, 25.0)
    ax_response.set_xticks(np.arange(0.0, 1.21, 0.2))
    ax_response.set_xlabel(r"Residual severity $\gamma_{\rm res}$")
    ax_response.set_ylabel(r"Worst-sign accuracy loss (pp)")
    format_ieee_axis(ax_response)
    ax_response.grid(True, linestyle="-.", linewidth=0.38, color="0.78")
    ax_response.legend(
        title="Calibration SNR", loc="upper left", ncol=2, frameon=True,
        borderpad=0.28, labelspacing=0.18, columnspacing=0.62,
        handlelength=1.35, title_fontsize=8.5, fontsize=8.5,
    )

    if args.joint_operating_rule:
        closure = pd.read_csv(args.joint_operating_rule).sort_values("snr_db")
        x = closure["snr_db"].to_numpy(float)
        point = closure["eta_amc_star"].to_numpy(float)
        lcb = closure["eta_lcb95_bonferroni_monotone"].to_numpy(float)
        grid_p90 = closure["baseline_test_joint_phase_bound_p90"].to_numpy(float)
        grid_p95 = closure["baseline_test_joint_phase_bound_p95"].to_numpy(float)
        hybrid_p90 = closure["test_joint_phase_bound_p90"].to_numpy(float)
        hybrid_p95 = closure["test_joint_phase_bound_p95"].to_numpy(float)
        closure_ylabel = r"Tail-to-$\eta^{\star}_{\rm AMC}$ ratio"
    else:
        x = operating["snr_db"].to_numpy(float)
        point = operating["e_mu_max_hz_per_s"].to_numpy(float)
        lcb = operating["e_mu_lcb95_bonferroni_monotone_hz_per_s"].to_numpy(float)
        grid_p90 = operating["baseline_test_mu_abs_error_p90"].to_numpy(float)
        grid_p95 = operating["baseline_test_mu_abs_error_p95"].to_numpy(float)
        hybrid_p90 = operating["test_mu_abs_error_p90"].to_numpy(float)
        hybrid_p95 = operating["test_mu_abs_error_p95"].to_numpy(float)
        closure_ylabel = r"Tail-to-$e_{\mu,\max}$ ratio"

    lcb_ratio = lcb / point
    grid_p90_ratio = grid_p90 / point
    grid_p95_ratio = grid_p95 / point
    hybrid_p90_ratio = hybrid_p90 / point
    hybrid_p95_ratio = hybrid_p95 / point

    ax_closure.axhline(
        1.0, color="#222222", linestyle="-", linewidth=1.15,
        label="Point limit", zorder=1,
    )
    ax_closure.plot(
        x, lcb_ratio, "o--", color="#1677C8", markerfacecolor="white",
        markeredgewidth=0.9, linewidth=1.25, markersize=3.8,
        label="Bonferroni LCB", zorder=4,
    )
    ax_closure.plot(
        x, grid_p90_ratio, "s-", color="#666666", markerfacecolor="white",
        markeredgewidth=0.9, linewidth=1.1, markersize=3.6,
        label="Coherent-grid P90", zorder=3,
    )
    ax_closure.plot(
        x, grid_p95_ratio, "^--", color="#999999", markerfacecolor="white",
        markeredgewidth=0.9, linewidth=1.1, markersize=4.0,
        label="Coherent-grid P95", zorder=3,
    )
    ax_closure.plot(
        x, hybrid_p90_ratio, "s-", color="#D55E00",
        linewidth=1.25, markersize=3.7, label="Hybrid P90", zorder=4,
    )
    ax_closure.plot(
        x, hybrid_p95_ratio, "^--", color="#E69F00",
        linewidth=1.25, markersize=4.1, label="Hybrid P95", zorder=4,
    )

    ax_closure.set_xlim(-7.7, 5.7)
    ax_closure.set_yscale("log")
    ax_closure.set_ylim(0.09, 16.5)
    ax_closure.set_xticks(x)
    ax_closure.set_xlabel("SNR (dB)")
    ax_closure.set_ylabel(closure_ylabel)
    format_ieee_axis(ax_closure)
    ax_closure.yaxis.set_major_locator(
        matplotlib.ticker.FixedLocator([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    )
    ax_closure.yaxis.set_major_formatter(
        matplotlib.ticker.FixedFormatter(["0.125", "0.25", "0.5", "1", "2", "4", "8", "16"])
    )
    ax_closure.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_closure.grid(True, linestyle="-.", linewidth=0.38, color="0.78")
    ax_closure.legend(
        loc="lower right", ncol=2, frameon=True, borderpad=0.28,
        labelspacing=0.16, columnspacing=0.52, handlelength=1.25,
        fontsize=7.2,
    )

    set_panel_aspect_10_9(ax_response)
    set_panel_aspect_10_9(ax_closure)
    fig.subplots_adjust(left=0.085, right=0.992, bottom=0.19, top=0.98, wspace=0.28)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    panel_files = save_fig2_sized_panels(fig, [ax_response, ax_closure], output)
    plt.close(fig)
    print(output)
    for path in panel_files:
        print(path)


if __name__ == "__main__":
    main()
