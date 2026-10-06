#!/usr/bin/env python
"""Plot circular-orbit Doppler-rate profiles for the paper scenario."""

from __future__ import annotations

import argparse
import json
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
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import MaxNLocator

from src.physics.leo_orbit import circular_pass_profile
from src.plotting.common import (
    IEEE_TRANS_PALETTE,
    boxed_legend,
    format_ieee_axis,
    save_axes_panels,
    set_panel_aspect_10_9,
    set_ieee_trans_style,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--carrier-ghz", type=float, default=30.0)
    p.add_argument("--altitude-km", type=float, default=600.0)
    p.add_argument("--min-elevation-deg", type=float, default=10.0)
    p.add_argument(
        "--max-elevation-deg",
        type=float,
        nargs="+",
        default=[20.0, 35.0, 50.0, 65.0, 80.0, 90.0],
    )
    p.add_argument(
        "--local-max-elevation-deg",
        type=float,
        nargs="+",
        default=[20.0, 22.0, 24.0, 26.0, 27.0, 28.0, 29.0, 30.0, 35.0, 50.0, 65.0, 80.0, 90.0],
        help="Maximum elevations used only for the local quadratic-model check.",
    )
    p.add_argument("--time-step-s", type=float, default=0.25)
    p.add_argument(
        "--local-frame-duration-ms",
        type=float,
        default=40.96,
        help="Frame duration used to quantify the unmodelled cubic phase residual.",
    )
    p.add_argument(
        "--local-time-step-ms",
        type=float,
        default=0.10,
        help="Fine circular-pass sampling interval for the local-model check.",
    )
    p.add_argument(
        "--local-start-step-s",
        type=float,
        default=0.10,
        help="Spacing between local-frame starting points along each visible pass.",
    )
    p.add_argument("--stress-mu-min", type=float, default=-8160.0)
    p.add_argument("--stress-mu-max", type=float, default=-180.0)
    p.add_argument(
        "--output",
        default="outputs/figures/paper/leo_orbit_doppler_rate_geometry.pdf",
    )
    p.add_argument(
        "--table-output",
        default="outputs/results/leo_orbit_geometry/profile_summary.csv",
    )
    return p.parse_args()


def set_style() -> None:
    set_ieee_trans_style()
    plt.rcParams.update(
        {
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
        }
    )


def _local_quadratic_model_error(
    *,
    carrier_hz: float,
    altitude_m: float,
    min_elevation_deg: float,
    max_elevation_deg: float,
    frame_duration_s: float,
    fine_step_s: float,
    start_step_s: float,
) -> dict[str, float | int]:
    """Quantify exact-Doppler phase missed by a local quadratic model.

    The phase model retains the instantaneous Doppler and Doppler rate at a
    frame start.  The integrated difference from the exact circular-pass
    Doppler is the cubic-and-higher-order residual over that frame.
    """

    profile = circular_pass_profile(
        carrier_hz=carrier_hz,
        altitude_m=altitude_m,
        max_elevation_deg=max_elevation_deg,
        min_elevation_deg=min_elevation_deg,
        time_step_s=fine_step_s,
    )
    window = max(int(round(frame_duration_s / fine_step_s)), 1)
    stride = max(int(round(start_step_s / fine_step_s)), 1)
    if profile.time_s.size <= window + 1:
        raise ValueError("Local validation frame exceeds the visible pass duration.")

    residuals: list[float] = []
    rate_values: list[float] = []
    for start in range(0, profile.time_s.size - window, stride):
        stop = start + window + 1
        time = profile.time_s[start:stop] - profile.time_s[start]
        frequency = profile.doppler_hz[start:stop]
        rate = float(profile.doppler_rate_hz_per_s[start])
        local_frequency = frequency[0] + rate * time
        delta_frequency = frequency - local_frequency
        phase_residual = np.empty_like(delta_frequency)
        phase_residual[0] = 0.0
        phase_residual[1:] = 2.0 * np.pi * np.cumsum(
            0.5 * (delta_frequency[1:] + delta_frequency[:-1]) * np.diff(time)
        )
        residuals.append(float(np.max(np.abs(phase_residual))))
        rate_values.append(rate)

    values = np.asarray(residuals, dtype=np.float64)
    return {
        "max_elevation_deg": float(max_elevation_deg),
        "frame_duration_ms": 1e3 * frame_duration_s,
        "start_count": int(values.size),
        "max_abs_phase_residual_rad": float(np.max(values)),
        "p95_abs_phase_residual_rad": float(np.quantile(values, 0.95)),
        "median_abs_phase_residual_rad": float(np.median(values)),
        "max_abs_mu_hz_per_s": float(np.max(np.abs(rate_values))),
    }


def main() -> None:
    args = parse_args()
    carrier = float(args.carrier_ghz) * 1e9
    altitude = float(args.altitude_km) * 1e3
    profiles = {}
    rows: list[dict[str, float | int]] = []
    samples: list[dict[str, float]] = []
    local_rows: list[dict[str, float | int]] = []
    for maximum in args.max_elevation_deg:
        profile = circular_pass_profile(
            carrier_hz=carrier,
            altitude_m=altitude,
            max_elevation_deg=float(maximum),
            min_elevation_deg=float(args.min_elevation_deg),
            time_step_s=float(args.time_step_s),
        )
        profiles[float(maximum)] = profile
        interior = slice(2, -2)
        mu = profile.doppler_rate_hz_per_s[interior]
        rows.append(
            {
                "carrier_ghz": float(args.carrier_ghz),
                "altitude_km": float(args.altitude_km),
                "min_elevation_deg": float(args.min_elevation_deg),
                "max_elevation_deg": float(maximum),
                "visible_duration_s": float(profile.time_s[-1] - profile.time_s[0]),
                "orbital_speed_m_per_s": float(profile.orbital_speed_m_per_s),
                "mu_min_hz_per_s": float(np.min(mu)),
                "mu_max_hz_per_s": float(np.max(mu)),
                "mu_median_hz_per_s": float(np.median(mu)),
                "stress_interval_fraction": float(
                    np.mean(
                        (mu >= float(args.stress_mu_min))
                        & (mu <= float(args.stress_mu_max))
                    )
                ),
            }
        )
        for t, elevation, doppler, rate in zip(
            profile.time_s,
            profile.elevation_deg,
            profile.doppler_hz,
            profile.doppler_rate_hz_per_s,
        ):
            samples.append(
                {
                    "max_elevation_deg": float(maximum),
                    "time_s": float(t),
                    "elevation_deg": float(elevation),
                    "doppler_hz": float(doppler),
                    "doppler_rate_hz_per_s": float(rate),
                }
            )

    for maximum in np.unique(np.asarray(args.local_max_elevation_deg, dtype=float)):
        local_rows.append(
            _local_quadratic_model_error(
                carrier_hz=carrier,
                altitude_m=altitude,
                min_elevation_deg=float(args.min_elevation_deg),
                max_elevation_deg=float(maximum),
                frame_duration_s=float(args.local_frame_duration_ms) * 1e-3,
                fine_step_s=float(args.local_time_step_ms) * 1e-3,
                start_step_s=float(args.local_start_step_s),
            )
        )

    table_path = Path(args.table_output)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(table_path, index=False)
    pd.DataFrame(samples).to_csv(table_path.with_name("profile_samples.csv"), index=False)
    local_summary = pd.DataFrame(local_rows)
    local_summary.to_csv(
        table_path.with_name("local_quadratic_approximation.csv"),
        index=False,
    )

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.32))
    colors = (
        IEEE_TRANS_PALETTE["red"],
        IEEE_TRANS_PALETTE["orange"],
        IEEE_TRANS_PALETTE["green"],
        IEEE_TRANS_PALETTE["cyan"],
        IEEE_TRANS_PALETTE["purple"],
        IEEE_TRANS_PALETTE["blue"],
    )
    styles = ("-", "-", "-", "-", "-", "-")
    markers = ("o", "s", "D", "^", "v", "P")
    for i, (maximum, profile) in enumerate(profiles.items()):
        axes[0].plot(
            profile.time_s / 60.0,
            profile.doppler_rate_hz_per_s / 1e3,
            color=colors[i % len(colors)],
            linestyle=styles[i % len(styles)],
            marker=markers[i % len(markers)],
            markevery=max(1, len(profile.time_s) // 12),
            markersize=2.3,
            markerfacecolor="white",
            markeredgewidth=0.6,
            label=fr"$e_{{\max}}={maximum:g}^\circ$",
        )
    axes[0].set_xlabel("Time from closest approach (min)")
    axes[0].set_ylabel(r"Doppler rate (kHz/s)")
    axes[0].set_xticks(np.arange(-5.0, 5.1, 1.0))
    axes[0].tick_params(axis="x", labelsize=8.1)
    format_ieee_axis(axes[0])
    set_panel_aspect_10_9(axes[0])
    boxed_legend(
        axes[0],
        loc="lower left",
        ncol=1,
        fontsize=8.5,
        labelspacing=0.22,
        borderpad=0.22,
    )

    reference = profiles[max(profiles)]
    visibility_edge_index = int(
        np.argmin(
            np.abs(
                reference.doppler_rate_hz_per_s
                - float(args.stress_mu_max)
            )
        )
    )
    visibility_edge_elevation_deg = float(
        reference.elevation_deg[visibility_edge_index]
    )
    local_plot = local_summary.sort_values("max_elevation_deg")
    elevation = local_plot["max_elevation_deg"].to_numpy(dtype=float)
    axes[1].plot(
        elevation,
        1e3 * local_plot["max_abs_phase_residual_rad"].to_numpy(dtype=float),
        color=IEEE_TRANS_PALETTE["blue"],
        marker="o",
        markerfacecolor="white",
        markeredgecolor=IEEE_TRANS_PALETTE["blue"],
        label="Maximum residual",
    )
    axes[1].plot(
        elevation,
        1e3 * local_plot["p95_abs_phase_residual_rad"].to_numpy(dtype=float),
        color=IEEE_TRANS_PALETTE["orange"],
        linestyle="-",
        marker="s",
        markerfacecolor="white",
        markeredgecolor=IEEE_TRANS_PALETTE["orange"],
        label="95th percentile",
    )
    phase_top = max(
        1.0,
        1.20 * float(
            np.max(1e3 * local_plot["max_abs_phase_residual_rad"].to_numpy(dtype=float))
        ),
    )
    if 10.0 <= phase_top:
        axes[1].axhline(
            10.0,
            color=IEEE_TRANS_PALETTE["gray"],
            linestyle=":",
            linewidth=0.9,
            label="10 mrad guide",
        )
    axes[1].set_xlabel(r"Maximum elevation $e_{\max}$ (deg)")
    axes[1].set_ylabel("Unmodelled phase (mrad)")
    axes[1].set_xticks((20.0, 30.0, 50.0, 65.0, 80.0, 90.0))
    axes[1].set_ylim(0.0, phase_top)
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
    format_ieee_axis(axes[1])
    set_panel_aspect_10_9(axes[1])
    boxed_legend(axes[1], loc="upper left")

    # A high-magnification end of the requested 20--30 degree interval resolves
    # the small maximum-versus-95th-percentile separation without altering data.
    inset = axes[1].inset_axes([0.46, 0.11, 0.46, 0.30])
    inset.plot(
        elevation,
        1e3 * local_plot["max_abs_phase_residual_rad"].to_numpy(dtype=float),
        color=IEEE_TRANS_PALETTE["blue"],
        marker="o",
        markerfacecolor="white",
        markeredgecolor=IEEE_TRANS_PALETTE["blue"],
        markersize=1.7,
        markeredgewidth=0.45,
    )
    inset.plot(
        elevation,
        1e3 * local_plot["p95_abs_phase_residual_rad"].to_numpy(dtype=float),
        color=IEEE_TRANS_PALETTE["orange"],
        linestyle="-",
        marker="s",
        markerfacecolor="white",
        markeredgecolor=IEEE_TRANS_PALETTE["orange"],
        markersize=1.7,
        markeredgewidth=0.45,
    )
    zoom = (elevation >= 27.8) & (elevation <= 30.2)
    zoom_values = np.concatenate(
        (
            1e3 * local_plot.loc[zoom, "max_abs_phase_residual_rad"].to_numpy(dtype=float),
            1e3 * local_plot.loc[zoom, "p95_abs_phase_residual_rad"].to_numpy(dtype=float),
        )
    )
    zoom_margin = max(0.002, 0.02 * float(np.ptp(zoom_values)))
    inset.set_xlim(27.8, 30.2)
    inset.set_ylim(float(np.min(zoom_values) - zoom_margin), float(np.max(zoom_values) + zoom_margin))
    inset.set_xticks((28.0, 29.0, 30.0))
    inset.yaxis.set_major_locator(MaxNLocator(nbins=3))
    inset.tick_params(
        direction="out",
        top=False,
        right=False,
        labelsize=6.8,
        width=0.5,
        length=1.8,
    )
    inset.grid(True, color="0.78", linestyle="-.", linewidth=0.28)
    for spine in inset.spines.values():
        spine.set_linewidth(0.5)
    # The dashed box and connector lines explicitly identify the enlarged
    # 28--30 degree interval in the parent panel.
    mark_inset(axes[1], inset, loc1=2, loc2=3, fc="none", ec="0.30", linewidth=0.7)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel_files = save_axes_panels(fig, axes, output)

    plt.close(fig)
    metadata = {
        "geometry": "spherical Earth, circular orbit, great-circle pass",
        "earth_rotation": "equatorial co-rotating approximation",
        "carrier_ghz": float(args.carrier_ghz),
        "altitude_km": float(args.altitude_km),
        "min_elevation_deg": float(args.min_elevation_deg),
        "stress_mu_interval_hz_per_s": [
            float(args.stress_mu_min),
            float(args.stress_mu_max),
        ],
        "visibility_edge_equivalent_elevation_deg": visibility_edge_elevation_deg,
        "local_quadratic_approximation": {
            "frame_duration_ms": float(args.local_frame_duration_ms),
            "fine_time_step_ms": float(args.local_time_step_ms),
            "start_step_s": float(args.local_start_step_s),
            "table": str(table_path.with_name("local_quadratic_approximation.csv")),
        },
    }
    table_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(local_summary.to_string(index=False))
    for panel in panel_files:
        print(f"Saved figure: {panel}")


if __name__ == "__main__":
    main()
