#!/usr/bin/env python
"""Plot pilot-aided Doppler-rate RMSE versus a coherent-pilot reference."""

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot mu RMSE-vs-SNR with an unnormalized coherent-pilot information reference."
    )
    p.add_argument("--raw-data", type=str, required=True)
    p.add_argument("--pilot-feature", type=str, required=True)
    p.add_argument("--output", type=str, default="outputs/figures/paper/mu_rmse_vs_snr_pilot_crlb.pdf")
    p.add_argument("--csv-output", type=str, default=None)
    p.add_argument("--mu-key", type=str, default="pilot_mu_hat")
    p.add_argument("--valid-key", type=str, default="pilot_valid")
    p.add_argument("--pilot-indices-key", type=str, default="pilot_indices")
    p.add_argument("--pilot-symbols-key", type=str, default="pilot_symbols")
    p.add_argument("--samples-per-symbol", type=int, default=8)
    p.add_argument(
        "--matched-filter-snr-gain-db",
        type=float,
        default=None,
        help=(
            "Sample-to-symbol SNR gain after unit-energy matched filtering. "
            "The default is 10*log10(samples_per_symbol)."
        ),
    )
    p.add_argument("--snr-bins", type=float, nargs="+", default=[-10, -7, -4, -1, 2, 5, 8, 11, 14, 17, 20])
    p.add_argument("--fig-width", type=float, default=3.45, help="Figure width in inches; 3.45 is IEEE single-column.")
    p.add_argument("--fig-height", type=float, default=3.105)
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def compute_pilot_crlb_mu_var(
    snr_linear: np.ndarray,
    pilot_times_s: np.ndarray,
    pilot_symbols: np.ndarray,
    *,
    unknown_fd0: bool,
) -> np.ndarray:
    """Compute the discrete-pilot CRLB for the unnormalized coherent model.

    The pilot-domain signal model is

        z_m = a_m exp(j(phi + 2*pi*f0*t_m + pi*mu*t_m**2)) + w_m.

    Pilot symbols and their sampling times are known. The carrier phase is a
    nuisance parameter. ``unknown_fd0=False`` matches the implemented estimator,
    whose residual frequency search is fixed to zero. ``unknown_fd0=True`` adds
    residual frequency as a nuisance parameter and provides a conservative
    joint-estimation reference.
    """

    rho = np.asarray(snr_linear, dtype=np.float64).reshape(-1)
    times = np.asarray(pilot_times_s, dtype=np.float64)
    symbols = np.asarray(pilot_symbols)
    if times.ndim == 1:
        times = np.broadcast_to(times[None, :], (rho.size, times.size))
    if symbols.ndim == 1:
        symbols = np.broadcast_to(symbols[None, :], times.shape)
    if times.shape != symbols.shape or times.shape[0] != rho.size:
        raise ValueError("Pilot times, symbols, and SNR arrays have incompatible shapes.")

    variances = np.full(rho.shape, np.nan, dtype=np.float64)
    for i, (rho_i, t_i, s_i) in enumerate(zip(rho, times, symbols)):
        valid = np.isfinite(t_i) & np.isfinite(s_i.real) & np.isfinite(s_i.imag)
        t = t_i[valid]
        weights = np.abs(s_i[valid]) ** 2
        if t.size < (3 if unknown_fd0 else 2) or rho_i <= 0:
            continue
        if unknown_fd0:
            derivatives = np.column_stack(
                [
                    np.ones_like(t),
                    2.0 * np.pi * t,
                    np.pi * t**2,
                ]
            )
        else:
            derivatives = np.column_stack(
                [
                    np.ones_like(t),
                    np.pi * t**2,
                ]
            )
        fisher = 2.0 * rho_i * (
            derivatives.T @ (weights[:, None] * derivatives)
        )
        covariance = np.linalg.pinv(fisher, rcond=1e-13)
        variances[i] = float(covariance[-1, -1])
    return variances


def set_trans_style() -> None:
    set_ieee_trans_style()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_output) if args.csv_output else out_path.with_suffix(".csv")

    with h5py.File(args.raw_data, "r") as raw, h5py.File(args.pilot_feature, "r") as feat:
        if args.mu_key not in feat:
            raise SystemExit(f"mu key not found: {args.mu_key}")
        snr_db = np.asarray(raw["snr_db"][:], dtype=np.float64)
        mu_true = np.asarray(raw["mu"][:], dtype=np.float64)
        mu_hat = np.asarray(feat[args.mu_key][:], dtype=np.float64)
        valid = np.isfinite(mu_hat)
        if args.valid_key and args.valid_key in feat:
            valid &= np.asarray(feat[args.valid_key][:], dtype=bool)
        fs = float(raw["fs"][0]) if "fs" in raw else float(feat["fs"][0])
        if args.pilot_indices_key not in raw:
            raise SystemExit(f"pilot index key not found: {args.pilot_indices_key}")
        pilot_indices = np.asarray(raw[args.pilot_indices_key][:], dtype=np.float64)
        if args.pilot_symbols_key in raw:
            pilot_symbols = np.asarray(raw[args.pilot_symbols_key][:])
        else:
            pilot_symbols = np.ones_like(pilot_indices, dtype=np.complex128)

    snr_bins = np.asarray(args.snr_bins, dtype=np.float64)
    if snr_bins.ndim != 1 or len(snr_bins) < 2:
        raise SystemExit("--snr-bins must contain at least two edges.")

    rows: list[dict[str, float | int]] = []
    err = mu_hat - mu_true
    snr_gain_db = (
        float(args.matched_filter_snr_gain_db)
        if args.matched_filter_snr_gain_db is not None
        else 10.0 * np.log10(max(int(args.samples_per_symbol), 1))
    )
    symbol_snr_linear = 10.0 ** ((snr_db + snr_gain_db) / 10.0)
    pilot_times_s = (
        pilot_indices * float(args.samples_per_symbol) / float(fs)
    )
    conditional_crlb_var = compute_pilot_crlb_mu_var(
        symbol_snr_linear,
        pilot_times_s,
        pilot_symbols,
        unknown_fd0=False,
    )
    for lo, hi in zip(snr_bins[:-1], snr_bins[1:]):
        if lo == snr_bins[0]:
            mask = (snr_db >= lo) & (snr_db <= hi) & valid
        else:
            mask = (snr_db > lo) & (snr_db <= hi) & valid
        if not np.any(mask):
            continue
        rmse = float(np.sqrt(np.mean(err[mask] ** 2)))
        mae = float(np.mean(np.abs(err[mask])))
        bias = float(np.mean(err[mask]))
        conditional_crlb_rmse = float(
            np.sqrt(np.mean(conditional_crlb_var[mask]))
        )
        rows.append(
            {
                "snr_low_db": float(lo),
                "snr_high_db": float(hi),
                "snr_center_db": float(0.5 * (lo + hi)),
                "count": int(np.sum(mask)),
                "pilot_rmse_hz_per_s": rmse,
                "pilot_mae_hz_per_s": mae,
                "pilot_bias_hz_per_s": bias,
                "conditional_crlb_rmse_hz_per_s": conditional_crlb_rmse,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(csv_path, index=False)

    set_trans_style()
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

    x = table["snr_center_db"].to_numpy()
    ax.semilogy(
        x,
        table["pilot_rmse_hz_per_s"].to_numpy(),
        color=IEEE_TRANS_PALETTE["black"],
        marker="o",
        markerfacecolor="white",
        markeredgecolor=IEEE_TRANS_PALETTE["black"],
        markeredgewidth=0.8,
        label="Estimator",
    )
    ax.semilogy(
        x,
        table["conditional_crlb_rmse_hz_per_s"].to_numpy(),
        color=IEEE_TRANS_PALETTE["red"],
        linestyle="--",
        marker="s",
        markerfacecolor=IEEE_TRANS_PALETTE["red"],
        markeredgecolor=IEEE_TRANS_PALETTE["red"],
        label="Coherent-pilot information reference",
    )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(r"RMSE of $\hat{\mu}$ (Hz/s)")
    ax.set_xlim(float(snr_bins[0]), float(snr_bins[-1]))
    ax.set_xticks(snr_bins)
    format_ieee_axis(ax)
    set_panel_aspect_10_9(ax)
    ax.grid(True, which="minor", axis="y", linestyle=":", alpha=0.3)
    boxed_legend(ax, loc="upper right", fontsize=8.0)

    fig.savefig(out_path)
    plt.close(fig)

    print(table.to_string(index=False))
    print(f"Saved figure: {out_path}")
    print(f"Saved table: {csv_path}") 


if __name__ == "__main__":
    main()
