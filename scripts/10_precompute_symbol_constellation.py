#!/usr/bin/env python
"""Precompute label-free all-candidate symbol-level constellation features.

This script uses a pilot/preamble-aided receiver front-end only for
compensation, timing, and complex-gain correction.  It does not use the true
modulation label when building features.  Instead, it computes a constellation
density image from data symbols and an all-candidate EVM feature vector.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
from tqdm import tqdm

from src.drc_hoc.compensation import compensate_full_doppler
from src.signal.modulation import available_modulations, get_constellation
from src.signal.pulse_shape import rrc_filter
from src.utils.io import save_json
from src.utils.math_utils import to_complex


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute symbol-level constellation features.")
    p.add_argument("--raw-data", type=str, default=None)
    p.add_argument("--mu-feature-data", type=str, default=None)
    p.add_argument(
        "--input-iq-feature-data",
        type=str,
        default=None,
        help="Optional HDF5 containing already compensated/equalized iq_comp; skips Doppler compensation.",
    )
    p.add_argument("--output", type=str, required=True)
    p.add_argument(
        "--reweight-existing",
        type=str,
        default=None,
        help=(
            "Reuse an existing symbol-feature HDF5 and update only the "
            "SNR-dependent complexity-penalty fields. This avoids repeating "
            "matched filtering and all-candidate distance calculations."
        ),
    )
    p.add_argument("--mu-key", type=str, default="pilot_mu_hat")
    p.add_argument("--valid-key", type=str, default="pilot_valid")
    p.add_argument("--fd0-key", type=str, default=None)
    p.add_argument("--sample-rate-hz", type=float, default=None)
    p.add_argument("--samples-per-symbol", type=int, default=8)
    p.add_argument("--rrc-beta", type=float, default=0.35)
    p.add_argument("--rrc-span", type=int, default=8)
    p.add_argument("--timing-phases", type=int, nargs="*", default=None)
    p.add_argument("--modulations", type=str, nargs="*", default=None)
    p.add_argument("--grid-size", type=int, default=64)
    p.add_argument("--grid-limit", type=float, default=2.0)
    p.add_argument("--image-scale", type=str, default="sqrt", choices=["none", "sqrt", "log"])
    p.add_argument("--phase-grid-size", type=int, default=1)
    p.add_argument("--symbol-trim", type=int, default=8)
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--no-center", action="store_true", help="Do not remove residual symbol mean before feature extraction.")
    p.add_argument("--no-power-normalize", action="store_true", help="Do not normalize equalized data symbols to unit power.")
    p.add_argument("--min-data-symbols", type=int, default=32)
    p.add_argument(
        "--candidate-score-mode",
        choices=["exact_mixture", "logistic"],
        default="exact_mixture",
        help=(
            "Candidate-template score used in the deployed descriptor. "
            "exact_mixture is parameter-free apart from the receiver-derived "
            "pilot noise variance; logistic is retained only for legacy ablations."
        ),
    )
    p.add_argument(
        "--complexity-penalty-lambda",
        type=float,
        default=0.01,
        help=(
            "Low-SNR modulation-order penalty strength added to candidate EVM scores. "
            "The effective penalty is lambda*sigmoid((threshold-SNR)/scale)*log(M)."
        ),
    )
    p.add_argument(
        "--complexity-penalty-snr-threshold",
        type=float,
        default=-3.0,
        help="SNR threshold in dB for low-SNR complexity penalty.",
    )
    p.add_argument(
        "--complexity-penalty-snr-scale",
        type=float,
        default=4.0,
        help="Positive SNR transition width in dB for low-SNR complexity penalty.",
    )
    p.add_argument(
        "--snr-source",
        type=str,
        default="pilot_evm",
        choices=["pilot_evm", "true", "none"],
        help=(
            "SNR used by the low-SNR complexity penalty. 'pilot_evm' is the "
            "receiver-realizable default, 'true' is retained only for controlled "
            "simulation ablations, and 'none' applies the maximum configured penalty."
        ),
    )
    p.add_argument(
        "--pilot-snr-offset-db",
        type=float,
        default=None,
        help=(
            "Offset added to -10*log10(pilot_evm). By default, the matched-filter "
            "processing gain 10*log10(samples_per_symbol) is removed so that the "
            "estimate matches the sample-domain SNR used by the simulator."
        ),
    )
    p.add_argument(
        "--apsk-ring-penalty-lambda",
        type=float,
        default=0.012,
        help=(
            "Low-SNR APSK ring-count penalty strength. Effective term is "
            "lambda*sigmoid((threshold-SNR)/scale)*R."
        ),
    )
    p.add_argument(
        "--apsk-phase-weight",
        type=float,
        default=1.0,
        help="Weight for intra-ring phase EVM in APSK complexity score.",
    )
    p.add_argument(
        "--apsk-occupancy-weight",
        type=float,
        default=0.25,
        help="Weight for ring occupancy chi-square in APSK complexity score.",
    )
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _decode_label_names(raw: h5py.File) -> list[str]:
    if "label_names" not in raw:
        return []
    out: list[str] = []
    for item in raw["label_names"][:]:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def _resolve_modulations(raw: h5py.File, requested: list[str] | None) -> tuple[str, ...]:
    if requested:
        mods = [m.upper() for m in requested]
    else:
        labels = _decode_label_names(raw)
        mods = [m.upper() for m in labels] if labels else available_modulations()
    valid = set(available_modulations())
    unknown = [m for m in mods if m not in valid]
    if unknown:
        raise SystemExit(f"Unsupported modulation(s): {unknown}. Available: {sorted(valid)}")
    return tuple(mods)


def _copy_meta(raw: h5py.File, out: h5py.File, keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in raw and key not in out:
            out.create_dataset(key, data=raw[key][:], compression="gzip")


def _read_valid(mu_file: h5py.File, key: str | None, i: int, mu_hat: float) -> bool:
    valid = bool(np.isfinite(mu_hat))
    if key and key in mu_file:
        valid = valid and bool(mu_file[key][i])
    return valid


def _estimate_gain_and_pilot_evm(
    matched: np.ndarray,
    pilot_indices: np.ndarray,
    pilot_symbols: np.ndarray,
    timing: int,
    sps: int,
) -> tuple[complex, float, int]:
    sym_idx = np.asarray(pilot_indices, dtype=np.int64).reshape(-1)
    pilot = np.asarray(pilot_symbols, dtype=np.complex128).reshape(-1)
    if sym_idx.size != pilot.size:
        return 1.0 + 0.0j, np.inf, 0

    sample_idx = sym_idx * sps + int(timing)
    valid = (sample_idx >= 0) & (sample_idx < len(matched)) & (np.abs(pilot) > 1e-12)
    sample_idx = sample_idx[valid]
    pilot = pilot[valid]
    if sample_idx.size == 0:
        return 1.0 + 0.0j, np.inf, 0

    obs = matched[sample_idx]
    gain = np.sum(obs * np.conj(pilot)) / (np.sum(np.abs(pilot) ** 2) + 1e-12)
    if not np.isfinite(gain) or abs(gain) < 1e-12:
        gain = 1.0 + 0.0j
    eq = obs / gain
    evm = float(np.mean(np.abs(eq - pilot) ** 2))
    return complex(gain), evm, int(sample_idx.size)


def _estimate_crossfit_pilot_evm(
    matched: np.ndarray,
    pilot_indices: np.ndarray,
    pilot_symbols: np.ndarray,
    timing: int,
    sps: int,
) -> tuple[float, int]:
    """Return two-fold cross-fitted pilot EVM for receiver-side SNR.

    Each pilot residual is evaluated with a complex gain estimated from the
    opposite alternating pilot fold.  This removes the optimistic in-sample
    gain-fit bias from the SNR proxy used by the low-SNR EVM penalties.
    """

    sym_idx = np.asarray(pilot_indices, dtype=np.int64).reshape(-1)
    pilot = np.asarray(pilot_symbols, dtype=np.complex128).reshape(-1)
    if sym_idx.size != pilot.size:
        return np.inf, 0
    sample_idx = sym_idx * sps + int(timing)
    valid = (sample_idx >= 0) & (sample_idx < len(matched)) & (np.abs(pilot) > 1e-12)
    sample_idx = sample_idx[valid]
    pilot = pilot[valid]
    if sample_idx.size < 4:
        return np.inf, int(sample_idx.size)

    obs = matched[sample_idx]
    squared_errors: list[np.ndarray] = []
    for held_out in (np.arange(sample_idx.size) % 2 == 0, np.arange(sample_idx.size) % 2 == 1):
        train = ~held_out
        if np.sum(train) < 2 or not np.any(held_out):
            continue
        gain = np.sum(obs[train] * np.conj(pilot[train])) / (
            np.sum(np.abs(pilot[train]) ** 2) + 1e-12
        )
        if not np.isfinite(gain) or abs(gain) < 1e-12:
            continue
        eq = obs[held_out] / gain
        squared_errors.append(np.abs(eq - pilot[held_out]) ** 2)
    if not squared_errors:
        return np.inf, int(sample_idx.size)
    return float(np.mean(np.concatenate(squared_errors))), int(sample_idx.size)


def _extract_equalized_symbols(
    matched: np.ndarray,
    gain: complex,
    timing: int,
    sps: int,
    pilot_indices: np.ndarray,
    symbol_trim: int,
    max_symbols: int,
) -> np.ndarray:
    num_symbols = len(matched) // sps
    if num_symbols <= 0:
        return np.asarray([], dtype=np.complex128)

    symbol_idx = np.arange(num_symbols, dtype=np.int64)
    sample_idx = symbol_idx * sps + int(timing)
    valid = sample_idx < len(matched)
    if symbol_trim > 0:
        valid &= (symbol_idx >= symbol_trim) & (symbol_idx < num_symbols - symbol_trim)

    pilot_set = set(np.asarray(pilot_indices, dtype=np.int64).reshape(-1).tolist())
    if pilot_set:
        is_pilot = np.asarray([int(x) in pilot_set for x in symbol_idx], dtype=bool)
        valid &= ~is_pilot

    sample_idx = sample_idx[valid]
    symbols = matched[sample_idx] / (gain + 1e-12)
    if max_symbols > 0 and symbols.size > max_symbols:
        keep = np.linspace(0, symbols.size - 1, max_symbols, dtype=np.int64)
        symbols = symbols[keep]
    return symbols.astype(np.complex128)


def _prepare_symbols(
    symbols: np.ndarray,
    center: bool,
    power_normalize: bool,
) -> np.ndarray:
    z = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if z.size == 0:
        return z
    if center:
        z = z - np.mean(z)
    if power_normalize:
        z = z / (np.sqrt(np.mean(np.abs(z) ** 2)) + 1e-12)
    return z.astype(np.complex128)


def _constellation_image(
    symbols: np.ndarray,
    grid_size: int,
    grid_limit: float,
    image_scale: str,
) -> np.ndarray:
    if symbols.size == 0:
        return np.zeros((1, grid_size, grid_size), dtype=np.float32)

    lim = float(abs(grid_limit))
    hist, _, _ = np.histogram2d(
        np.real(symbols),
        np.imag(symbols),
        bins=int(grid_size),
        range=[[-lim, lim], [-lim, lim]],
    )
    hist = hist.T.astype(np.float32)
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)

    if image_scale == "sqrt":
        hist = np.sqrt(hist)
    elif image_scale == "log":
        hist = np.log1p(hist * float(max(symbols.size, 1)))
    elif image_scale != "none":
        raise ValueError(f"Unsupported image_scale: {image_scale}")

    max_val = float(np.max(hist))
    if max_val > 0:
        hist = hist / max_val
    return hist[None, :, :].astype(np.float32)


def _score_evm_all_candidates(
    symbols: np.ndarray,
    modulations: tuple[str, ...],
    constellations: dict[str, np.ndarray],
    phase_grid: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    evm = []
    best_phase = []
    if symbols.size == 0:
        return (
            np.full(len(modulations), np.inf, dtype=np.float32),
            [float("nan") for _ in modulations],
        )
    for modulation in modulations:
        constellation = constellations[modulation]
        best_score = np.inf
        best_angle = 0.0
        for phase in phase_grid:
            rotated = symbols * np.exp(-1j * float(phase))
            distances = np.abs(rotated[:, None] - constellation[None, :]) ** 2
            score = float(np.mean(np.min(distances, axis=1)))
            if score < best_score:
                best_score = score
                best_angle = float(phase)
        evm.append(best_score)
        best_phase.append(best_angle)
    return np.asarray(evm, dtype=np.float32), best_phase


def _wrap_angle(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _constellation_rings(constellation: np.ndarray) -> list[dict[str, np.ndarray | float]]:
    radii = np.abs(np.asarray(constellation, dtype=np.complex128).reshape(-1))
    rounded = np.round(radii, decimals=8)
    ring_values = np.unique(rounded)
    rings: list[dict[str, np.ndarray | float]] = []
    for value in np.sort(ring_values):
        mask = rounded == value
        points = constellation[mask]
        rings.append(
            {
                "radius": float(np.mean(np.abs(points))),
                "phases": np.sort(np.angle(points)),
                "count": int(points.size),
            }
        )
    return rings


def _snr_weight(snr_db: float, base_lambda: float, threshold_db: float, scale_db: float) -> float:
    if base_lambda <= 0.0 or not np.isfinite(base_lambda):
        return 0.0
    if not np.isfinite(snr_db):
        return float(base_lambda)
    scale = max(float(abs(scale_db)), 1e-6)
    x = np.clip((float(threshold_db) - float(snr_db)) / scale, -60.0, 60.0)
    return float(base_lambda / (1.0 + np.exp(-x)))


def _pilot_sample_snr_db(
    pilot_evm: float,
    samples_per_symbol: int,
    offset_db: float | None,
) -> float:
    """Estimate sample-domain SNR from matched-filter pilot EVM."""

    evm = float(pilot_evm)
    if not np.isfinite(evm) or evm <= 0.0:
        return np.nan
    if offset_db is None:
        offset_db = -10.0 * np.log10(max(int(samples_per_symbol), 1))
    return float(-10.0 * np.log10(max(evm, 1e-12)) + float(offset_db))


def _select_penalty_snr(
    source: str,
    true_snr_db: float,
    pilot_snr_db: float,
) -> float:
    if source == "pilot_evm":
        return float(pilot_snr_db)
    if source == "true":
        return float(true_snr_db)
    if source == "none":
        return np.nan
    raise ValueError(f"Unsupported SNR source: {source}")


def _decode_strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in values
    )


def _reweight_existing(args: argparse.Namespace, output: Path) -> None:
    """Rebuild only SNR-dependent EVM fields from stored label-free scores."""

    source = Path(args.reweight_existing)
    if not source.exists():
        raise SystemExit(f"Existing symbol-feature file not found: {source}")
    if source.resolve() == output.resolve():
        raise SystemExit("--reweight-existing and --output must be different files.")

    with h5py.File(source, "r") as src:
        required = {
            "all_candidate_evm",
            "candidate_modulations",
            "evm_features",
            "evm_feature_names",
            "pilot_evm",
        }
        missing = sorted(required.difference(src.keys()))
        if missing:
            raise SystemExit(f"Existing symbol-feature file is missing: {missing}")

        modulations = _decode_strings(src["candidate_modulations"][:])
        names = _decode_strings(src["evm_feature_names"][:])
        name_to_index = {name: i for i, name in enumerate(names)}
        constellations = {m: get_constellation(m).astype(np.complex128) for m in modulations}
        orders = np.asarray([max(len(constellations[m]), 1) for m in modulations], dtype=np.float64)
        log_orders = np.log(orders)
        ring_counts = {
            m: len(_constellation_rings(constellations[m]))
            for m in modulations
        }

        evm = src["all_candidate_evm"][:].astype(np.float64)
        features = src["evm_features"][:].astype(np.float64)
        pilot_evm = src["pilot_evm"][:].astype(np.float64)
        true_snr = (
            src["snr_db"][:].astype(np.float64)
            if "snr_db" in src
            else np.full(evm.shape[0], np.nan, dtype=np.float64)
        )
        offset_db = (
            float(args.pilot_snr_offset_db)
            if args.pilot_snr_offset_db is not None
            else float(-10.0 * np.log10(max(int(args.samples_per_symbol), 1)))
        )
        pilot_snr = np.where(
            np.isfinite(pilot_evm) & (pilot_evm > 0.0),
            -10.0 * np.log10(np.maximum(pilot_evm, 1e-12)) + offset_db,
            np.nan,
        )
        if args.snr_source == "pilot_evm":
            penalty_snr = pilot_snr
        elif args.snr_source == "true":
            penalty_snr = true_snr
        else:
            penalty_snr = np.full(evm.shape[0], np.nan, dtype=np.float64)

        lambda_eff = np.asarray(
            [
                _snr_weight(
                    value,
                    args.complexity_penalty_lambda,
                    args.complexity_penalty_snr_threshold,
                    args.complexity_penalty_snr_scale,
                )
                for value in penalty_snr
            ],
            dtype=np.float64,
        )
        ring_lambda_eff = np.asarray(
            [
                _snr_weight(
                    value,
                    args.apsk_ring_penalty_lambda,
                    args.complexity_penalty_snr_threshold,
                    args.complexity_penalty_snr_scale,
                )
                for value in penalty_snr
            ],
            dtype=np.float64,
        )
        penalized = evm + lambda_eff[:, None] * log_orders[None, :]
        penalized_sorted = np.sort(penalized, axis=1)
        penalized_margin = (penalized_sorted[:, 1] - penalized_sorted[:, 0]) / (
            np.abs(penalized_sorted[:, 1]) + 1e-12
        )

        for j, modulation in enumerate(modulations):
            features[:, name_to_index[f"penalized_evm_{modulation}"]] = penalized[:, j]

        def column(name: str) -> np.ndarray:
            return features[:, name_to_index[name]]

        def penalized_column(modulation: str) -> np.ndarray:
            if modulation not in modulations:
                return np.zeros(evm.shape[0], dtype=np.float64)
            return penalized[:, modulations.index(modulation)]

        features[:, name_to_index["snr_complexity_lambda_eff"]] = lambda_eff
        features[:, name_to_index["apsk_ring_lambda_eff"]] = ring_lambda_eff
        features[:, name_to_index["penalized_evm_min"]] = penalized_sorted[:, 0]
        features[:, name_to_index["penalized_evm_second"]] = penalized_sorted[:, 1]
        features[:, name_to_index["penalized_evm_margin"]] = penalized_margin

        p16qam = penalized_column("16QAM")
        p64qam = penalized_column("64QAM")
        p16apsk = penalized_column("16APSK")
        p32apsk = penalized_column("32APSK")
        features[:, name_to_index["penalized_evm_16qam_minus_64qam"]] = p16qam - p64qam
        features[:, name_to_index["penalized_evm_16qam_div_64qam"]] = p16qam / (p64qam + 1e-12)
        features[:, name_to_index["penalized_evm_16apsk_minus_32apsk"]] = p16apsk - p32apsk
        features[:, name_to_index["penalized_evm_16apsk_div_32apsk"]] = p16apsk / (p32apsk + 1e-12)

        def apsk_complexity(modulation: str, prefix: str) -> np.ndarray:
            if modulation not in modulations:
                return np.zeros(evm.shape[0], dtype=np.float64)
            return (
                column(f"{prefix}_ring_radius_evm")
                + float(args.apsk_phase_weight) * column(f"{prefix}_intra_ring_phase_evm")
                + float(args.apsk_occupancy_weight) * column(f"{prefix}_ring_occupancy_chi2")
                + ring_lambda_eff * float(ring_counts[modulation])
                + lambda_eff * np.log(float(max(len(constellations[modulation]), 1)))
            )

        apsk16 = apsk_complexity("16APSK", "apsk16")
        apsk32 = apsk_complexity("32APSK", "apsk32")
        features[:, name_to_index["apsk16_complexity_score"]] = apsk16
        features[:, name_to_index["apsk32_complexity_score"]] = apsk32
        features[:, name_to_index["apsk_complexity_16_minus_32"]] = apsk16 - apsk32

        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        with h5py.File(output, "w") as out:
            for key in src.keys():
                src.copy(key, out)
            for key in (
                "evm_features",
                "all_candidate_penalized_evm",
                "penalized_evm_best_index",
                "penalized_evm_margin",
                "snr_complexity_lambda_eff",
                "apsk_ring_lambda_eff",
            ):
                del out[key]
            out.create_dataset("evm_features", data=features.astype(np.float32))
            out.create_dataset(
                "all_candidate_penalized_evm",
                data=penalized.astype(np.float32),
            )
            out.create_dataset(
                "penalized_evm_best_index",
                data=np.argmin(penalized, axis=1).astype(np.int64),
            )
            out.create_dataset(
                "penalized_evm_margin",
                data=penalized_margin.astype(np.float32),
            )
            out.create_dataset(
                "snr_complexity_lambda_eff",
                data=lambda_eff.astype(np.float32),
            )
            out.create_dataset(
                "apsk_ring_lambda_eff",
                data=ring_lambda_eff.astype(np.float32),
            )
            for key, values in (
                ("pilot_snr_est_db", pilot_snr),
                ("penalty_snr_db", penalty_snr),
            ):
                if key in out:
                    del out[key]
                out.create_dataset(key, data=values.astype(np.float32))

            old_config = json.loads(src.attrs.get("config_json", "{}"))
            old_config.update(
                {
                    "reweighted_from": str(source),
                    "snr_source": args.snr_source,
                    "pilot_snr_offset_db": offset_db,
                    "complexity_penalty_lambda": args.complexity_penalty_lambda,
                    "complexity_penalty_snr_threshold": args.complexity_penalty_snr_threshold,
                    "complexity_penalty_snr_scale": args.complexity_penalty_snr_scale,
                    "apsk_ring_penalty_lambda": args.apsk_ring_penalty_lambda,
                    "apsk_phase_weight": args.apsk_phase_weight,
                    "apsk_occupancy_weight": args.apsk_occupancy_weight,
                }
            )
            out.attrs["config_json"] = json.dumps(old_config)

    valid_snr = np.isfinite(pilot_snr) & np.isfinite(true_snr)
    error = pilot_snr[valid_snr] - true_snr[valid_snr]
    summary = {
        "output": str(output),
        "reweighted_from": str(source),
        "num_samples": int(evm.shape[0]),
        "evm_dim": int(features.shape[1]),
        "candidate_modulations": list(modulations),
        "snr_source": args.snr_source,
        "pilot_snr_offset_db": offset_db,
        "pilot_snr_mae_db": float(np.mean(np.abs(error))) if error.size else np.nan,
        "pilot_snr_rmse_db": float(np.sqrt(np.mean(error**2))) if error.size else np.nan,
        "pilot_snr_bias_db": float(np.mean(error)) if error.size else np.nan,
    }
    save_json(summary, output.with_suffix(".summary.json"))
    print(summary)


def _penalize_evm_by_complexity(
    evm: np.ndarray,
    modulations: tuple[str, ...],
    constellations: dict[str, np.ndarray],
    snr_db: float,
    complexity_penalty_lambda: float,
    complexity_penalty_snr_threshold: float,
    complexity_penalty_snr_scale: float,
) -> tuple[np.ndarray, float]:
    lambda_eff = _snr_weight(
        snr_db,
        complexity_penalty_lambda,
        complexity_penalty_snr_threshold,
        complexity_penalty_snr_scale,
    )
    constellation_sizes = np.asarray([max(len(constellations[m]), 1) for m in modulations], dtype=np.float64)
    log_orders = np.log(constellation_sizes)
    penalized = np.where(np.isfinite(evm), evm.astype(np.float64) + lambda_eff * log_orders, np.inf)
    return penalized.astype(np.float32), lambda_eff


def _apsk_ring_metrics(
    symbols: np.ndarray,
    constellation: np.ndarray,
    best_phase: float,
) -> tuple[float, float, float, float]:
    """Return radius-fit EVM, intra-ring phase EVM, ring separation, occupancy fit."""

    if symbols.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    rings = _constellation_rings(constellation)
    if not rings:
        return 0.0, 0.0, 0.0, 0.0

    z = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if np.isfinite(best_phase):
        z = z * np.exp(-1j * float(best_phase))

    abs_z = np.abs(z)
    phases = np.angle(z)
    ring_radii = np.asarray([float(ring["radius"]) for ring in rings], dtype=np.float64)
    assign = np.argmin(np.abs(abs_z[:, None] - ring_radii[None, :]), axis=1)
    assigned_radii = ring_radii[assign]

    radius_evm = float(np.mean((abs_z - assigned_radii) ** 2) / (np.mean(assigned_radii**2) + 1e-12))

    phase_terms = np.zeros_like(abs_z, dtype=np.float64)
    for ring_idx, ring in enumerate(rings):
        mask = assign == ring_idx
        if not np.any(mask):
            continue
        ring_phases = np.asarray(ring["phases"], dtype=np.float64)
        diff = _wrap_angle(phases[mask, None] - ring_phases[None, :])
        min_diff = np.min(np.abs(diff), axis=1)
        phase_terms[mask] = (ring_radii[ring_idx] ** 2) * (min_diff**2)
    phase_evm = float(np.mean(phase_terms) / (np.mean(assigned_radii**2) + 1e-12))

    observed_means: list[float] = []
    observed_stds: list[float] = []
    min_count = max(3, int(0.02 * abs_z.size))
    for ring_idx in range(len(rings)):
        vals = abs_z[assign == ring_idx]
        if vals.size >= min_count:
            observed_means.append(float(np.mean(vals)))
            observed_stds.append(float(np.std(vals)))

    if len(observed_means) < 2:
        separation = 0.0
    else:
        means = np.sort(np.asarray(observed_means, dtype=np.float64))
        min_gap = float(np.min(np.diff(means)))
        within = float(np.mean(observed_stds)) if observed_stds else 0.0
        separation = min_gap / (within + 1e-12)

    observed_probs = np.bincount(assign, minlength=len(rings)).astype(np.float64)
    observed_probs = observed_probs / (float(abs_z.size) + 1e-12)
    expected_counts = np.asarray([float(ring["count"]) for ring in rings], dtype=np.float64)
    expected_probs = expected_counts / (float(np.sum(expected_counts)) + 1e-12)
    occupancy_chi2 = float(np.sum((observed_probs - expected_probs) ** 2 / (expected_probs + 1e-12)))

    return radius_evm, phase_evm, float(separation), occupancy_chi2


def _evm_feature_vector(
    evm: np.ndarray,
    modulations: tuple[str, ...],
    pilot_evm: float,
    symbols: np.ndarray,
    constellations: dict[str, np.ndarray],
    phase_best: list[float] | np.ndarray,
    snr_db: float = np.nan,
    complexity_penalty_lambda: float = 0.0,
    complexity_penalty_snr_threshold: float = 0.0,
    complexity_penalty_snr_scale: float = 3.0,
    apsk_ring_penalty_lambda: float = 0.0,
    apsk_phase_weight: float = 1.0,
    apsk_occupancy_weight: float = 0.25,
    candidate_score_mode: str = "logistic",
) -> tuple[np.ndarray, list[str]]:
    finite = np.where(np.isfinite(evm), evm, np.nan)
    order = np.argsort(np.where(np.isfinite(evm), evm, np.inf))
    evm_min = float(evm[order[0]]) if order.size else np.inf
    evm_second = float(evm[order[1]]) if order.size > 1 else evm_min
    evm_margin = float((evm_second - evm_min) / (abs(evm_second) + 1e-12)) if np.isfinite(evm_second) else 0.0

    def get_evm(name: str) -> float:
        if name in modulations:
            return float(evm[modulations.index(name)])
        return 0.0

    evm16 = get_evm("16QAM")
    evm64 = get_evm("64QAM")
    evm16apsk = get_evm("16APSK")
    evm32apsk = get_evm("32APSK")
    if candidate_score_mode == "exact_mixture":
        sigma2 = max(float(pilot_evm), 1e-8)
        exact_scores = []
        for idx, modulation in enumerate(modulations):
            rotated = np.asarray(symbols, dtype=np.complex128) * np.exp(-1j * float(phase_best[idx]))
            distances = np.abs(rotated[:, None] - constellations[modulation][None, :]) ** 2
            scaled = -distances / sigma2
            row_max = np.max(scaled, axis=1)
            log_mix = row_max + np.log(
                np.mean(np.exp(scaled - row_max[:, None]), axis=1) + 1e-300
            )
            exact_scores.append(float(-sigma2 * np.mean(log_mix)))
        penalized = np.asarray(exact_scores, dtype=np.float64)
        lambda_eff = 0.0
    elif candidate_score_mode == "logistic":
        penalized, lambda_eff = _penalize_evm_by_complexity(
            evm,
            modulations,
            constellations,
            snr_db,
            complexity_penalty_lambda,
            complexity_penalty_snr_threshold,
            complexity_penalty_snr_scale,
        )
    else:
        raise ValueError(f"Unsupported candidate_score_mode: {candidate_score_mode}")
    penalized_finite = np.where(np.isfinite(penalized), penalized, np.nan)
    penalized_order = np.argsort(np.where(np.isfinite(penalized), penalized, np.inf))
    penalized_min = float(penalized[penalized_order[0]]) if penalized_order.size else np.inf
    penalized_second = float(penalized[penalized_order[1]]) if penalized_order.size > 1 else penalized_min
    penalized_margin = (
        float((penalized_second - penalized_min) / (abs(penalized_second) + 1e-12))
        if np.isfinite(penalized_second)
        else 0.0
    )

    def get_penalized(name: str) -> float:
        if name in modulations:
            return float(penalized[modulations.index(name)])
        return 0.0

    def get_phase(name: str) -> float:
        if name in modulations:
            idx = modulations.index(name)
            if idx < len(phase_best):
                return float(phase_best[idx])
        return 0.0

    apsk16_radius, apsk16_phase, apsk16_sep, apsk16_occ = (
        _apsk_ring_metrics(symbols, constellations["16APSK"], get_phase("16APSK"))
        if "16APSK" in constellations
        else (0.0, 0.0, 0.0, 0.0)
    )
    apsk32_radius, apsk32_phase, apsk32_sep, apsk32_occ = (
        _apsk_ring_metrics(symbols, constellations["32APSK"], get_phase("32APSK"))
        if "32APSK" in constellations
        else (0.0, 0.0, 0.0, 0.0)
    )
    ring_counts = {m: len(_constellation_rings(c)) for m, c in constellations.items()}
    apsk_ring_lambda_eff = (
        0.0
        if candidate_score_mode == "exact_mixture"
        else _snr_weight(
            snr_db,
            apsk_ring_penalty_lambda,
            complexity_penalty_snr_threshold,
            complexity_penalty_snr_scale,
        )
    )
    if "16APSK" in constellations:
        apsk16_complexity = (
            apsk16_radius
            + float(apsk_phase_weight) * apsk16_phase
            + float(apsk_occupancy_weight) * apsk16_occ
            + apsk_ring_lambda_eff * float(ring_counts.get("16APSK", 0))
            + lambda_eff * np.log(float(max(len(constellations["16APSK"]), 1)))
        )
    else:
        apsk16_complexity = 0.0
    if "32APSK" in constellations:
        apsk32_complexity = (
            apsk32_radius
            + float(apsk_phase_weight) * apsk32_phase
            + float(apsk_occupancy_weight) * apsk32_occ
            + apsk_ring_lambda_eff * float(ring_counts.get("32APSK", 0))
            + lambda_eff * np.log(float(max(len(constellations["32APSK"]), 1)))
        )
    else:
        apsk32_complexity = 0.0
    pevm16 = get_penalized("16QAM")
    pevm64 = get_penalized("64QAM")
    pevm16apsk = get_penalized("16APSK")
    pevm32apsk = get_penalized("32APSK")
    abs_z = np.abs(symbols) if symbols.size else np.asarray([0.0])
    extras = np.asarray(
        [
            evm_min,
            evm_second,
            evm_margin,
            evm16 - evm64,
            evm16 / (evm64 + 1e-12),
            float(pilot_evm) if np.isfinite(pilot_evm) else 0.0,
            float(np.mean(abs_z)),
            float(np.std(abs_z)),
            float(np.quantile(abs_z, 0.25)),
            float(np.quantile(abs_z, 0.75)),
            evm16apsk - evm32apsk,
            evm16apsk / (evm32apsk + 1e-12),
            apsk16_radius,
            apsk16_phase,
            apsk16_sep,
            apsk32_radius,
            apsk32_phase,
            apsk32_sep,
            apsk16_radius - apsk32_radius,
            apsk16_phase - apsk32_phase,
            apsk32_sep - apsk16_sep,
            apsk16_occ,
            apsk32_occ,
            apsk16_occ - apsk32_occ,
            float(lambda_eff),
            float(apsk_ring_lambda_eff),
            penalized_min,
            penalized_second,
            penalized_margin,
            pevm16 - pevm64,
            pevm16 / (pevm64 + 1e-12),
            pevm16apsk - pevm32apsk,
            pevm16apsk / (pevm32apsk + 1e-12),
            float(apsk16_complexity),
            float(apsk32_complexity),
            float(apsk16_complexity - apsk32_complexity),
        ],
        dtype=np.float32,
    )
    names = [f"evm_{m}" for m in modulations] + [
        "evm_min",
        "evm_second",
        "evm_margin",
        "evm_16qam_minus_64qam",
        "evm_16qam_div_64qam",
        "pilot_evm",
        "abs_mean",
        "abs_std",
        "abs_q25",
        "abs_q75",
        "evm_16apsk_minus_32apsk",
        "evm_16apsk_div_32apsk",
        "apsk16_ring_radius_evm",
        "apsk16_intra_ring_phase_evm",
        "apsk16_inter_ring_sep",
        "apsk32_ring_radius_evm",
        "apsk32_intra_ring_phase_evm",
        "apsk32_inter_ring_sep",
        "apsk_ring_radius_16_minus_32",
        "apsk_phase_16_minus_32",
        "apsk_sep_32_minus_16",
        "apsk16_ring_occupancy_chi2",
        "apsk32_ring_occupancy_chi2",
        "apsk_occupancy_16_minus_32",
        "snr_complexity_lambda_eff",
        "apsk_ring_lambda_eff",
        "penalized_evm_min",
        "penalized_evm_second",
        "penalized_evm_margin",
        "penalized_evm_16qam_minus_64qam",
        "penalized_evm_16qam_div_64qam",
        "penalized_evm_16apsk_minus_32apsk",
        "penalized_evm_16apsk_div_32apsk",
        "apsk16_complexity_score",
        "apsk32_complexity_score",
        "apsk_complexity_16_minus_32",
    ]
    penalized_feature_names = [f"penalized_evm_{m}" for m in modulations]
    if candidate_score_mode == "exact_mixture":
        # These two columns encoded validation-selected penalty weights in the
        # legacy 50-D schema.  They are identically zero for the parameter-free
        # exact score and are omitted from the deployed 48-D representation.
        keep = np.ones(len(extras), dtype=bool)
        keep[[24, 25]] = False
        extras = extras[keep]
        names = [
            name for name in names
            if name not in {"snr_complexity_lambda_eff", "apsk_ring_lambda_eff"}
        ]
    feature_values = np.concatenate(
        [
            np.nan_to_num(finite, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(penalized_finite, nan=0.0, posinf=0.0, neginf=0.0),
            extras,
        ]
    )
    return feature_values.astype(np.float32), [f"evm_{m}" for m in modulations] + penalized_feature_names + names[len(modulations):]


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}. Use --overwrite to replace it.")
    if args.reweight_existing:
        _reweight_existing(args, output)
        return
    if not args.raw_data or not args.mu_feature_data:
        raise SystemExit(
            "--raw-data and --mu-feature-data are required unless "
            "--reweight-existing is used."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    with ExitStack() as stack:
        raw = stack.enter_context(h5py.File(args.raw_data, "r"))
        mf = stack.enter_context(h5py.File(args.mu_feature_data, "r"))
        input_iq = (
            stack.enter_context(h5py.File(args.input_iq_feature_data, "r"))
            if args.input_iq_feature_data
            else None
        )
        required = ("iq", "pilot_indices", "pilot_symbols")
        missing = [key for key in required if key not in raw]
        if missing:
            raise SystemExit(f"Raw data is missing required pilot datasets: {missing}")
        if args.mu_key not in mf:
            raise SystemExit(f"mu key not found in {args.mu_feature_data}: {args.mu_key}")

        n = int(raw["iq"].shape[0])
        fs = float(args.sample_rate_hz) if args.sample_rate_hz is not None else float(raw["fs"][0])
        sps = max(int(args.samples_per_symbol), 1)
        taps = rrc_filter(beta=args.rrc_beta, span=args.rrc_span, sps=sps)
        timing_phases = tuple(range(sps)) if args.timing_phases is None else tuple(int(x) % sps for x in args.timing_phases)
        modulations = _resolve_modulations(raw, args.modulations)
        constellations = {m: get_constellation(m).astype(np.complex128) for m in modulations}
        phase_grid = (
            np.asarray([0.0], dtype=np.float64)
            if args.phase_grid_size <= 1
            else np.linspace(0.0, 2.0 * np.pi, int(args.phase_grid_size), endpoint=False, dtype=np.float64)
        )

        # Build one dummy feature vector to determine dimension/names.
        dummy_evm = np.zeros(len(modulations), dtype=np.float32)
        dummy_features, evm_feature_names = _evm_feature_vector(
            dummy_evm,
            modulations,
            0.0,
            np.asarray([0.0 + 0.0j]),
            constellations,
            [0.0 for _ in modulations],
            candidate_score_mode=args.candidate_score_mode,
        )
        evm_dim = int(dummy_features.size)

        with h5py.File(output, "w") as out:
            out.create_dataset(
                "constellation_img",
                shape=(n, 1, int(args.grid_size), int(args.grid_size)),
                dtype="float32",
                chunks=(1, 1, int(args.grid_size), int(args.grid_size)),
                compression="lzf",
            )
            out.create_dataset("evm_features", shape=(n, evm_dim), dtype="float32")
            out.create_dataset("all_candidate_evm", shape=(n, len(modulations)), dtype="float32")
            out.create_dataset(
                "all_candidate_penalized_evm",
                shape=(n, len(modulations)),
                dtype="float32",
            )
            out.create_dataset("symbol_valid", shape=(n,), dtype="bool")
            out.create_dataset("symbol_count", shape=(n,), dtype="int64")
            out.create_dataset("timing_phase", shape=(n,), dtype="int64")
            out.create_dataset("pilot_evm", shape=(n,), dtype="float32")
            out.create_dataset("pilot_evm_in_sample", shape=(n,), dtype="float32")
            out.create_dataset("complex_gain_real", shape=(n,), dtype="float32")
            out.create_dataset("complex_gain_imag", shape=(n,), dtype="float32")
            out.create_dataset("evm_best_index", shape=(n,), dtype="int64")
            out.create_dataset("penalized_evm_best_index", shape=(n,), dtype="int64")
            out.create_dataset("evm_margin", shape=(n,), dtype="float32")
            out.create_dataset("penalized_evm_margin", shape=(n,), dtype="float32")
            out.create_dataset("snr_complexity_lambda_eff", shape=(n,), dtype="float32")
            out.create_dataset("apsk_ring_lambda_eff", shape=(n,), dtype="float32")
            out.create_dataset("pilot_snr_est_db", shape=(n,), dtype="float32")
            out.create_dataset("penalty_snr_db", shape=(n,), dtype="float32")
            out.create_dataset("phase_best", shape=(n, len(modulations)), dtype="float32")
            out.create_dataset("mu_hat", shape=(n,), dtype="float32")
            out.create_dataset("external_mu_valid", shape=(n,), dtype="bool")

            out.create_dataset("candidate_modulations", data=np.asarray([m.encode("utf-8") for m in modulations], dtype="S32"))
            out.create_dataset("evm_feature_names", data=np.asarray([m.encode("utf-8") for m in evm_feature_names], dtype="S64"))
            _copy_meta(
                raw,
                out,
                (
                    "track_id",
                    "frame_id",
                    "frame_time_s",
                    "snr_db",
                    "label",
                    "label_names",
                    "modulation",
                    "fs",
                    "fc",
                    "T",
                    "pilot_indices",
                    "pilot_sample_indices",
                    "pilot_symbols",
                    "pilot_pattern",
                    "num_pilots",
                ),
            )

            iterator = range(n)
            if not args.no_progress:
                iterator = tqdm(iterator, desc="Symbol constellation features")

            valid_count = 0
            for i in iterator:
                mu_hat = float(mf[args.mu_key][i])
                fd0_hat = float(mf[args.fd0_key][i]) if args.fd0_key and args.fd0_key in mf else 0.0
                external_valid = _read_valid(mf, args.valid_key, i, mu_hat)
                out["mu_hat"][i] = mu_hat
                out["external_mu_valid"][i] = external_valid

                if not external_valid:
                    out["constellation_img"][i] = np.zeros((1, args.grid_size, args.grid_size), dtype=np.float32)
                    out["evm_features"][i] = np.zeros(evm_dim, dtype=np.float32)
                    out["all_candidate_evm"][i] = np.zeros(len(modulations), dtype=np.float32)
                    out["all_candidate_penalized_evm"][i] = np.zeros(len(modulations), dtype=np.float32)
                    out["symbol_valid"][i] = False
                    out["symbol_count"][i] = 0
                    out["timing_phase"][i] = -1
                    out["pilot_evm"][i] = np.inf
                    out["pilot_evm_in_sample"][i] = np.inf
                    out["evm_best_index"][i] = -1
                    out["penalized_evm_best_index"][i] = -1
                    out["evm_margin"][i] = 0.0
                    out["penalized_evm_margin"][i] = 0.0
                    out["snr_complexity_lambda_eff"][i] = 0.0
                    out["apsk_ring_lambda_eff"][i] = 0.0
                    out["pilot_snr_est_db"][i] = np.nan
                    out["penalty_snr_db"][i] = np.nan
                    continue

                if input_iq is not None:
                    r_comp = to_complex(input_iq["iq_comp"][i])
                else:
                    x = to_complex(raw["iq"][i])
                    r_comp = compensate_full_doppler(
                        x,
                        sample_rate_hz=fs,
                        fd0_hat_hz=fd0_hat,
                        mu_hat_hz_per_s=mu_hat,
                        centered=False,
                    )
                matched = np.convolve(r_comp, taps, mode="same")

                best_timing = -1
                best_gain = 1.0 + 0.0j
                best_pilot_evm = np.inf
                best_pilot_count = 0
                for timing in timing_phases:
                    gain, pilot_evm, pilot_count = _estimate_gain_and_pilot_evm(
                        matched,
                        raw["pilot_indices"][i],
                        raw["pilot_symbols"][i],
                        timing=int(timing),
                        sps=sps,
                    )
                    if pilot_count > 0 and pilot_evm < best_pilot_evm:
                        best_timing = int(timing)
                        best_gain = gain
                        best_pilot_evm = pilot_evm
                        best_pilot_count = pilot_count

                if best_timing < 0:
                    best_timing = 0

                crossfit_pilot_evm, _ = _estimate_crossfit_pilot_evm(
                    matched,
                    raw["pilot_indices"][i],
                    raw["pilot_symbols"][i],
                    timing=best_timing,
                    sps=sps,
                )
                # The cross-fitted estimate is the receiver-side SNR proxy.
                # Keep the in-sample value for calibration diagnostics only.
                pilot_evm_for_features = (
                    crossfit_pilot_evm
                    if np.isfinite(crossfit_pilot_evm)
                    else best_pilot_evm
                )

                symbols = _extract_equalized_symbols(
                    matched,
                    best_gain,
                    best_timing,
                    sps,
                    raw["pilot_indices"][i],
                    symbol_trim=args.symbol_trim,
                    max_symbols=args.max_symbols,
                )
                symbols = _prepare_symbols(
                    symbols,
                    center=not args.no_center,
                    power_normalize=not args.no_power_normalize,
                )
                valid = bool(symbols.size >= args.min_data_symbols)

                image = _constellation_image(symbols, args.grid_size, args.grid_limit, args.image_scale)
                evm, phase_best = _score_evm_all_candidates(symbols, modulations, constellations, phase_grid)
                true_snr_db = float(raw["snr_db"][i]) if "snr_db" in raw else np.nan
                pilot_snr_db = _pilot_sample_snr_db(
                    pilot_evm_for_features,
                    samples_per_symbol=sps,
                    offset_db=args.pilot_snr_offset_db,
                )
                penalty_snr_db = _select_penalty_snr(
                    args.snr_source,
                    true_snr_db=true_snr_db,
                    pilot_snr_db=pilot_snr_db,
                )
                evm_features, _ = _evm_feature_vector(
                    evm,
                    modulations,
                    pilot_evm_for_features,
                    symbols,
                    constellations,
                    phase_best,
                    snr_db=penalty_snr_db,
                    complexity_penalty_lambda=args.complexity_penalty_lambda,
                    complexity_penalty_snr_threshold=args.complexity_penalty_snr_threshold,
                    complexity_penalty_snr_scale=args.complexity_penalty_snr_scale,
                    apsk_ring_penalty_lambda=args.apsk_ring_penalty_lambda,
                    apsk_phase_weight=args.apsk_phase_weight,
                    apsk_occupancy_weight=args.apsk_occupancy_weight,
                    candidate_score_mode=args.candidate_score_mode,
                )
                score_names = [f"penalized_evm_{m}" for m in modulations]
                score_indices = [evm_feature_names.index(name) for name in score_names]
                penalized_evm = evm_features[score_indices].astype(np.float64)
                if args.candidate_score_mode == "exact_mixture":
                    lambda_eff = 0.0
                    apsk_ring_lambda_eff = 0.0
                else:
                    lambda_eff = float(evm_features[evm_feature_names.index("snr_complexity_lambda_eff")])
                    apsk_ring_lambda_eff = float(evm_features[evm_feature_names.index("apsk_ring_lambda_eff")])
                best_idx = int(np.argmin(np.where(np.isfinite(evm), evm, np.inf))) if evm.size else -1
                evm_sorted = np.sort(np.where(np.isfinite(evm), evm, np.inf))
                margin = float((evm_sorted[1] - evm_sorted[0]) / (abs(evm_sorted[1]) + 1e-12)) if evm_sorted.size > 1 else 0.0
                penalized_best_idx = (
                    int(np.argmin(np.where(np.isfinite(penalized_evm), penalized_evm, np.inf)))
                    if penalized_evm.size
                    else -1
                )
                penalized_sorted = np.sort(np.where(np.isfinite(penalized_evm), penalized_evm, np.inf))
                penalized_margin = (
                    float((penalized_sorted[1] - penalized_sorted[0]) / (abs(penalized_sorted[1]) + 1e-12))
                    if penalized_sorted.size > 1
                    else 0.0
                )

                out["constellation_img"][i] = image
                out["evm_features"][i] = evm_features
                out["all_candidate_evm"][i] = np.nan_to_num(evm, nan=0.0, posinf=0.0, neginf=0.0)
                out["all_candidate_penalized_evm"][i] = np.nan_to_num(
                    penalized_evm,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                out["symbol_valid"][i] = valid
                out["symbol_count"][i] = int(symbols.size)
                out["timing_phase"][i] = int(best_timing)
                out["pilot_evm"][i] = float(pilot_evm_for_features)
                out["pilot_evm_in_sample"][i] = float(best_pilot_evm)
                out["complex_gain_real"][i] = float(np.real(best_gain))
                out["complex_gain_imag"][i] = float(np.imag(best_gain))
                out["evm_best_index"][i] = best_idx
                out["penalized_evm_best_index"][i] = penalized_best_idx
                out["evm_margin"][i] = margin
                out["penalized_evm_margin"][i] = penalized_margin
                out["snr_complexity_lambda_eff"][i] = float(lambda_eff)
                out["apsk_ring_lambda_eff"][i] = float(apsk_ring_lambda_eff)
                out["pilot_snr_est_db"][i] = float(pilot_snr_db)
                out["penalty_snr_db"][i] = float(penalty_snr_db)
                out["phase_best"][i] = np.asarray(phase_best, dtype=np.float32)
                valid_count += int(valid)

            config = {
                "raw_data": args.raw_data,
                "input_iq_feature_data": args.input_iq_feature_data,
                "mu_feature_data": args.mu_feature_data,
                "mu_key": args.mu_key,
                "valid_key": args.valid_key,
                "fd0_key": args.fd0_key,
                "sample_rate_hz": fs,
                "samples_per_symbol": sps,
                "rrc_beta": args.rrc_beta,
                "rrc_span": args.rrc_span,
                "timing_phases": timing_phases,
                "modulations": modulations,
                "grid_size": args.grid_size,
                "grid_limit": args.grid_limit,
                "image_scale": args.image_scale,
                "phase_grid_size": args.phase_grid_size,
                "symbol_trim": args.symbol_trim,
                "max_symbols": args.max_symbols,
                "center": not args.no_center,
                "power_normalize": not args.no_power_normalize,
                "min_data_symbols": args.min_data_symbols,
                "complexity_penalty_lambda": args.complexity_penalty_lambda,
                "complexity_penalty_snr_threshold": args.complexity_penalty_snr_threshold,
                "complexity_penalty_snr_scale": args.complexity_penalty_snr_scale,
                "snr_source": args.snr_source,
                "pilot_snr_offset_db": (
                    float(args.pilot_snr_offset_db)
                    if args.pilot_snr_offset_db is not None
                    else float(-10.0 * np.log10(max(sps, 1)))
                ),
                "pilot_evm_estimator": "two_fold_crossfit_gain_residual",
                "apsk_ring_penalty_lambda": args.apsk_ring_penalty_lambda,
                "apsk_phase_weight": args.apsk_phase_weight,
                "apsk_occupancy_weight": args.apsk_occupancy_weight,
                "candidate_score_mode": args.candidate_score_mode,
            }
            out.attrs["config_json"] = json.dumps(config)

    summary = {
        "raw_data": args.raw_data,
        "mu_feature_data": args.mu_feature_data,
        "output": str(output),
        "num_samples": int(n),
        "valid_fraction": float(valid_count / max(n, 1)),
        "evm_dim": int(evm_dim),
        "grid_size": int(args.grid_size),
        "candidate_modulations": list(modulations),
        "snr_source": args.snr_source,
    }
    if output.exists():
        with h5py.File(output, "r") as f:
            if "pilot_snr_est_db" in f and "snr_db" in f:
                estimate = f["pilot_snr_est_db"][:].astype(np.float64)
                truth = f["snr_db"][:].astype(np.float64)
                valid_snr = np.isfinite(estimate) & np.isfinite(truth)
                if np.any(valid_snr):
                    error = estimate[valid_snr] - truth[valid_snr]
                    summary["pilot_snr_mae_db"] = float(np.mean(np.abs(error)))
                    summary["pilot_snr_rmse_db"] = float(np.sqrt(np.mean(error**2)))
                    summary["pilot_snr_bias_db"] = float(np.mean(error))
    save_json(summary, output.with_suffix(".summary.json"))
    print(summary)


if __name__ == "__main__":
    main()
