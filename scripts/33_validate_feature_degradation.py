#!/usr/bin/env python
"""Validate quadratic-phase feature degradation and blind-mu error tails."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from math import factorial
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
import torch
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from src.datasets.feature_dataset import _preprocess_iq
from src.drc_hoc.cumulants import _partitions_tuple
from src.models import DRCDualNet
from src.physics.gamma_metric import discrete_attenuation_factor
from src.plotting.common import (
    IEEE_TRANS_PALETTE,
    boxed_legend,
    format_ieee_axis,
    save_axes_panels,
    set_panel_aspect_10_9,
    set_ieee_trans_style,
)
from src.signal.modulation import get_constellation
from src.signal.pulse_shape import rrc_filter


METHODS = ("no_comp", "blind", "pilot", "oracle")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--feature-root",
        default="data/features/mu_comp_ablation",
    )
    p.add_argument(
        "--result-root",
        default="outputs/results/mu_comp_ablation",
    )
    p.add_argument("--split", default="val")
    p.add_argument(
        "--output-root",
        default="outputs/results/feature_degradation_validation",
    )
    p.add_argument(
        "--figure",
        default="outputs/figures/paper/gamma_feature_degradation_validation.pdf",
    )
    p.add_argument("--epsilon", type=float, default=0.10)
    p.add_argument(
        "--raw-data",
        default="data/processed/leo_7mods_snr_balanced_trainval_pilot.h5",
        help="Pilot-bearing source frames corresponding to the controlled oracle features.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--classification-batch-size", type=int, default=64)
    p.add_argument(
        "--iq-only-checkpoint",
        default="outputs/checkpoints/exact_mixture/representation_ablation/protocol_a/iq_only/seed_41/best.pt",
    )
    p.add_argument(
        "--evm-only-checkpoint",
        default="outputs/checkpoints/exact_mixture/representation_ablation/protocol_a/constellation_only/seed_41/best.pt",
    )
    p.add_argument(
        "--iq-evm-checkpoint",
        default="outputs/checkpoints/exact_mixture/protocol_a/proposed/seed_41/best.pt",
    )
    p.add_argument(
        "--final-iq-evm-checkpoint",
        default="outputs/checkpoints/hybrid_dfrft/protocol_b/proposed/seed_41/best.pt",
        help="Deployed compact I/Q--constellation checkpoint used by --classification-only.",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the paper figure from the existing controlled-scan CSV.",
    )
    p.add_argument("--high-snr-min-db", type=float, default=5.0)
    p.add_argument("--iq-chunk-size", type=int, default=64)
    p.add_argument("--controlled-samples-per-class-snr", type=int, default=4)
    p.add_argument("--controlled-seed", type=int, default=2026)
    p.add_argument(
        "--controlled-frame-lengths",
        type=int,
        nargs="+",
        default=(2048, 4096, 8192),
        help=(
            "Prefix lengths used to verify that residual severity, rather than "
            "frame length alone, predicts feature stability."
        ),
    )
    p.add_argument(
        "--controlled-gamma-grid",
        type=float,
        nargs="+",
        default=(
            0.0,
            0.025,
            0.05,
            0.075,
            0.10,
            0.125,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.36,
            0.37,
            0.38,
            0.39,
            0.40,
        ),
    )
    p.add_argument(
        "--controlled-classification-gamma-grid",
        type=float,
        nargs="+",
        default=(
            0.0,
            0.025,
            0.05,
            0.075,
            0.10,
            0.125,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.60,
            0.80,
            1.00,
            1.50,
            2.00,
            3.00,
        ),
        help=(
            "Residual-severity grid for fixed-model accuracy. It extends beyond "
            "the HOC diagnostic grid so the onset of end-to-end AMC degradation is visible."
        ),
    )
    p.add_argument(
        "--controlled-only",
        action="store_true",
        help=(
            "Run the controlled multi-duration representation scan and the fixed-model "
            "I/Q, constellation descriptors, and fused accuracy sweep."
        ),
    )
    p.add_argument(
        "--skip-classification-sweep",
        action="store_true",
        help="Skip the fixed-model residual-severity accuracy sweep.",
    )
    p.add_argument(
        "--classification-only",
        action="store_true",
        help=(
            "Run only a fixed-model, SNR-conditioned residual-severity scan. "
            "The existing representation metrics are retained for Fig. 3(a)--(c)."
        ),
    )
    p.add_argument(
        "--classification-raw-data",
        default=None,
        help="Raw HDF5 for --classification-only. Defaults to --raw-data.",
    )
    p.add_argument(
        "--classification-feature-data",
        default=None,
        help="Oracle-compensated feature HDF5 for --classification-only.",
    )
    p.add_argument(
        "--classification-oracle-correct-estimated-iq",
        action="store_true",
        help=(
            "Treat classification-feature-data as estimated-compensation I/Q and "
            "remove its stored residual mu/CFO error in memory before injecting "
            "the controlled residual severity."
        ),
    )
    p.add_argument(
        "--classification-splits",
        default=None,
        help="Split NPZ for --classification-only.",
    )
    p.add_argument(
        "--classification-split",
        default="test",
        choices=("train", "val", "test"),
    )
    p.add_argument(
        "--classification-snr-db",
        type=float,
        nargs="+",
        default=None,
        help="Exact SNR strata for the conditional scan, for example -5 0 5 10.",
    )
    p.add_argument(
        "--classification-samples-per-class-snr",
        type=int,
        default=20,
        help="Controlled base frames per payload class and exact SNR stratum.",
    )
    p.add_argument(
        "--classification-output",
        default=None,
        help="CSV path for the conditional scan. Defaults under --output-root.",
    )
    p.add_argument(
        "--classification-detail-output",
        default=None,
        help="Optional per-frame paired-scan CSV for bootstrap calibration.",
    )
    p.add_argument(
        "--reuse-classification-scan",
        action="store_true",
        help="Reuse an existing --classification-output instead of recomputing it.",
    )
    return p.parse_args()


def set_trans_style() -> None:
    set_ieee_trans_style()


def _decode(values: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def _load_symbol_helpers():
    """Load the canonical label-free EVM implementation without duplicating it."""

    path = PROJECT_ROOT / "scripts" / "10_precompute_symbol_constellation.py"
    spec = importlib.util.spec_from_file_location("symbol_feature_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load symbol-feature helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _select_controlled_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    snr_db: np.ndarray,
    samples_per_class_snr: int,
    seed: int,
    high_snr_min_db: float | None = None,
    snr_values: np.ndarray | None = None,
) -> np.ndarray:
    """Select a reproducible class--SNR-balanced controlled subset."""

    rng = np.random.default_rng(seed)
    selected_positions: list[int] = []
    if snr_values is None:
        if high_snr_min_db is None:
            raise ValueError("Either high_snr_min_db or snr_values is required.")
        selected_snr_values = np.unique(snr_db[snr_db >= high_snr_min_db])
    else:
        selected_snr_values = np.unique(np.asarray(snr_values, dtype=np.float64))
    for label in np.unique(labels):
        for snr in selected_snr_values:
            candidates = np.flatnonzero((labels == label) & np.isclose(snr_db, snr))
            if candidates.size == 0:
                raise ValueError(
                    f"No controlled frames for label={int(label)} at SNR={float(snr):g} dB."
                )
            count = min(samples_per_class_snr, candidates.size)
            selected_positions.extend(
                rng.choice(candidates, size=count, replace=False).tolist()
            )
    if not selected_positions:
        raise ValueError("Controlled residual scan selected no validation frames.")
    return np.sort(indices[np.asarray(selected_positions, dtype=np.int64)])


def _phase_sensitive_mask(feature_names: list[str]) -> np.ndarray:
    mask: list[bool] = []
    pattern = re.compile(r"C(\d)(\d)_")
    for name in feature_names:
        match = pattern.match(name)
        if not match:
            mask.append(False)
            continue
        p, q = int(match.group(1)), int(match.group(2))
        mask.append((p - 2 * q) != 0)
    return np.asarray(mask, dtype=bool)


def _hoc_orders(feature_names: list[str]) -> tuple[tuple[int, int], ...]:
    orders: list[tuple[int, int]] = []
    for name in feature_names:
        match = re.match(r"C(\d)(\d)_", name)
        if not match:
            continue
        order = (int(match.group(1)), int(match.group(2)))
        if order not in orders:
            orders.append(order)
    return tuple(orders)


def _extract_hoc_features_batch(
    x: np.ndarray,
    orders: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Vectorized equivalent of the configured real/imag/abs HOC extractor."""

    z = np.asarray(x, dtype=np.complex128)
    z = z - np.mean(z, axis=1, keepdims=True)
    power = np.mean(np.abs(z) ** 2, axis=1, keepdims=True)
    z = z / np.sqrt(power + 1e-12)
    z_conj = np.conj(z)
    moments: dict[tuple[int, int], np.ndarray] = {}

    def moment(p: int, q: int) -> np.ndarray:
        key = (p, q)
        if key not in moments:
            moments[key] = np.mean((z ** (p - q)) * (z_conj**q), axis=1)
        return moments[key]

    columns: list[np.ndarray] = []
    for p, q in orders:
        types = (0,) * (p - q) + (1,) * q
        cumulant = np.zeros(z.shape[0], dtype=np.complex128)
        for partition in _partitions_tuple(p):
            coeff = ((-1) ** (len(partition) - 1)) * factorial(len(partition) - 1)
            product = np.ones(z.shape[0], dtype=np.complex128)
            for block in partition:
                block_q = sum(types[i] for i in block)
                product *= moment(len(block), block_q)
            cumulant += coeff * product
        columns.extend((np.real(cumulant), np.imag(cumulant), np.abs(cumulant)))
    return np.stack(columns, axis=1).astype(np.float64)


def _relative_vector_error(
    estimate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    delta = estimate[:, mask] - reference[:, mask]
    numerator = np.linalg.norm(delta, axis=1)
    denominator = np.linalg.norm(reference[:, mask], axis=1)
    scale = np.maximum(denominator, np.quantile(denominator, 0.05) * 0.1 + 1e-8)
    return numerator / scale


def _iq_coherence(
    method_path: Path,
    oracle_path: Path,
    indices: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    values = np.empty(indices.size, dtype=np.float64)
    with h5py.File(method_path, "r") as method, h5py.File(oracle_path, "r") as oracle:
        for start in range(0, indices.size, chunk_size):
            stop = min(start + chunk_size, indices.size)
            idx = indices[start:stop]
            # h5py requires monotonically increasing fancy indices.
            order = np.argsort(idx)
            sorted_idx = idx[order]
            m = np.asarray(method["iq_comp"][sorted_idx], dtype=np.float64)
            o = np.asarray(oracle["iq_comp"][sorted_idx], dtype=np.float64)
            m_complex = m[:, 0] + 1j * m[:, 1]
            o_complex = o[:, 0] + 1j * o[:, 1]
            numerator = np.abs(np.sum(np.conj(o_complex) * m_complex, axis=1))
            denominator = np.sqrt(
                np.sum(np.abs(o_complex) ** 2, axis=1)
                * np.sum(np.abs(m_complex) ** 2, axis=1)
            )
            chunk_values = numerator / np.maximum(denominator, 1e-12)
            values[start:stop][order] = chunk_values
    return values


def _wilson_interval(correct: int, count: int, z: float = 1.959964) -> tuple[float, float]:
    if count <= 0:
        return float("nan"), float("nan")
    p = correct / count
    denom = 1.0 + z**2 / count
    center = (p + z**2 / (2.0 * count)) / denom
    radius = z * np.sqrt(p * (1.0 - p) / count + z**2 / (4.0 * count**2)) / denom
    return float(center - radius), float(center + radius)


def _aggregate_continuous(
    gamma: np.ndarray,
    values: np.ndarray,
    bins: np.ndarray,
    method: str,
    metric: str,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (gamma >= lo) & (gamma < hi) & np.isfinite(values)
        if not np.any(mask):
            continue
        group = values[mask]
        rows.append(
            {
                "method": method,
                "metric": metric,
                "gamma_low": float(lo),
                "gamma_high": float(hi),
                "gamma_center": float(0.5 * (lo + hi)),
                "count": int(group.size),
                "mean": float(np.mean(group)),
                "median": float(np.median(group)),
                "q10": float(np.quantile(group, 0.10)),
                "q90": float(np.quantile(group, 0.90)),
            }
        )
    return rows


def _controlled_residual_scan(
    feature_path: Path,
    indices: np.ndarray,
    labels: np.ndarray,
    snr_db: np.ndarray,
    feature_names: list[str],
    gamma_grid: np.ndarray,
    high_snr_min_db: float,
    samples_per_class_snr: int,
    seed: int,
    chunk_size: int,
    frame_lengths: tuple[int, ...],
    epsilon: float,
) -> pd.DataFrame:
    selected_indices = _select_controlled_indices(
        indices,
        labels,
        snr_db,
        samples_per_class_snr,
        seed,
        high_snr_min_db=high_snr_min_db,
    )
    with h5py.File(feature_path, "r") as feature:
        iq = np.asarray(feature["iq_comp"][selected_indices], dtype=np.float64)
        oracle_hoc = np.asarray(feature["h_drc"][selected_indices], dtype=np.float64)
        fs_values = np.asarray(feature["fs"][selected_indices], dtype=np.float64)
        duration_values = np.asarray(feature["T"][selected_indices], dtype=np.float64)

    fs = float(np.median(fs_values))
    duration = float(np.median(duration_values))
    if not np.allclose(fs_values, fs) or not np.allclose(duration_values, duration):
        raise ValueError("Controlled scan requires a common sample rate and duration.")

    base = iq[:, 0] + 1j * iq[:, 1]
    sensitive = _phase_sensitive_mask(feature_names)
    orders = _hoc_orders(feature_names)
    rows: list[dict[str, float | int | str]] = []

    valid_lengths = tuple(sorted({int(n) for n in frame_lengths if 16 <= int(n) <= base.shape[1]}))
    if not valid_lengths:
        raise ValueError("No controlled frame length is within [16, N].")
    if base.shape[1] not in valid_lengths:
        valid_lengths = (*valid_lengths, int(base.shape[1]))

    for frame_length in valid_lengths:
        # Each duration gets its own reference HOC. This makes the comparison
        # conditional on the same waveform prefix and avoids comparing a
        # truncated frame with an 8192-sample cumulant estimate.
        reference = base[:, :frame_length]
        reference_hoc_parts: list[np.ndarray] = []
        for start in range(0, reference.shape[0], chunk_size):
            reference_hoc_parts.append(
                _extract_hoc_features_batch(reference[start : start + chunk_size], orders)
            )
        reference_hoc = np.vstack(reference_hoc_parts)
        frame_duration = float(frame_length / fs)
        time_s = np.arange(frame_length, dtype=np.float64) / fs

        for gamma in gamma_grid:
            signs = (1.0,) if np.isclose(gamma, 0.0) else (-1.0, 1.0)
            hoc_parts: list[np.ndarray] = []
            coherence_parts: list[np.ndarray] = []
            residual_mu = float(gamma) / (np.pi * frame_duration**2)
            for sign in signs:
                phase = np.exp(1j * np.pi * sign * residual_mu * time_s**2)
                for start in range(0, reference.shape[0], chunk_size):
                    stop = min(start + chunk_size, reference.shape[0])
                    reference_iq = reference[start:stop]
                    distorted_iq = reference_iq * phase[None, :]
                    hoc_parts.append(_extract_hoc_features_batch(distorted_iq, orders))
                    numerator = np.abs(
                        np.sum(np.conj(reference_iq) * distorted_iq, axis=1)
                    )
                    denominator = np.sqrt(
                        np.sum(np.abs(reference_iq) ** 2, axis=1)
                        * np.sum(np.abs(distorted_iq) ** 2, axis=1)
                    )
                    coherence_parts.append(numerator / np.maximum(denominator, 1e-12))

            hoc_error = _relative_vector_error(
                np.vstack(hoc_parts),
                np.vstack([reference_hoc for _ in signs]),
                sensitive,
            )
            coherence = np.concatenate(coherence_parts)
            iq_loss = 1.0 - coherence
            stable = (hoc_error <= float(epsilon)) & (iq_loss <= float(epsilon))
            common = {
                "gamma_res": float(gamma),
                "mu_res_hz_per_s": residual_mu,
                "frame_length": int(frame_length),
                "frame_duration_ms": 1e3 * frame_duration,
                "count": int(hoc_error.size),
            }
            for metric, values in (
                ("hoc_sensitive_relative_error", hoc_error),
                ("iq_coherence", coherence),
                ("iq_coherence_loss", iq_loss),
                ("feature_stable", stable.astype(np.float64)),
            ):
                rows.append(
                    {
                        "metric": metric,
                        **common,
                        "median": float(np.median(values)),
                        "q10": float(np.quantile(values, 0.10)),
                        "q90": float(np.quantile(values, 0.90)),
                        "mean": float(np.mean(values)),
                    }
                )
    return pd.DataFrame(rows)


def _load_fixed_classifier(
    checkpoint_path: Path,
    *,
    num_classes: int,
    hoc_dim: int,
    evm_dim: int,
    device: torch.device,
) -> tuple[DRCDualNet, dict[str, object], dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_config = dict(checkpoint.get("config", {}))
    model_config = dict(run_config.get("model_config", {}))
    model_config.update(
        {
            "num_classes": int(num_classes),
            "hoc_dim": int(hoc_dim),
            "evm_dim": int(evm_dim),
        }
    )
    model = DRCDualNet(**model_config)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.to(device)
    model.eval()
    return model, run_config, model_config


def _standardize_evm(features: np.ndarray, stats: dict[str, object] | None) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if stats is None:
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return np.nan_to_num((values - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


def _resolve_symbol_config(run_config: dict[str, object]) -> dict[str, object]:
    """Read the exact training-time EVM construction options from its HDF5 file."""

    symbol_path = Path(str(run_config["symbol_feature_data"]))
    with h5py.File(symbol_path, "r") as symbol_file:
        config_json = symbol_file.attrs.get("config_json", "{}")
    config = json.loads(config_json)
    # Older feature files predate the explicit mode and are logistic legacy
    # artifacts; new deployed files record exact_mixture.
    config.setdefault("candidate_score_mode", "logistic")
    required = (
        "samples_per_symbol",
        "rrc_beta",
        "rrc_span",
        "symbol_trim",
        "max_symbols",
        "complexity_penalty_lambda",
        "complexity_penalty_snr_threshold",
        "complexity_penalty_snr_scale",
        "apsk_ring_penalty_lambda",
        "apsk_phase_weight",
        "apsk_occupancy_weight",
        "pilot_snr_offset_db",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Symbol-feature configuration is missing: {missing}")
    return config


def _build_evm_features_for_frame(
    frame: np.ndarray,
    *,
    pilot_indices: np.ndarray,
    pilot_symbols: np.ndarray,
    symbol_helpers,
    taps: np.ndarray,
    sps: int,
    modulations: tuple[str, ...],
    constellations: dict[str, np.ndarray],
    config: dict[str, object],
) -> tuple[np.ndarray, bool]:
    """Reconstruct the deployed label-free EVM vector for one residual frame."""

    matched = np.convolve(frame, taps, mode="same")
    gain, pilot_evm_in_sample, _ = symbol_helpers._estimate_gain_and_pilot_evm(
        matched,
        pilot_indices,
        pilot_symbols,
        timing=0,
        sps=sps,
    )
    pilot_evm, _ = symbol_helpers._estimate_crossfit_pilot_evm(
        matched,
        pilot_indices,
        pilot_symbols,
        timing=0,
        sps=sps,
    )
    if not np.isfinite(pilot_evm):
        pilot_evm = pilot_evm_in_sample
    symbols = symbol_helpers._extract_equalized_symbols(
        matched,
        gain,
        0,
        sps,
        pilot_indices,
        symbol_trim=int(config["symbol_trim"]),
        max_symbols=int(config["max_symbols"]),
    )
    symbols = symbol_helpers._prepare_symbols(symbols, center=True, power_normalize=True)
    phase_grid = np.asarray([0.0], dtype=np.float64)
    evm, phase_best = symbol_helpers._score_evm_all_candidates(
        symbols,
        modulations,
        constellations,
        phase_grid,
    )
    pilot_snr_db = symbol_helpers._pilot_sample_snr_db(
        pilot_evm,
        samples_per_symbol=sps,
        offset_db=float(config["pilot_snr_offset_db"]),
    )
    features, _ = symbol_helpers._evm_feature_vector(
        evm,
        modulations,
        pilot_evm,
        symbols,
        constellations,
        phase_best,
        snr_db=pilot_snr_db,
        complexity_penalty_lambda=float(config["complexity_penalty_lambda"]),
        complexity_penalty_snr_threshold=float(config["complexity_penalty_snr_threshold"]),
        complexity_penalty_snr_scale=float(config["complexity_penalty_snr_scale"]),
        apsk_ring_penalty_lambda=float(config["apsk_ring_penalty_lambda"]),
        apsk_phase_weight=float(config["apsk_phase_weight"]),
        apsk_occupancy_weight=float(config["apsk_occupancy_weight"]),
        candidate_score_mode=str(config["candidate_score_mode"]),
    )
    valid = bool(symbols.size >= 32)
    return np.asarray(features, dtype=np.float32), valid


@torch.inference_mode()
def _predict_fixed_classifier(
    model: DRCDualNet,
    model_config: dict[str, object],
    iq: np.ndarray,
    evm_features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    use_iq = bool(model_config.get("use_iq_stream", True))
    use_evm = bool(model_config.get("use_evm_features", True))
    for start in range(0, iq.shape[0], batch_size):
        stop = min(start + batch_size, iq.shape[0])
        iq_batch = torch.from_numpy(iq[start:stop]).to(device) if use_iq else None
        evm_batch = (
            torch.from_numpy(evm_features[start:stop]).to(device) if use_evm else None
        )
        logits = model(iq_batch, None, None, evm_features=evm_batch)
        predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(predictions).astype(np.int64)


def _controlled_classification_scan(
    *,
    raw_path: Path,
    feature_path: Path,
    indices: np.ndarray,
    labels: np.ndarray,
    snr_db: np.ndarray,
    gamma_grid: np.ndarray,
    samples_per_class_snr: int,
    seed: int,
    checkpoints: dict[str, Path],
    device_name: str,
    batch_size: int,
    high_snr_min_db: float | None = None,
    snr_values: np.ndarray | None = None,
    oracle_correct_estimated_iq: bool = False,
    residual_cfo_hz: float = 0.0,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Measure fixed AMC classifiers after controlled residual-phase injection."""

    selected_indices = _select_controlled_indices(
        indices,
        labels,
        snr_db,
        samples_per_class_snr,
        seed,
        high_snr_min_db=high_snr_min_db,
        snr_values=snr_values,
    )
    requested_device = torch.device(device_name)
    device = requested_device if requested_device.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")

    with h5py.File(feature_path, "r") as feature, h5py.File(raw_path, "r") as raw:
        iq_raw = np.asarray(feature["iq_comp"][selected_indices], dtype=np.float64)
        base = iq_raw[:, 0] + 1j * iq_raw[:, 1]
        frame_labels = np.asarray(raw["label"][selected_indices], dtype=np.int64)
        frame_snr = np.asarray(raw["snr_db"][selected_indices], dtype=np.float64)
        pilot_indices = np.asarray(raw["pilot_indices"][selected_indices], dtype=np.int64)
        pilot_symbols = np.asarray(raw["pilot_symbols"][selected_indices])
        fs_values = np.asarray(raw["fs"][selected_indices], dtype=np.float64)
        label_names = tuple(name.upper() for name in _decode(raw["label_names"][:]))

        if oracle_correct_estimated_iq:
            required_feature_keys = {"mu_hat", "external_fd0_hat"}
            missing = sorted(required_feature_keys.difference(feature.keys()))
            if missing:
                raise KeyError(
                    "In-memory oracle correction requires feature fields: "
                    + ", ".join(missing)
                )
            mu_hat = np.asarray(feature["mu_hat"][selected_indices], dtype=np.float64)
            fd0_hat = np.asarray(feature["external_fd0_hat"][selected_indices], dtype=np.float64)
            mu_true = np.asarray(raw["mu"][selected_indices], dtype=np.float64)
            fd0_true = np.asarray(raw["fd0"][selected_indices], dtype=np.float64)
            sample_index = np.arange(base.shape[1], dtype=np.float64)[None, :]
            time_matrix = sample_index / fs_values[:, None]
            residual_phase = (
                2.0 * np.pi * (fd0_true - fd0_hat)[:, None] * time_matrix
                + np.pi * (mu_true - mu_hat)[:, None] * time_matrix**2
            )
            base = base * np.exp(-1j * residual_phase)

    fs = float(np.median(fs_values))
    if not np.allclose(fs_values, fs):
        raise ValueError("Controlled classification scan requires a common sample rate.")

    loaded: dict[str, tuple[DRCDualNet, dict[str, object], dict[str, object]]] = {}
    for name, checkpoint in checkpoints.items():
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing fixed classifier checkpoint: {checkpoint}")
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_config = dict(checkpoint_payload.get("config", {}))
        loaded[name] = _load_fixed_classifier(
            checkpoint,
            num_classes=len(label_names),
            hoc_dim=21,
            evm_dim=int(checkpoint_config.get("evm_dim", 48)),
            device=device,
        )

    canonical_run_config = loaded["iq_evm"][1]
    symbol_config = _resolve_symbol_config(canonical_run_config)
    sps = int(symbol_config["samples_per_symbol"])
    taps = rrc_filter(
        beta=float(symbol_config["rrc_beta"]),
        span=int(symbol_config["rrc_span"]),
        sps=sps,
    )
    helpers = _load_symbol_helpers()
    modulations = tuple(name.upper() for name in symbol_config["modulations"])
    if modulations != label_names:
        raise ValueError(
            "Controlled scan expects candidate modulation order to match raw label names: "
            f"{modulations} != {label_names}"
        )
    constellations = {name: get_constellation(name).astype(np.complex128) for name in modulations}
    duration = float(base.shape[1] / fs)
    time_s = np.arange(base.shape[1], dtype=np.float64) / fs
    rows: list[dict[str, float | int | str]] = []
    detail_parts: list[pd.DataFrame] = []

    for gamma in gamma_grid:
        signs = (1.0,) if np.isclose(gamma, 0.0) else (-1.0, 1.0)
        residual_mu = float(gamma) / (np.pi * duration**2)
        distorted_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        evm_parts: list[np.ndarray] = []
        valid_parts: list[bool] = []
        for sign in signs:
            phase = np.exp(
                1j * (
                    2.0 * np.pi * float(residual_cfo_hz) * time_s
                    + np.pi * sign * residual_mu * time_s**2
                )
            )
            distorted = base * phase[None, :]
            distorted_parts.append(distorted)
            label_parts.append(frame_labels)
            for frame, frame_pilot_indices, frame_pilot_symbols in zip(
                distorted, pilot_indices, pilot_symbols
            ):
                feature_vector, valid = _build_evm_features_for_frame(
                    frame,
                    pilot_indices=frame_pilot_indices,
                    pilot_symbols=frame_pilot_symbols,
                    symbol_helpers=helpers,
                    taps=taps,
                    sps=sps,
                    modulations=modulations,
                    constellations=constellations,
                    config=symbol_config,
                )
                evm_parts.append(feature_vector)
                valid_parts.append(valid)

        distorted_all = np.vstack(distorted_parts)
        labels_all = np.concatenate(label_parts)
        snr_all = np.concatenate([frame_snr] * len(signs))
        frame_index_all = np.concatenate([selected_indices] * len(signs))
        sign_all = np.concatenate(
            [np.full(len(selected_indices), sign, dtype=np.float64) for sign in signs]
        )
        iq_input = np.stack(
            [
                _preprocess_iq(
                    np.stack((np.real(frame), np.imag(frame)), axis=0),
                    representation="iq",
                    normalize="zscore",
                )
                for frame in distorted_all
            ],
            axis=0,
        )
        evm_raw = np.vstack(evm_parts).astype(np.float32)

        for name, (model, run_config, model_config) in loaded.items():
            feature_indices = np.asarray(
                run_config.get("evm_feature_indices", np.arange(evm_raw.shape[1])),
                dtype=np.int64,
            )
            evm_input = _standardize_evm(
                evm_raw[:, feature_indices],
                run_config.get("evm_stats") if bool(run_config.get("standardize_evm", False)) else None,
            )
            predicted = _predict_fixed_classifier(
                model,
                model_config,
                iq_input,
                evm_input,
                device=device,
                batch_size=batch_size,
            )
            correct = predicted == labels_all
            detail_parts.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "frame_index": frame_index_all,
                        "label": labels_all,
                        "snr_db": snr_all,
                        "gamma_res": float(gamma),
                        "residual_cfo_hz": float(residual_cfo_hz),
                        "injection_sign": sign_all,
                        "predicted": predicted,
                        "correct": correct.astype(np.int8),
                    }
                )
            )
            if snr_values is None:
                groups = [(None, np.ones(correct.size, dtype=bool))]
            else:
                groups = [
                    (float(snr), np.isclose(snr_all, snr))
                    for snr in np.asarray(snr_values, dtype=np.float64)
                ]
            for snr_value, mask in groups:
                count = int(np.sum(mask))
                if count == 0:
                    continue
                low, high = _wilson_interval(int(np.sum(correct[mask])), count)
                row: dict[str, float | int | str] = {
                    "model": name,
                    "gamma_res": float(gamma),
                    "mu_res_hz_per_s": residual_mu,
                    "residual_cfo_hz": float(residual_cfo_hz),
                    "count": count,
                    "accuracy": float(np.mean(correct[mask])),
                    "wilson95_low": low,
                    "wilson95_high": high,
                    "symbol_valid_fraction": float(np.mean(valid_parts)),
                }
                if snr_value is not None:
                    row["snr_db"] = snr_value
                rows.append(row)

    for model, _, _ in loaded.values():
        model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    detail = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
    return pd.DataFrame(rows), selected_indices, detail


def _aggregate_accuracy(
    gamma: np.ndarray,
    correct: np.ndarray,
    bins: np.ndarray,
    method: str,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (gamma >= lo) & (gamma < hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        hits = int(np.sum(correct[mask]))
        low, high = _wilson_interval(hits, count)
        rows.append(
            {
                "method": method,
                "gamma_low": float(lo),
                "gamma_high": float(hi),
                "gamma_center": float(0.5 * (lo + hi)),
                "count": count,
                "accuracy": hits / count,
                "wilson95_low": low,
                "wilson95_high": high,
            }
        )
    return rows


def _plot(
    controlled: pd.DataFrame,
    classification: pd.DataFrame | None,
    output: Path,
    epsilon: float,
    analytical_boundary: float,
    empirical_boundary: float,
) -> None:
    set_trans_style()
    fig, grid = plt.subplots(2, 2, figsize=(7.05, 6.20))
    axes = [grid[0, 0], grid[0, 1], grid[1, 0]]
    accuracy_ax = grid[1, 1]
    colors = (
        IEEE_TRANS_PALETTE["blue"],
        IEEE_TRANS_PALETTE["green"],
        IEEE_TRANS_PALETTE["purple"],
    )
    styles = ("-", "-", "-")
    markers = ("o", "s", "^")
    x_limit = 0.40
    specs = (
        ("hoc_sensitive_relative_error", "median", "Representative cumulant error", (0.0, 0.60)),
        ("iq_coherence_loss", "median", r"I/Q coherence loss ($\times 10^{-3}$)", (0.0, 8.5)),
        ("feature_stable", "mean", "Joint feature stability probability", (-0.02, 1.02)),
    )
    plotted: list[list[tuple[np.ndarray, np.ndarray]]] = [[], [], []]

    for axis_index, (metric, statistic, ylabel, ylim) in enumerate(specs):
        ax = axes[axis_index]
        subset = controlled[controlled["metric"] == metric]
        for idx, (frame_length, group) in enumerate(subset.groupby("frame_length", sort=True)):
            group = group.sort_values("gamma_res")
            keep = group["gamma_res"].to_numpy(dtype=float) <= x_limit
            x = group.loc[keep, "gamma_res"].to_numpy(dtype=float)
            value = group.loc[keep, statistic].to_numpy(dtype=float)
            if metric == "iq_coherence_loss":
                value = 1.0e3 * value
            plotted[axis_index].append((x, value))
            ax.plot(
                x,
                value,
                color=colors[idx % len(colors)],
                linestyle=styles[idx % len(styles)],
                marker=markers[idx % len(markers)],
                markerfacecolor="white",
                markeredgecolor=colors[idx % len(colors)],
                markeredgewidth=0.75,
                markersize=3.2,
                linewidth=1.25,
                label=fr"$N={int(frame_length)}$",
            )
        ax.set_xlabel(r"Residual severity $\gamma_{\rm res}$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, x_limit)
        ax.set_ylim(*ylim)
        format_ieee_axis(ax)
        set_panel_aspect_10_9(ax)

    boxed_legend(axes[0], loc="upper left", fontsize=5.9)
    boxed_legend(axes[1], loc="lower right", fontsize=5.9)
    boxed_legend(axes[2], loc="upper right", fontsize=5.9)

    def _draw_inset(
        ax: plt.Axes,
        series: list[tuple[np.ndarray, np.ndarray]],
        *,
        bounds: tuple[float, float, float, float],
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        xticks: tuple[float, ...],
        yticks: tuple[float, ...],
        ylabel: str | None = None,
        show_indicator: bool = True,
    ) -> plt.Axes:
        inset = ax.inset_axes(bounds)
        for idx, (x, value) in enumerate(series):
            keep = (x >= xlim[0]) & (x <= xlim[1])
            inset.plot(
                x[keep],
                value[keep],
                color=colors[idx % len(colors)],
                linestyle=styles[idx % len(styles)],
                marker=markers[idx % len(markers)],
                markerfacecolor="white",
                markeredgecolor=colors[idx % len(colors)],
                markeredgewidth=0.55,
                markersize=2.25,
                linewidth=0.85,
            )
        inset.set_xlim(*xlim)
        inset.set_ylim(*ylim)
        inset.set_xticks(xticks)
        inset.set_yticks(yticks)
        if ylabel is not None:
            inset.set_ylabel(ylabel, fontsize=4.6, labelpad=1.0)
        inset.tick_params(
            direction="out",
            top=False,
            right=False,
            width=0.45,
            length=1.6,
            labelsize=3.9,
            pad=0.7,
        )
        for spine in inset.spines.values():
            spine.set_linewidth(0.45)
        inset.grid(True, color="0.78", linestyle=":", linewidth=0.3)
        if show_indicator:
            mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="0.35", linewidth=0.45)
        return inset

    _draw_inset(
        axes[1],
        plotted[1],
        bounds=(0.10, 0.57, 0.48, 0.32),
        xlim=(0.37, 0.39),
        ylim=(6.00, 6.80),
        xticks=(0.37, 0.38, 0.39),
        yticks=(6.00, 6.40, 6.80),
    )
    _draw_inset(
        axes[2],
        plotted[2],
        bounds=(0.49, 0.21, 0.44, 0.32),
        xlim=(0.05, 0.15),
        ylim=(0.0, 0.62),
        xticks=(0.05, 0.10, 0.15),
        yticks=(0.0, 0.2, 0.4, 0.6),
    )

    if classification is None or classification.empty:
        raise ValueError("The controlled classification CSV is required to plot Fig. 3(d).")
    conditioned = "snr_db" in classification and classification["snr_db"].notna().any()
    if conditioned:
        deployed = classification[classification["model"] == "iq_evm"]
        if deployed.empty:
            raise ValueError("SNR-conditioned Fig. 3(d) requires the iq_evm model.")
        styles = (
            (IEEE_TRANS_PALETTE["black"], "o"),
            (IEEE_TRANS_PALETTE["red"], "s"),
            (IEEE_TRANS_PALETTE["blue"], "^"),
            (IEEE_TRANS_PALETTE["green"], "D"),
        )
        for index, (snr_value, group) in enumerate(deployed.groupby("snr_db", sort=True)):
            color, marker = styles[index % len(styles)]
            group = group.sort_values("gamma_res")
            accuracy_ax.plot(
                group["gamma_res"],
                100.0 * group["accuracy"],
                color=color,
                linestyle="-",
                marker=marker,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.75,
                markersize=3.2,
                linewidth=1.25,
                label=fr"SNR = {float(snr_value):g} dB",
            )
        accuracy_ax.set_ylabel("I/Q--constellation accuracy (%)")
    else:
        accuracy_styles = {
            "iq_only": (IEEE_TRANS_PALETTE["blue"], "o", "I/Q only"),
            "evm_only": (IEEE_TRANS_PALETTE["orange"], "s", "EVM only"),
            "iq_evm": (IEEE_TRANS_PALETTE["purple"], "^", "I/Q + EVM"),
        }
        for name, group in classification.groupby("model", sort=False):
            if name not in accuracy_styles:
                continue
            color, marker, label = accuracy_styles[name]
            group = group.sort_values("gamma_res")
            accuracy_ax.plot(
                group["gamma_res"],
                100.0 * group["accuracy"],
                color=color,
                linestyle="-",
                marker=marker,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.75,
                markersize=3.2,
                linewidth=1.25,
                label=label,
            )
        accuracy_ax.set_ylabel("Fixed-model accuracy (%)")
    accuracy_ax.set_xlabel(r"Residual severity $\gamma_{\rm res}$")
    accuracy_limit = float(np.max(classification["gamma_res"].to_numpy(dtype=float)))
    accuracy_ax.set_xlim(0.0, max(accuracy_limit, x_limit))
    if accuracy_limit > x_limit:
        accuracy_ax.set_xticks(np.linspace(0.0, accuracy_limit, 7))
    accuracy_ax.set_ylim(0.0, 102.0)
    accuracy_ax.set_yticks((0, 20, 40, 60, 80, 100))
    format_ieee_axis(accuracy_ax)
    set_panel_aspect_10_9(accuracy_ax)
    boxed_legend(accuracy_ax, loc="lower left", fontsize=5.9)

    fig.subplots_adjust(left=0.085, right=0.995, top=0.98, bottom=0.115, hspace=0.47, wspace=0.38)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_axes_panels(fig, [axes[0], axes[1], axes[2], accuracy_ax], output)
    plt.close(fig)


def _first_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float:
    """Linearly interpolate the first upward threshold crossing."""
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    hit = np.flatnonzero(y >= threshold)
    if hit.size == 0:
        return float("nan")
    i = int(hit[0])
    if i == 0 or np.isclose(y[i], y[i - 1]):
        return float(x[i])
    return float(x[i - 1] + (threshold - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))


def _attenuation_boundary(k: int, epsilon: float) -> float:
    """Bisection boundary for 1-|A_k|=epsilon before the first lobe crossing."""
    lo, hi = 0.0, 10.0
    target = 1.0 - float(epsilon)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if float(discrete_attenuation_factor(mid, k=k)) > target:
            lo = mid
        else:
            hi = mid
    return float(hi)


def _load_split_indices(path: Path, split: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    with np.load(path) as splits:
        if split not in splits:
            raise KeyError(f"Split '{split}' is absent from {path}. Available: {splits.files}")
        return np.asarray(splits[split], dtype=np.int64)


def _classification_output_path(args: argparse.Namespace, output_root: Path) -> Path:
    return (
        Path(args.classification_output)
        if args.classification_output
        else output_root / "controlled_classification_accuracy_by_gamma_snr.csv"
    )


def _plot_with_existing_feature_metrics(
    output_root: Path,
    classification: pd.DataFrame,
    figure: Path,
    epsilon: float,
) -> None:
    controlled_path = output_root / "controlled_feature_metrics_by_gamma.csv"
    if not controlled_path.exists():
        raise FileNotFoundError(
            "The SNR-conditioned scan requires the existing Fig. 3(a)--(c) metrics: "
            f"{controlled_path}"
        )
    controlled = pd.read_csv(controlled_path)
    boundaries = pd.DataFrame(
        [
            {"k": k, "gamma_boundary": _attenuation_boundary(k, epsilon)}
            for k in (1, 2, 4, 6)
        ]
    )
    analytical_boundary = float(
        boundaries.loc[boundaries["k"].isin((2, 4, 6)), "gamma_boundary"].min()
    )
    primary_length = int(controlled["frame_length"].max())
    hoc = controlled[
        (controlled["metric"] == "hoc_sensitive_relative_error")
        & (controlled["frame_length"] == primary_length)
    ]
    empirical_boundary = _first_crossing(
        hoc["gamma_res"].to_numpy(), hoc["median"].to_numpy(), epsilon
    )
    _plot(
        controlled,
        classification,
        figure,
        epsilon,
        analytical_boundary,
        empirical_boundary,
    )


def main() -> None:
    args = parse_args()
    feature_root = Path(args.feature_root)
    result_root = Path(args.result_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.classification_only:
        if not args.classification_feature_data:
            raise ValueError("--classification-only requires --classification-feature-data.")
        if not args.classification_splits:
            raise ValueError("--classification-only requires --classification-splits.")
        if not args.classification_snr_db:
            raise ValueError(
                "--classification-only requires exact --classification-snr-db strata."
            )

        raw_path = Path(args.classification_raw_data or args.raw_data)
        feature_path = Path(args.classification_feature_data)
        split_path = Path(args.classification_splits)
        for path in (raw_path, feature_path, split_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing SNR-conditioned scan input: {path}")

        classification_path = _classification_output_path(args, output_root)
        if args.reuse_classification_scan:
            if not classification_path.exists():
                raise FileNotFoundError(
                    "--reuse-classification-scan requires an existing CSV: "
                    f"{classification_path}"
                )
            classification = pd.read_csv(classification_path)
        else:
            indices = _load_split_indices(split_path, args.classification_split)
            with h5py.File(raw_path, "r") as raw:
                labels = np.asarray(raw["label"][indices], dtype=np.int64)
                snr_db = np.asarray(raw["snr_db"][indices], dtype=np.float64)
            gamma_grid = np.unique(
                np.asarray(args.controlled_classification_gamma_grid, dtype=np.float64)
            )
            if np.any(gamma_grid < 0.0):
                raise ValueError("Controlled residual severities must be nonnegative.")
            classification, selected_indices, classification_detail = _controlled_classification_scan(
                raw_path=raw_path,
                feature_path=feature_path,
                indices=indices,
                labels=labels,
                snr_db=snr_db,
                gamma_grid=gamma_grid,
                samples_per_class_snr=int(args.classification_samples_per_class_snr),
                seed=int(args.controlled_seed),
                checkpoints={"iq_evm": Path(args.final_iq_evm_checkpoint)},
                device_name=args.device,
                batch_size=int(args.classification_batch_size),
                snr_values=np.asarray(args.classification_snr_db, dtype=np.float64),
                oracle_correct_estimated_iq=bool(
                    args.classification_oracle_correct_estimated_iq
                ),
            )
            classification_path.parent.mkdir(parents=True, exist_ok=True)
            classification.to_csv(classification_path, index=False)
            np.save(
                classification_path.with_name(
                    f"{classification_path.stem}_indices.npy"
                ),
                selected_indices,
            )
            if args.classification_detail_output:
                detail_path = Path(args.classification_detail_output)
                detail_path.parent.mkdir(parents=True, exist_ok=True)
                classification_detail.to_csv(detail_path, index=False)

        _plot_with_existing_feature_metrics(
            output_root,
            classification,
            Path(args.figure),
            float(args.epsilon),
        )
        summary = (
            classification.groupby(["snr_db", "gamma_res"], as_index=False)
            .agg(count=("count", "sum"), accuracy=("accuracy", "mean"))
            .sort_values(["snr_db", "gamma_res"])
        )
        summary.to_csv(
            classification_path.with_name(f"{classification_path.stem}_summary.csv"),
            index=False,
        )
        print(summary.to_string(index=False))
        print(f"Saved conditional accuracy scan: {classification_path}")
        print(f"Updated separate panels for: {args.figure}")
        return

    if args.plot_only:
        controlled_path = output_root / "controlled_feature_metrics_by_gamma.csv"
        classification_path = _classification_output_path(args, output_root)
        if not controlled_path.exists():
            raise FileNotFoundError(
                "Missing controlled-scan result for --plot-only: "
                f"{controlled_path}"
            )
        controlled = pd.read_csv(controlled_path)
        if not classification_path.exists():
            raise FileNotFoundError(
                "Missing fixed-model controlled accuracy result for --plot-only: "
                f"{classification_path}"
            )
        classification = pd.read_csv(classification_path)
        boundaries = pd.DataFrame(
            [
                {"k": k, "gamma_boundary": _attenuation_boundary(k, float(args.epsilon))}
                for k in (1, 2, 4, 6)
            ]
        )
        analytical_boundary = float(
            boundaries.loc[
                boundaries["k"].isin((2, 4, 6)), "gamma_boundary"
            ].min()
        )
        primary_length = int(controlled["frame_length"].max())
        hoc = controlled[
            (controlled["metric"] == "hoc_sensitive_relative_error")
            & (controlled["frame_length"] == primary_length)
        ]
        empirical_boundary = _first_crossing(
            hoc["gamma_res"].to_numpy(),
            hoc["median"].to_numpy(),
            float(args.epsilon),
        )
        _plot(
            controlled,
            classification,
            Path(args.figure),
            float(args.epsilon),
            analytical_boundary,
            empirical_boundary,
        )
        return

    feature_paths = {
        method: feature_root / method / "hoc_iqcomp.h5" for method in METHODS
    }
    eval_paths = {
        method: result_root / f"{method}_{args.split}.csv" for method in METHODS
    }
    missing = [str(path) for path in (*feature_paths.values(), *eval_paths.values()) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required ablation artifacts:\n" + "\n".join(missing))

    evaluation = {method: pd.read_csv(path) for method, path in eval_paths.items()}
    indices = evaluation["oracle"]["index"].to_numpy(dtype=np.int64)
    with h5py.File(feature_paths["oracle"], "r") as oracle:
        feature_names = _decode(oracle["feature_names"][:])
        oracle_hoc = np.asarray(oracle["h_drc"][indices], dtype=np.float64)
        snr_db = np.asarray(oracle["snr_db"][indices], dtype=np.float64)
        labels = np.asarray(oracle["label"][indices], dtype=np.int64)
    sensitive = _phase_sensitive_mask(feature_names)
    invariant = ~sensitive
    high_snr = snr_db >= float(args.high_snr_min_db)

    if args.controlled_only:
        controlled_gamma = np.unique(
            np.asarray(args.controlled_gamma_grid, dtype=np.float64)
        )
        classification_gamma = np.unique(
            np.asarray(args.controlled_classification_gamma_grid, dtype=np.float64)
        )
        if np.any(controlled_gamma < 0.0) or np.any(classification_gamma < 0.0):
            raise ValueError("Controlled residual severities must be nonnegative.")
        controlled = _controlled_residual_scan(
            feature_path=feature_paths["oracle"],
            indices=indices,
            labels=labels,
            snr_db=snr_db,
            feature_names=feature_names,
            gamma_grid=controlled_gamma,
            high_snr_min_db=float(args.high_snr_min_db),
            samples_per_class_snr=int(args.controlled_samples_per_class_snr),
            seed=int(args.controlled_seed),
            chunk_size=int(args.iq_chunk_size),
            frame_lengths=tuple(int(value) for value in args.controlled_frame_lengths),
            epsilon=float(args.epsilon),
        )
        controlled.to_csv(output_root / "controlled_feature_metrics_by_gamma.csv", index=False)
        boundary_table = pd.DataFrame(
            [
                {
                    "k": k,
                    "epsilon": float(args.epsilon),
                    "gamma_boundary": _attenuation_boundary(k, float(args.epsilon)),
                }
                for k in (1, 2, 4, 6)
            ]
        )
        boundary_table.to_csv(output_root / "theoretical_gamma_boundaries.csv", index=False)
        analytical_boundary = float(
            boundary_table.loc[
                boundary_table["k"].isin((2, 4, 6)), "gamma_boundary"
            ].min()
        )
        primary_length = int(controlled["frame_length"].max())
        hoc_controlled = controlled[
            (controlled["metric"] == "hoc_sensitive_relative_error")
            & (controlled["frame_length"] == primary_length)
        ]
        empirical_boundary = _first_crossing(
            hoc_controlled["gamma_res"].to_numpy(),
            hoc_controlled["median"].to_numpy(),
            float(args.epsilon),
        )
        classification_path = output_root / "controlled_classification_accuracy_by_gamma.csv"
        if args.skip_classification_sweep:
            if not classification_path.exists():
                raise FileNotFoundError(
                    "--skip-classification-sweep requires an existing controlled accuracy CSV: "
                    f"{classification_path}"
                )
            classification = pd.read_csv(classification_path)
            selected_indices = _select_controlled_indices(
                indices,
                labels,
                snr_db,
                int(args.controlled_samples_per_class_snr),
                int(args.controlled_seed),
                high_snr_min_db=float(args.high_snr_min_db),
            )
        else:
            classification, selected_indices, _ = _controlled_classification_scan(
                raw_path=Path(args.raw_data),
                feature_path=feature_paths["oracle"],
                indices=indices,
                labels=labels,
                snr_db=snr_db,
                gamma_grid=classification_gamma,
                high_snr_min_db=float(args.high_snr_min_db),
                samples_per_class_snr=int(args.controlled_samples_per_class_snr),
                seed=int(args.controlled_seed),
                checkpoints={
                    "iq_only": Path(args.iq_only_checkpoint),
                    "evm_only": Path(args.evm_only_checkpoint),
                    "iq_evm": Path(args.iq_evm_checkpoint),
                },
                device_name=args.device,
                batch_size=int(args.classification_batch_size),
            )
            classification.to_csv(classification_path, index=False)
            np.save(output_root / "controlled_classification_indices.npy", selected_indices)
        _plot(
            controlled,
            classification,
            Path(args.figure),
            epsilon=float(args.epsilon),
            analytical_boundary=analytical_boundary,
            empirical_boundary=empirical_boundary,
        )
        stability_rows: list[dict[str, float | int]] = []
        for frame_length, group in controlled[
            controlled["metric"] == "feature_stable"
        ].groupby("frame_length", sort=True):
            group = group.sort_values("gamma_res")
            below = group[group["mean"] < 0.90]
            stability_rows.append(
                {
                    "frame_length": int(frame_length),
                    "frame_duration_ms": float(group.iloc[0]["frame_duration_ms"]),
                    "gamma_first_below_p90": (
                        float(below.iloc[0]["gamma_res"]) if not below.empty else float("nan")
                    ),
                    "stability_at_gamma_0p05": float(
                        group.iloc[np.argmin(np.abs(group["gamma_res"].to_numpy() - 0.05))]["mean"]
                    ),
                    "stability_at_gamma_0p10": float(
                        group.iloc[np.argmin(np.abs(group["gamma_res"].to_numpy() - 0.10))]["mean"]
                    ),
                }
            )
        stability_table = pd.DataFrame(stability_rows)
        stability_table.to_csv(
            output_root / "feature_stability_by_frame_length.csv", index=False
        )
        (output_root / "controlled_summary.json").write_text(
            json.dumps(
                {
                    "mode": "controlled_only",
                    "high_snr_min_db": float(args.high_snr_min_db),
                    "epsilon": float(args.epsilon),
                    "selected_base_frames": int(selected_indices.size),
                    "controlled_gamma_grid": controlled_gamma.tolist(),
                    "controlled_classification_gamma_grid": classification_gamma.tolist(),
                    "frame_lengths": [int(value) for value in sorted(controlled["frame_length"].unique())],
                    "analytical_proxy_boundary": analytical_boundary,
                    "empirical_hoc_median_crossing": empirical_boundary,
                    "feature_stability": stability_rows,
                    "fixed_model_accuracy_csv": str(classification_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(stability_table.to_string(index=False))
        print(f"Saved separate panels for: {args.figure}")
        return

    gamma_bins = np.asarray(
        [0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0],
        dtype=np.float64,
    )
    metric_rows: list[dict[str, float | int | str]] = []
    accuracy_rows: list[dict[str, float | int | str]] = []

    for method in METHODS:
        with h5py.File(feature_paths[method], "r") as feature:
            gamma = np.asarray(feature["gamma_res"][indices], dtype=np.float64)
            hoc = np.asarray(feature["h_drc"][indices], dtype=np.float64)
        sensitive_error = _relative_vector_error(hoc, oracle_hoc, sensitive)
        invariant_error = _relative_vector_error(hoc, oracle_hoc, invariant)
        coherence = (
            np.ones(indices.size, dtype=np.float64)
            if method == "oracle"
            else _iq_coherence(
                feature_paths[method],
                feature_paths["oracle"],
                indices,
                chunk_size=int(args.iq_chunk_size),
            )
        )
        metric_rows.extend(
            _aggregate_continuous(
                gamma[high_snr],
                sensitive_error[high_snr],
                gamma_bins,
                method,
                "hoc_sensitive_relative_error",
            )
        )
        metric_rows.extend(
            _aggregate_continuous(
                gamma[high_snr],
                invariant_error[high_snr],
                gamma_bins,
                method,
                "hoc_invariant_relative_error",
            )
        )
        metric_rows.extend(
            _aggregate_continuous(
                gamma[high_snr],
                coherence[high_snr],
                gamma_bins,
                method,
                "iq_coherence",
            )
        )
        correct = evaluation[method]["correct"].to_numpy(dtype=bool)
        accuracy_rows.extend(_aggregate_accuracy(gamma, correct, gamma_bins, method))

    metrics = pd.DataFrame(metric_rows)
    accuracy = pd.DataFrame(accuracy_rows)
    metrics.to_csv(output_root / "feature_metrics_by_gamma.csv", index=False)
    accuracy.to_csv(output_root / "accuracy_by_gamma.csv", index=False)

    controlled_gamma = np.unique(
        np.asarray(args.controlled_gamma_grid, dtype=np.float64)
    )
    classification_gamma = np.unique(
        np.asarray(args.controlled_classification_gamma_grid, dtype=np.float64)
    )
    if np.any(controlled_gamma < 0.0) or np.any(classification_gamma < 0.0):
        raise ValueError("Controlled residual severities must be nonnegative.")
    controlled = _controlled_residual_scan(
        feature_path=feature_paths["oracle"],
        indices=indices,
        labels=labels,
        snr_db=snr_db,
        feature_names=feature_names,
        gamma_grid=controlled_gamma,
        high_snr_min_db=float(args.high_snr_min_db),
        samples_per_class_snr=int(args.controlled_samples_per_class_snr),
        seed=int(args.controlled_seed),
        chunk_size=int(args.iq_chunk_size),
        frame_lengths=tuple(int(value) for value in args.controlled_frame_lengths),
        epsilon=float(args.epsilon),
    )
    controlled.to_csv(
        output_root / "controlled_feature_metrics_by_gamma.csv",
        index=False,
    )

    gamma_grid = np.linspace(0.0, 48.0, 241)
    attenuation_rows: list[dict[str, float | int]] = []
    for k in (1, 2, 4, 6):
        for gamma, value in zip(gamma_grid, discrete_attenuation_factor(gamma_grid, k=k)):
            attenuation_rows.append(
                {"gamma": float(gamma), "k": int(k), "attenuation": float(value)}
            )
    attenuation = pd.DataFrame(attenuation_rows)
    attenuation.to_csv(output_root / "theoretical_attenuation.csv", index=False)

    blind = evaluation["blind"].copy()
    with h5py.File(feature_paths["blind"], "r") as feature:
        blind["mu_error_feature"] = np.asarray(feature["mu_error"][blind["index"]], dtype=np.float64)
        blind["gamma_res"] = np.asarray(feature["gamma_res"][blind["index"]], dtype=np.float64)
    blind["abs_mu_error"] = np.abs(blind["mu_error_feature"])
    dfrft_rows: list[dict[str, float | int]] = []
    for snr, group in blind.groupby("snr_db", sort=True):
        error = group["mu_error_feature"].to_numpy(dtype=np.float64)
        absolute = np.abs(error)
        dfrft_rows.append(
            {
                "snr_db": float(snr),
                "count": int(len(group)),
                "mae_hz_per_s": float(np.mean(absolute)),
                "rmse_hz_per_s": float(np.sqrt(np.mean(error**2))),
                "median_abs_hz_per_s": float(np.median(absolute)),
                "p90_abs_hz_per_s": float(np.quantile(absolute, 0.90)),
                "p95_abs_hz_per_s": float(np.quantile(absolute, 0.95)),
                "p99_abs_hz_per_s": float(np.quantile(absolute, 0.99)),
                "outlier_gt_500": float(np.mean(absolute > 500.0)),
                "outlier_gt_1000": float(np.mean(absolute > 1000.0)),
                "accuracy": float(group["correct"].mean()),
            }
        )
    dfrft_by_snr = pd.DataFrame(dfrft_rows)
    dfrft_by_snr.to_csv(output_root / "dfrft_error_by_snr.csv", index=False)

    error_bins = np.asarray([0.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, np.inf])
    conditional_rows: list[dict[str, float | int | str]] = []
    for lo, hi in zip(error_bins[:-1], error_bins[1:]):
        mask = (blind["abs_mu_error"] >= lo) & (blind["abs_mu_error"] < hi)
        if not np.any(mask):
            continue
        conditional_rows.append(
            {
                "abs_error_low": float(lo),
                "abs_error_high": float(hi),
                "count": int(np.sum(mask)),
                "accuracy": float(blind.loc[mask, "correct"].mean()),
                "median_gamma_res": float(blind.loc[mask, "gamma_res"].median()),
            }
        )
    pd.DataFrame(conditional_rows).to_csv(
        output_root / "dfrft_accuracy_by_error.csv",
        index=False,
    )

    boundary_rows: list[dict[str, float | int]] = []
    for k in (1, 2, 4, 6):
        boundary = _attenuation_boundary(k, float(args.epsilon))
        boundary_rows.append(
            {"k": k, "epsilon": float(args.epsilon), "gamma_boundary": boundary}
        )
    boundary_table = pd.DataFrame(boundary_rows)
    boundary_table.to_csv(output_root / "theoretical_gamma_boundaries.csv", index=False)

    analytical_boundary = float(
        boundary_table.loc[boundary_table["k"].isin((2, 4, 6)), "gamma_boundary"].min()
    )
    primary_length = int(controlled["frame_length"].max())
    hoc_controlled = controlled[
        (controlled["metric"] == "hoc_sensitive_relative_error")
        & (controlled["frame_length"] == primary_length)
    ]
    empirical_boundary = _first_crossing(
        hoc_controlled["gamma_res"].to_numpy(),
        hoc_controlled["median"].to_numpy(),
        float(args.epsilon),
    )
    classification_path = output_root / "controlled_classification_accuracy_by_gamma.csv"
    if args.skip_classification_sweep:
        if not classification_path.exists():
            raise FileNotFoundError(
                "--skip-classification-sweep requires an existing controlled accuracy CSV: "
                f"{classification_path}"
            )
        classification = pd.read_csv(classification_path)
    else:
        classification, selected_indices, _ = _controlled_classification_scan(
            raw_path=Path(args.raw_data),
            feature_path=feature_paths["oracle"],
            indices=indices,
            labels=labels,
            snr_db=snr_db,
            gamma_grid=classification_gamma,
            high_snr_min_db=float(args.high_snr_min_db),
            samples_per_class_snr=int(args.controlled_samples_per_class_snr),
            seed=int(args.controlled_seed),
            checkpoints={
                "iq_only": Path(args.iq_only_checkpoint),
                "evm_only": Path(args.evm_only_checkpoint),
                "iq_evm": Path(args.iq_evm_checkpoint),
            },
            device_name=args.device,
            batch_size=int(args.classification_batch_size),
        )
        classification.to_csv(classification_path, index=False)
        np.save(output_root / "controlled_classification_indices.npy", selected_indices)
    figure_path = Path(args.figure)
    _plot(
        controlled,
        classification,
        figure_path,
        epsilon=float(args.epsilon),
        analytical_boundary=analytical_boundary,
        empirical_boundary=empirical_boundary,
    )
    summary = {
        "feature_names": feature_names,
        "phase_sensitive_features": [
            name for name, keep in zip(feature_names, sensitive) if keep
        ],
        "high_snr_min_db": float(args.high_snr_min_db),
        "epsilon": float(args.epsilon),
        "hoc_phase_sensitivity_indices": [2, 4, 6],
        "analytical_feature_boundary": analytical_boundary,
        "empirical_hoc_error_boundary": empirical_boundary,
        "fixed_model_accuracy_csv": str(classification_path),
        "figure": str(figure_path),
        "controlled_residual_scan": {
            "gamma_grid": controlled_gamma.tolist(),
            "samples_per_class_snr": int(args.controlled_samples_per_class_snr),
            "seed": int(args.controlled_seed),
            "frame_lengths": sorted(
                int(value) for value in controlled["frame_length"].unique()
            ),
            "counts": {
                f"{metric}_N{int(frame_length)}": int(group["count"].max())
                for (metric, frame_length), group in controlled.groupby(["metric", "frame_length"])
            },
        },
        "dfrft_overall": {
            "mae_hz_per_s": float(blind["abs_mu_error"].mean()),
            "rmse_hz_per_s": float(
                np.sqrt(np.mean(blind["mu_error_feature"].to_numpy() ** 2))
            ),
            "median_abs_hz_per_s": float(blind["abs_mu_error"].median()),
            "p90_abs_hz_per_s": float(blind["abs_mu_error"].quantile(0.90)),
            "p95_abs_hz_per_s": float(blind["abs_mu_error"].quantile(0.95)),
            "p99_abs_hz_per_s": float(blind["abs_mu_error"].quantile(0.99)),
            "outlier_gt_500": float(np.mean(blind["abs_mu_error"] > 500.0)),
            "outlier_gt_1000": float(np.mean(blind["abs_mu_error"] > 1000.0)),
        },
    }
    stability = controlled[controlled["metric"] == "feature_stable"].copy()
    stability_rows: list[dict[str, float | int]] = []
    for frame_length, group in stability.groupby("frame_length", sort=True):
        group = group.sort_values("gamma_res")
        below = group[group["mean"] < 0.90]
        crossing = float(below.iloc[0]["gamma_res"]) if not below.empty else float("nan")
        stability_rows.append(
            {
                "frame_length": int(frame_length),
                "frame_duration_ms": float(group.iloc[0]["frame_duration_ms"]),
                "gamma_first_below_p90": crossing,
                "stability_at_gamma_0p05": float(
                    group.iloc[np.argmin(np.abs(group["gamma_res"].to_numpy() - 0.05))]["mean"]
                ),
                "stability_at_gamma_0p10": float(
                    group.iloc[np.argmin(np.abs(group["gamma_res"].to_numpy() - 0.10))]["mean"]
                ),
            }
        )
    stability_table = pd.DataFrame(stability_rows)
    stability_table.to_csv(output_root / "feature_stability_by_frame_length.csv", index=False)
    summary["empirical_feature_stability"] = stability_rows
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(boundary_table.to_string(index=False))
    print(dfrft_by_snr.to_string(index=False))
    print(json.dumps(summary["dfrft_overall"], indent=2))
    print(f"Saved separate panels for: {figure_path}")


if __name__ == "__main__":
    main()
