#!/usr/bin/env python
"""Precompute pilot-aided per-frame Doppler-rate estimates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.drc_hoc.pilot_estimator import PilotMuEstimator, PilotMuEstimatorConfig, PilotMuResult, make_grid
from src.drc_hoc.hybrid_dfrft_estimator import HybridDFRFTConfig, HybridDFRFTPilotEstimator
from src.signal.pulse_shape import rrc_filter
from src.utils.io import save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute pilot-aided mu_hat estimates.")
    p.add_argument("--raw-data", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--sample-rate-hz", type=float, default=None)
    p.add_argument("--samples-per-symbol", type=int, default=8)
    p.add_argument("--rrc-beta", type=float, default=0.35)
    p.add_argument("--rrc-span", type=int, default=8)
    p.add_argument("--timing-phases", type=int, nargs="*", default=[0])
    p.add_argument("--mu-min", type=float, default=-8160.0)
    p.add_argument("--mu-max", type=float, default=-180.0)
    p.add_argument("--coarse-step", type=float, default=40.0)
    p.add_argument("--fine-radius", type=float, default=120.0)
    p.add_argument("--fine-step", type=float, default=5.0)
    p.add_argument("--fd0-min", type=float, default=0.0)
    p.add_argument("--fd0-max", type=float, default=0.0)
    p.add_argument("--fd0-step", type=float, default=100.0)
    p.add_argument(
        "--fd0-truth-key",
        type=str,
        default="fd0",
        help=(
            "Raw HDF5 CFO reference used only for diagnostic error metrics. "
            "Use fd0_frame_start when an injected integer timing shift changes "
            "the phase-law linear coefficient in the frame-start coordinate."
        ),
    )
    p.add_argument("--fine-fd0-radius", type=float, default=None)
    p.add_argument("--fine-fd0-step", type=float, default=None)
    p.add_argument("--amplitude-score", action="store_true", help="Use amplitude-weighted coherent score instead of phase-only score.")
    p.add_argument(
        "--pilot-weighting",
        choices=("phase", "coherent", "soft", "thresholded_phase"),
        default="coherent",
        help="Pilot preprocessing mode. The coherent AWGN receiver is the default.",
    )
    p.add_argument("--soft-weight-scale", type=float, default=1.0)
    p.add_argument(
        "--amplitude-threshold-rel",
        type=float,
        default=0.35,
        help="Median-relative amplitude floor for thresholded phase-only weighting.",
    )
    p.add_argument("--min-pilots", type=int, default=8)
    p.add_argument(
        "--center-time",
        action="store_true",
        help="Orthogonalize the joint CFO--rate search at the pilot-time centroid.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Use CUDA batch scoring for the fixed-timing coherent receiver, including joint CFO--Doppler searches.",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--estimator-type",
        choices=("coherent_grid", "dfrft_only", "hybrid_dfrft"),
        default="coherent_grid",
        help="Coherent grid, DFRFT acquisition only, or DFRFT acquisition with coherent local refinement.",
    )
    p.add_argument("--dfrft-coarse-mu-step", type=float, default=160.0)
    p.add_argument("--dfrft-fft-size", type=int, default=2048)
    p.add_argument("--dfrft-fine-mu-radius", type=float, default=200.0)
    p.add_argument("--dfrft-fine-mu-step", type=float, default=5.0)
    p.add_argument("--dfrft-fine-fd0-radius", type=float, default=30.0)
    p.add_argument("--dfrft-fine-fd0-step", type=float, default=1.0)
    return p.parse_args()


def copy_meta(raw: h5py.File, out: h5py.File, keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in raw and key not in out:
            out.create_dataset(key, data=raw[key][:], compression="gzip")


def _cuda_coherent_supported(args: argparse.Namespace) -> bool:
    """Whether the fixed-timing coherent search can use CUDA batching."""
    return bool(
        str(args.device).startswith("cuda")
        and args.estimator_type == "coherent_grid"
        and torch.cuda.is_available()
        and args.pilot_weighting == "coherent"
        and not args.center_time
        and tuple(args.timing_phases) == (0,)
        and int(args.batch_size) > 0
    )


def _cuda_hybrid_supported(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device).startswith("cuda")
        and torch.cuda.is_available()
        and args.estimator_type in {"dfrft_only", "hybrid_dfrft"}
        and args.pilot_weighting == "coherent"
        and tuple(args.timing_phases) == (0,)
        and int(args.batch_size) > 0
    )


@torch.inference_mode()
def _estimate_hybrid_cuda_batch(
    iq: np.ndarray,
    pilot_indices: np.ndarray,
    pilot_symbols: np.ndarray,
    *,
    config: PilotMuEstimatorConfig,
    hybrid_config: HybridDFRFTConfig,
    device: torch.device,
    rrc_taps: np.ndarray,
    refine: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CUDA batch implementation of the frozen pilot DFRFT--coherent hybrid."""

    frame = np.asarray(iq, dtype=np.float32)
    indices = np.asarray(pilot_indices, dtype=np.int64)
    symbols = np.asarray(pilot_symbols, dtype=np.complex64)
    if frame.ndim != 3 or frame.shape[1] != 2:
        raise ValueError(f"Expected [B, 2, N] I/Q batch, got {frame.shape}.")
    if indices.ndim != 2 or symbols.ndim != 2 or indices.shape != symbols.shape:
        raise ValueError("CUDA hybrid path requires rectangular pilot arrays.")

    batch, _, length = frame.shape
    sample_indices = indices * int(config.samples_per_symbol)
    valid = (sample_indices >= 0) & (sample_indices < length) & (np.abs(symbols) > 1e-12)
    if not np.all(valid):
        raise ValueError("CUDA hybrid path requires valid known pilot positions for every frame.")
    spacing = np.diff(sample_indices, axis=1)
    if spacing.size == 0 or not np.all(spacing == spacing[0, 0]):
        raise ValueError("CUDA hybrid path requires uniformly spaced comb pilots.")

    kernel = torch.as_tensor(rrc_taps[::-1].copy(), dtype=torch.float32, device=device).reshape(1, 1, -1)
    real = torch.as_tensor(frame[:, 0], dtype=torch.float32, device=device).unsqueeze(1)
    imag = torch.as_tensor(frame[:, 1], dtype=torch.float32, device=device).unsqueeze(1)
    padding = int(kernel.shape[-1] // 2)
    matched_real = torch.nn.functional.conv1d(real, kernel, padding=padding).squeeze(1)
    matched_imag = torch.nn.functional.conv1d(imag, kernel, padding=padding).squeeze(1)
    matched = torch.complex(matched_real, matched_imag)

    sample_index_t = torch.as_tensor(sample_indices, dtype=torch.long, device=device)
    observed = torch.gather(matched, 1, sample_index_t)
    pilot = torch.as_tensor(symbols, dtype=torch.complex64, device=device)
    z = observed * torch.conj(pilot) / (torch.abs(pilot) ** 2 + 1e-12)
    t = sample_index_t.to(torch.float32) / float(config.sample_rate_hz)
    center = t.mean(dim=1)
    tau = t - center[:, None]
    pilot_count = int(z.shape[1])

    coarse_mu_grid = torch.as_tensor(
        make_grid(config.mu_min, config.mu_max, hybrid_config.coarse_mu_step_hz_per_s),
        dtype=torch.float32,
        device=device,
    )
    coarse_phase = torch.polar(
        torch.ones((batch, coarse_mu_grid.numel(), pilot_count), dtype=torch.float32, device=device),
        -np.pi * coarse_mu_grid[None, :, None] * tau[:, None, :].square(),
    )
    fft_size = max(int(hybrid_config.fft_size), pilot_count)
    spectrum = torch.fft.fftshift(
        torch.fft.fft(z[:, None, :] * coarse_phase, n=fft_size, dim=-1),
        dim=-1,
    )
    power = torch.abs(spectrum).square() / float(pilot_count * pilot_count)
    dt = float(spacing[0, 0]) / float(config.sample_rate_hz)
    frequency = torch.fft.fftshift(torch.fft.fftfreq(fft_size, d=dt, device=device))
    lower = float(config.fd0_min_hz) + center[:, None] * coarse_mu_grid[None, :]
    upper = float(config.fd0_max_hz) + center[:, None] * coarse_mu_grid[None, :]
    allowed = (
        (frequency[None, None, :] >= torch.minimum(lower, upper)[:, :, None])
        & (frequency[None, None, :] <= torch.maximum(lower, upper)[:, :, None])
    )
    flat_power = power.masked_fill(~allowed, -torch.inf).reshape(batch, -1)
    coarse_score, flat_index = torch.max(flat_power, dim=1)
    coarse_mu_index = torch.div(flat_index, fft_size, rounding_mode="floor")
    frequency_index = flat_index.remainder(fft_size)
    coarse_mu = coarse_mu_grid[coarse_mu_index]
    centered_cfo = frequency[frequency_index]
    coarse_fd0 = centered_cfo - coarse_mu * center

    if not refine:
        return (
            coarse_mu.cpu().numpy().astype(np.float64),
            coarse_fd0.cpu().numpy().astype(np.float64),
            coarse_score.cpu().numpy().astype(np.float64),
            np.full(batch, np.nan, dtype=np.float64),
            np.full(batch, pilot_count, dtype=np.int64),
        )

    mu_step = float(hybrid_config.fine_mu_step_hz_per_s)
    mu_radius = float(hybrid_config.fine_mu_radius_hz_per_s)
    mu_offsets = torch.arange(-mu_radius, mu_radius + 0.5 * mu_step, mu_step, device=device)
    fine_mu = torch.clamp(
        coarse_mu[:, None] + mu_offsets[None, :],
        min=float(config.mu_min),
        max=float(config.mu_max),
    )
    fd_step = float(hybrid_config.fine_fd0_step_hz)
    fd_radius = float(hybrid_config.fine_fd0_radius_hz)
    fd_offsets = torch.arange(-fd_radius, fd_radius + 0.5 * fd_step, fd_step, device=device)
    fine_phase = torch.polar(
        torch.ones((batch, fine_mu.shape[1], pilot_count), dtype=torch.float32, device=device),
        -np.pi * fine_mu[:, :, None] * t[:, None, :].square(),
    )
    best_score = torch.full((batch,), -torch.inf, dtype=torch.float32, device=device)
    second_score = torch.full_like(best_score, -torch.inf)
    best_mu = coarse_mu.clone()
    best_fd0 = coarse_fd0.clone()
    for offset in fd_offsets:
        candidate_fd0 = torch.clamp(
            coarse_fd0 + offset,
            min=float(config.fd0_min_hz),
            max=float(config.fd0_max_hz),
        )
        linear = torch.polar(
            torch.ones((batch, pilot_count), dtype=torch.float32, device=device),
            -2.0 * np.pi * candidate_fd0[:, None] * t,
        )
        scores = torch.abs(torch.sum(fine_phase * (z * linear)[:, None, :], dim=-1)).square()
        scores = scores / float(pilot_count * pilot_count)
        values, locations = torch.topk(scores, k=min(2, scores.shape[1]), dim=1)
        local_best = values[:, 0]
        local_second = values[:, 1] if values.shape[1] > 1 else torch.full_like(local_best, -torch.inf)
        improved = local_best > best_score
        second_score = torch.where(
            improved,
            torch.maximum(best_score, local_second),
            torch.maximum(second_score, local_best),
        )
        best_score = torch.where(improved, local_best, best_score)
        selected_mu = torch.gather(fine_mu, 1, locations[:, :1]).squeeze(1)
        best_mu = torch.where(improved, selected_mu, best_mu)
        best_fd0 = torch.where(improved, candidate_fd0, best_fd0)
    margin = (best_score - second_score) / (torch.abs(second_score) + 1e-12)
    return (
        best_mu.cpu().numpy().astype(np.float64),
        best_fd0.cpu().numpy().astype(np.float64),
        best_score.cpu().numpy().astype(np.float64),
        margin.cpu().numpy().astype(np.float64),
        np.full(batch, pilot_count, dtype=np.int64),
    )


@torch.inference_mode()
def _estimate_coherent_cuda_batch(
    iq: np.ndarray,
    pilot_indices: np.ndarray,
    pilot_symbols: np.ndarray,
    *,
    config: PilotMuEstimatorConfig,
    device: torch.device,
    rrc_taps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batch the coherent CFO--Doppler search for fixed pilot timing.

    The score matches :meth:`PilotMuEstimator._score_grid`.  The implementation
    streams CFO candidates over CUDA rather than materializing a four-dimensional
    tensor, which keeps memory bounded for a two-dimensional coarse grid.
    """
    frame = np.asarray(iq, dtype=np.float32)
    indices = np.asarray(pilot_indices, dtype=np.int64)
    symbols = np.asarray(pilot_symbols, dtype=np.complex64)
    if frame.ndim != 3 or frame.shape[1] != 2:
        raise ValueError(f"Expected [B, 2, N] I/Q batch, got {frame.shape}.")
    if indices.ndim != 2 or symbols.ndim != 2 or indices.shape != symbols.shape:
        raise ValueError("CUDA coherent path requires rectangular pilot index/symbol arrays.")

    batch, _, length = frame.shape
    sample_indices = indices * int(config.samples_per_symbol)
    valid = (sample_indices >= 0) & (sample_indices < length) & (np.abs(symbols) > 1e-12)
    if not np.all(valid):
        raise ValueError("CUDA coherent path requires valid known pilot positions for every frame.")

    # torch.conv1d uses cross correlation, hence the tap reversal.  This is the
    # same centered convolution used by the scalar estimator.
    kernel = torch.as_tensor(rrc_taps[::-1].copy(), dtype=torch.float32, device=device).reshape(1, 1, -1)
    real = torch.as_tensor(frame[:, 0], dtype=torch.float32, device=device).unsqueeze(1)
    imag = torch.as_tensor(frame[:, 1], dtype=torch.float32, device=device).unsqueeze(1)
    padding = int(kernel.shape[-1] // 2)
    matched_real = torch.nn.functional.conv1d(real, kernel, padding=padding).squeeze(1)
    matched_imag = torch.nn.functional.conv1d(imag, kernel, padding=padding).squeeze(1)
    matched = torch.complex(matched_real, matched_imag)

    sample_index_t = torch.as_tensor(sample_indices, dtype=torch.long, device=device)
    observed = torch.gather(matched, 1, sample_index_t)
    pilot = torch.as_tensor(symbols, dtype=torch.complex64, device=device)
    z = observed * torch.conj(pilot) / (torch.abs(pilot) ** 2 + 1e-12)
    t = sample_index_t.to(torch.float32) / float(config.sample_rate_hz)
    t2 = t.square()
    pilot_count = int(z.shape[1])

    def score_mu_grid(
        mu: torch.Tensor,
        fd0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return top two scores and the best mu index for one CFO per frame."""
        # mu: [G] or [B, G]; fd0: scalar or [B].
        if mu.ndim == 1:
            angle = -np.pi * mu.view(1, -1, 1) * t2[:, None, :]
        else:
            angle = -np.pi * mu[:, :, None] * t2[:, None, :]
        phase = torch.polar(torch.ones_like(angle), angle)
        fd0 = fd0.reshape(-1)
        if fd0.numel() == 1:
            fd0 = fd0.expand(batch)
        linear_angle = -2.0 * np.pi * fd0[:, None] * t
        linear = torch.polar(torch.ones_like(linear_angle), linear_angle)
        accum = torch.sum((z * linear)[:, None, :] * phase, dim=-1)
        scores = torch.abs(accum).square() / float(pilot_count * pilot_count)
        if scores.shape[1] == 1:
            return scores[:, 0], torch.zeros(batch, dtype=torch.long, device=device), torch.full_like(scores[:, 0], -torch.inf)
        values, locations = torch.topk(scores, k=2, dim=1)
        return values[:, 0], locations[:, 0], values[:, 1]

    def update_global(
        best_score: torch.Tensor,
        second_score: torch.Tensor,
        best_mu: torch.Tensor,
        best_fd0: torch.Tensor,
        score: torch.Tensor,
        local_second: torch.Tensor,
        mu_value: torch.Tensor,
        fd0_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        improved = score > best_score
        next_second = torch.maximum(second_score, torch.maximum(best_score, local_second))
        retained_second = torch.maximum(second_score, torch.maximum(score, local_second))
        return (
            torch.where(improved, score, best_score),
            torch.where(improved, next_second, retained_second),
            torch.where(improved, mu_value, best_mu),
            torch.where(improved, fd0_value, best_fd0),
        )

    coarse_mu_grid = torch.as_tensor(
        make_grid(config.mu_min, config.mu_max, config.coarse_step_hz_per_s),
        dtype=torch.float32,
        device=device,
    )
    coarse_fd0_grid = torch.as_tensor(
        make_grid(config.fd0_min_hz, config.fd0_max_hz, config.fd0_step_hz),
        dtype=torch.float32,
        device=device,
    )
    best_score = torch.full((batch,), -torch.inf, dtype=torch.float32, device=device)
    second_score = torch.full_like(best_score, -torch.inf)
    coarse_mu = torch.zeros(batch, dtype=torch.float32, device=device)
    coarse_fd0 = torch.zeros(batch, dtype=torch.float32, device=device)
    for fd0_candidate in coarse_fd0_grid:
        score, index, local_second = score_mu_grid(coarse_mu_grid, fd0_candidate.reshape(1))
        best_score, second_score, coarse_mu, coarse_fd0 = update_global(
            best_score,
            second_score,
            coarse_mu,
            coarse_fd0,
            score,
            local_second,
            coarse_mu_grid[index],
            fd0_candidate.expand(batch),
        )

    radius = float(config.fine_radius_hz_per_s)
    step = float(config.fine_step_hz_per_s)
    if radius > 0.0 and step > 0.0:
        offsets = torch.arange(-radius, radius + 0.5 * step, step, dtype=torch.float32, device=device)
        fine_mu_grid = torch.clamp(coarse_mu[:, None] + offsets[None, :], min=float(config.mu_min), max=float(config.mu_max))
    else:
        fine_mu_grid = coarse_mu[:, None]

    fd0_radius = float(config.fine_fd0_radius_hz or 0.0)
    fd0_step = float(config.fine_fd0_step_hz or config.fd0_step_hz)
    if fd0_radius > 0.0 and fd0_step > 0.0:
        fd0_offsets = torch.arange(-fd0_radius, fd0_radius + 0.5 * fd0_step, fd0_step, dtype=torch.float32, device=device)
    else:
        fd0_offsets = torch.zeros(1, dtype=torch.float32, device=device)

    fine_score = torch.full_like(best_score, -torch.inf)
    fine_second = torch.full_like(best_score, -torch.inf)
    fine_mu = coarse_mu.clone()
    fine_fd0 = coarse_fd0.clone()
    for fd0_offset in fd0_offsets:
        candidate_fd0 = torch.clamp(
            coarse_fd0 + fd0_offset,
            min=float(config.fd0_min_hz),
            max=float(config.fd0_max_hz),
        )
        score, index, local_second = score_mu_grid(fine_mu_grid, candidate_fd0)
        selected_mu = torch.gather(fine_mu_grid, 1, index[:, None]).squeeze(1)
        fine_score, fine_second, fine_mu, fine_fd0 = update_global(
            fine_score,
            fine_second,
            fine_mu,
            fine_fd0,
            score,
            local_second,
            selected_mu,
            candidate_fd0,
        )
    margin = (fine_score - fine_second) / (torch.abs(fine_second) + 1e-12)
    return (
        fine_mu.cpu().numpy().astype(np.float64),
        fine_fd0.cpu().numpy().astype(np.float64),
        fine_score.cpu().numpy().astype(np.float64),
        margin.cpu().numpy().astype(np.float64),
        np.full(batch, pilot_count, dtype=np.int64),
    )


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}. Use --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    with h5py.File(args.raw_data, "r") as raw:
        required = ("iq", "pilot_indices", "pilot_symbols")
        missing = [key for key in required if key not in raw]
        if missing:
            raise SystemExit(f"Raw data is missing pilot datasets: {missing}. Regenerate with pilot.enabled=true.")

        fs = float(args.sample_rate_hz) if args.sample_rate_hz is not None else float(raw["fs"][0])
        cfg = PilotMuEstimatorConfig(
            sample_rate_hz=fs,
            samples_per_symbol=args.samples_per_symbol,
            rrc_beta=args.rrc_beta,
            rrc_span=args.rrc_span,
            timing_phases=tuple(args.timing_phases),
            mu_min=args.mu_min,
            mu_max=args.mu_max,
            coarse_step_hz_per_s=args.coarse_step,
            fine_radius_hz_per_s=args.fine_radius,
            fine_step_hz_per_s=args.fine_step,
            fd0_min_hz=args.fd0_min,
            fd0_max_hz=args.fd0_max,
            fd0_step_hz=args.fd0_step,
            fine_fd0_radius_hz=args.fine_fd0_radius,
            fine_fd0_step_hz=args.fine_fd0_step,
            phase_only=(args.pilot_weighting == "phase") if args.pilot_weighting else not args.amplitude_score,
            pilot_weighting=args.pilot_weighting,
            soft_weight_scale=args.soft_weight_scale,
            amplitude_threshold_rel=args.amplitude_threshold_rel,
            min_pilots=args.min_pilots,
            center_time=args.center_time,
        )
        hybrid_cfg = HybridDFRFTConfig(
            coarse_mu_step_hz_per_s=args.dfrft_coarse_mu_step,
            fft_size=args.dfrft_fft_size,
            fine_mu_radius_hz_per_s=args.dfrft_fine_mu_radius,
            fine_mu_step_hz_per_s=args.dfrft_fine_mu_step,
            fine_fd0_radius_hz=args.dfrft_fine_fd0_radius,
            fine_fd0_step_hz=args.dfrft_fine_fd0_step,
        )
        estimator = (
            HybridDFRFTPilotEstimator(cfg, hybrid_cfg)
            if args.estimator_type in {"dfrft_only", "hybrid_dfrft"}
            else PilotMuEstimator(cfg)
        )
        n = int(raw["iq"].shape[0])

        with h5py.File(output, "w") as out:
            for key, dtype in {
                "mu_hat": "float32",
                "pilot_mu_hat": "float32",
                "pilot_fd0_hat": "float32",
                "pilot_score": "float32",
                "pilot_score_margin_rel": "float32",
                "pilot_timing": "int64",
                "pilot_num_used": "int64",
                "pilot_valid": "bool",
            }.items():
                out.create_dataset(key, shape=(n,), dtype=dtype, compression="gzip")

            copy_meta(
                raw,
                out,
                (
                    "track_id",
                    "frame_id",
                    "frame_time_s",
                    "snr_db",
                    "fd0",
                    "fd0_frame_start",
                    "timing_offset_samples",
                    "label",
                    "label_names",
                    "modulation",
                    "pilot_indices",
                    "pilot_sample_indices",
                    "pilot_symbols",
                    "pilot_pattern",
                    "num_pilots",
                    "fs",
                    "fc",
                    "T",
                ),
            )

            has_mu = "mu" in raw
            if has_mu:
                out.create_dataset("mu_error", shape=(n,), dtype="float32", compression="gzip")
                out.create_dataset("gamma_res", shape=(n,), dtype="float32", compression="gzip")
            has_fd0 = bool(args.fd0_truth_key and args.fd0_truth_key in raw)
            if has_fd0:
                out.create_dataset("fd0_error", shape=(n,), dtype="float32", compression="gzip")

            use_cuda_coherent = _cuda_coherent_supported(args)
            use_cuda_hybrid = _cuda_hybrid_supported(args)
            use_cuda_batch = use_cuda_coherent or use_cuda_hybrid
            if use_cuda_batch:
                device = torch.device(args.device)
                rrc = rrc_filter(beta=args.rrc_beta, span=args.rrc_span, sps=args.samples_per_symbol)
                description = (
                    ("Pilot DFRFT acquisition" if args.estimator_type == "dfrft_only" else
                     "Pilot DFRFT--coherent hybrid estimation")
                    if use_cuda_hybrid
                    else "Pilot-aided coherent mu estimation"
                )
                progress = None if args.no_progress else tqdm(total=n, desc=description)
                for start in range(0, n, int(args.batch_size)):
                    stop = min(start + int(args.batch_size), n)
                    estimate_function = (
                        _estimate_hybrid_cuda_batch if use_cuda_hybrid else _estimate_coherent_cuda_batch
                    )
                    estimate_kwargs = {
                        "config": cfg,
                        "device": device,
                        "rrc_taps": rrc,
                    }
                    if use_cuda_hybrid:
                        estimate_kwargs["hybrid_config"] = hybrid_cfg
                        estimate_kwargs["refine"] = args.estimator_type == "hybrid_dfrft"
                    mu_hat, fd0_hat, score, margin, pilot_count = estimate_function(
                        raw["iq"][start:stop],
                        raw["pilot_indices"][start:stop],
                        raw["pilot_symbols"][start:stop],
                        **estimate_kwargs,
                    )
                    valid = np.isfinite(mu_hat)
                    out["mu_hat"][start:stop] = mu_hat.astype(np.float32)
                    out["pilot_mu_hat"][start:stop] = mu_hat.astype(np.float32)
                    out["pilot_fd0_hat"][start:stop] = fd0_hat.astype(np.float32)
                    out["pilot_score"][start:stop] = score.astype(np.float32)
                    out["pilot_score_margin_rel"][start:stop] = margin.astype(np.float32)
                    out["pilot_timing"][start:stop] = 0
                    out["pilot_num_used"][start:stop] = pilot_count
                    out["pilot_valid"][start:stop] = valid
                    if has_mu:
                        truth = np.asarray(raw["mu"][start:stop], dtype=np.float64)
                        error = np.where(valid, mu_hat - truth, np.nan)
                        if "T" in raw:
                            duration = np.asarray(raw["T"][start:stop], dtype=np.float64)
                        else:
                            duration = np.full(stop - start, raw["iq"].shape[-1] / fs, dtype=np.float64)
                        out["mu_error"][start:stop] = error.astype(np.float32)
                        out["gamma_res"][start:stop] = (np.pi * np.abs(error) * duration**2).astype(np.float32)
                    if has_fd0:
                        truth_fd0 = np.asarray(raw[args.fd0_truth_key][start:stop], dtype=np.float64)
                        out["fd0_error"][start:stop] = np.where(valid, fd0_hat - truth_fd0, np.nan).astype(np.float32)
                    if progress is not None:
                        progress.update(stop - start)
                if progress is not None:
                    progress.close()
            else:
                iterator = range(n)
                if not args.no_progress:
                    iterator = tqdm(iterator, desc="Pilot-aided mu estimation")

                for i in iterator:
                    result = estimator.estimate(raw["iq"][i], raw["pilot_indices"][i], raw["pilot_symbols"][i])
                    if args.estimator_type == "hybrid_dfrft":
                        result = result.estimate
                    elif args.estimator_type == "dfrft_only":
                        result = PilotMuResult(
                            result.coarse_mu_hz_per_s,
                            result.coarse_fd0_hz,
                            result.coarse_score,
                            result.coarse_margin_rel,
                            result.estimate.timing,
                            result.estimate.num_pilots_used,
                            result.estimate.valid,
                        )
                    mu_hat = result.mu_hat_hz_per_s
                    out["mu_hat"][i] = mu_hat
                    out["pilot_mu_hat"][i] = mu_hat
                    out["pilot_fd0_hat"][i] = result.fd0_hat_hz
                    out["pilot_score"][i] = result.score
                    out["pilot_score_margin_rel"][i] = result.score_margin_rel
                    out["pilot_timing"][i] = result.timing
                    out["pilot_num_used"][i] = result.num_pilots_used
                    out["pilot_valid"][i] = result.valid
                    if has_mu:
                        err = float(mu_hat - raw["mu"][i]) if result.valid else np.nan
                        out["mu_error"][i] = err
                        T = float(raw["T"][i]) if "T" in raw else len(raw["iq"][i, 0]) / fs
                        out["gamma_res"][i] = np.pi * abs(err) * T**2 if np.isfinite(err) else np.nan
                    if has_fd0:
                        err_fd0 = (
                            float(result.fd0_hat_hz - raw[args.fd0_truth_key][i])
                            if result.valid
                            else np.nan
                        )
                        out["fd0_error"][i] = err_fd0

            out.attrs["pilot_mu_config_json"] = json.dumps(cfg.to_dict())
            out.attrs["pilot_estimator_type"] = args.estimator_type
            if args.estimator_type in {"dfrft_only", "hybrid_dfrft"}:
                out.attrs["hybrid_dfrft_config_json"] = json.dumps(hybrid_cfg.to_dict())
            out.attrs["raw_data"] = args.raw_data

        summary = {
            "raw_data": args.raw_data,
            "output": str(output),
            "num_samples": n,
            "config": cfg.to_dict(),
            "estimator_type": args.estimator_type,
            "hybrid_dfrft_config": (
                hybrid_cfg.to_dict()
                if args.estimator_type in {"dfrft_only", "hybrid_dfrft"}
                else None
            ),
            "fd0_truth_key": args.fd0_truth_key if has_fd0 else None,
        }
        with h5py.File(output, "r") as out:
            valid = out["pilot_valid"][:].astype(bool)
            summary["valid_fraction"] = float(np.mean(valid))
            if has_mu:
                e = out["mu_error"][:]
                e = e[np.isfinite(e)]
                summary["pilot_mae"] = float(np.mean(np.abs(e)))
                summary["pilot_rmse"] = float(np.sqrt(np.mean(e**2)))
            if has_fd0:
                e_fd0 = out["fd0_error"][:]
                e_fd0 = e_fd0[np.isfinite(e_fd0)]
                summary["pilot_fd0_mae"] = float(np.mean(np.abs(e_fd0)))
                summary["pilot_fd0_rmse"] = float(np.sqrt(np.mean(e_fd0**2)))
        save_json(summary, output.with_suffix(".summary.json"))
        print(summary)


if __name__ == "__main__":
    main()
