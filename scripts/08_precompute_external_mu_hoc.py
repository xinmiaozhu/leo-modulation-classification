#!/usr/bin/env python
"""Precompute DRC-HOC features using an externally supplied mu_hat.

This is intended for pilot-aided or trajectory-level mu estimates.  The normal
``11_precompute_drc_hoc.py`` script estimates mu internally before HOC
compensation; this script skips that blind estimator and uses an existing HDF5
dataset such as ``pilot_mu_hat`` or ``mu_hat_joint``.
"""

from __future__ import annotations

import argparse
import sys
from math import factorial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.drc_hoc.compensation import compensate_constant_doppler, conjugate_quadratic_compensation, residual_gamma
from src.drc_hoc.cumulants import _partitions_tuple, hoc_feature_names
from src.drc_hoc.extractor import DRCHOCConfig
from src.physics.gamma_metric import compute_gamma
from src.utils.config import load_config
from src.utils.io import save_json
from src.utils.logger import setup_logger
from src.utils.math_utils import complex_to_iq, to_complex


def _feature_from_complex(value: complex, representation: str) -> list[float]:
    if representation == "real_imag_abs":
        return [float(np.real(value)), float(np.imag(value)), float(np.abs(value))]
    if representation == "real_imag":
        return [float(np.real(value)), float(np.imag(value))]
    if representation == "abs":
        return [float(np.abs(value))]
    if representation == "real_imag_abs_phase":
        return [float(np.real(value)), float(np.imag(value)), float(np.abs(value)), float(np.angle(value))]
    raise ValueError(f"Unsupported representation: {representation}")


def _prepare_hoc_signal(x: np.ndarray, max_order: int, eps: float = 1.0e-12) -> tuple[np.ndarray, dict[tuple[int, int], complex]]:
    z = np.asarray(x, dtype=np.complex128).reshape(-1)
    z = z - np.mean(z)
    power = np.mean(np.abs(z) ** 2)
    z = z / np.sqrt(power + eps)

    zp = [np.ones_like(z)]
    cp = [np.ones_like(z)]
    z_conj = np.conj(z)
    for _ in range(max_order):
        zp.append(zp[-1] * z)
        cp.append(cp[-1] * z_conj)

    moments: dict[tuple[int, int], complex] = {(0, 0): 1.0 + 0.0j}
    for p in range(1, max_order + 1):
        for q in range(p + 1):
            moments[(p, q)] = complex(np.mean(zp[p - q] * cp[q]))
    return z, moments


def _cumulant_from_moments(types: tuple[int, ...], moments: dict[tuple[int, int], complex]) -> complex:
    total = 0.0 + 0.0j
    for partition in _partitions_tuple(len(types)):
        coeff = ((-1) ** (len(partition) - 1)) * factorial(len(partition) - 1)
        prod = 1.0 + 0.0j
        for block in partition:
            p = len(block)
            q = sum(types[i] for i in block)
            prod *= moments[(p, q)]
        total += coeff * prod
    return complex(total)


def extract_hoc_features_fast(
    x: np.ndarray,
    orders: tuple[tuple[int, int], ...],
    representation: str,
) -> np.ndarray:
    max_order = max(p for p, _ in orders)
    _, moments = _prepare_hoc_signal(x, max_order=max_order)
    feats: list[float] = []
    for p, q in orders:
        types = (0,) * (p - q) + (1,) * q
        c = _cumulant_from_moments(types, moments)
        feats.extend(_feature_from_complex(c, representation))
    return np.asarray(feats, dtype=np.float32)


def compute_V_hoc_fast(
    x_comp: np.ndarray,
    num_subwindows: int,
    orders: tuple[tuple[int, int], ...],
    representation: str,
) -> float:
    if num_subwindows <= 1:
        return 0.0
    x_comp = np.asarray(x_comp)
    seg_len = len(x_comp) // num_subwindows
    if seg_len < 8:
        raise ValueError(f"Subwindow too short for HOC extraction: N={len(x_comp)}, L={num_subwindows}.")
    h_arr = np.stack(
        [
            extract_hoc_features_fast(
                x_comp[i * seg_len : (i + 1) * seg_len],
                orders=orders,
                representation=representation,
            )
            for i in range(num_subwindows)
        ],
        axis=0,
    )
    h_mean = np.mean(h_arr, axis=0)
    return float(np.mean(np.sum((h_arr - h_mean) ** 2, axis=1)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute HOC features after external-mu compensation.")
    p.add_argument("--raw-data", type=str, required=True)
    p.add_argument("--mu-feature-data", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--mu-key", type=str, default="pilot_mu_hat")
    p.add_argument("--score-key", type=str, default="pilot_score")
    p.add_argument("--valid-key", type=str, default="pilot_valid")
    p.add_argument("--fd0-key", type=str, default=None)
    p.add_argument("--selected-k", type=int, default=0)
    p.add_argument("--num-subwindows", type=int, default=None)
    p.add_argument(
        "--skip-v-hoc",
        action="store_true",
        help=(
            "Skip local-subwindow HOC variance and store zero in V_hoc. "
            "Use this when the downstream model does not consume metadata."
        ),
    )
    p.add_argument(
        "--skip-hoc",
        action="store_true",
        help=(
            "Skip global HOC reconstruction and store zero placeholders. "
            "Use only when the downstream model does not consume HOC and "
            "the file is needed solely for iq_comp."
        ),
    )
    p.add_argument("--compensate-fd0", action="store_true")
    p.add_argument("--save-iq-comp", action="store_true", help="Save compensated I/Q as iq_comp for the model raw-IQ branch.")
    p.add_argument("--device", type=str, default="cpu", help="Device for the I/Q-only CUDA fast path.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size for the I/Q-only CUDA fast path.")
    p.add_argument(
        "--iq-compression",
        choices=("lzf", "none"),
        default="lzf",
        help="HDF5 compression for iq_comp. Use 'none' for fast contiguous writes.",
    )
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def build_config(args: argparse.Namespace, raw_h5_path: str) -> DRCHOCConfig:
    cfg_dict = {}
    if args.config:
        cfg = load_config(args.config)
        if "drc_hoc" in cfg:
            cfg_dict = cfg.drc_hoc.to_dict() if hasattr(cfg.drc_hoc, "to_dict") else dict(cfg.drc_hoc)
        else:
            cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)

    with h5py.File(raw_h5_path, "r") as f:
        n = int(f["iq"].shape[-1])
        fs = float(f["fs"][0]) if "fs" in f else float(cfg_dict.get("sample_rate_hz", 1.0e6))
        cfg_dict["sample_rate_hz"] = fs
        cfg_dict["observation_time_s"] = n / fs

    if args.num_subwindows is not None:
        cfg_dict["num_subwindows"] = args.num_subwindows
    if args.compensate_fd0:
        cfg_dict["compensate_fd0"] = True

    cfg_dict.pop("batch_size", None)

    # Estimator-only config fields may still be present; DRCHOCConfig accepts
    # them, but this script uses only compensation/HOC-related fields.
    return DRCHOCConfig(**cfg_dict)


def copy_meta(raw: h5py.File, out: h5py.File, keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in raw and key not in out:
            out.create_dataset(key, data=raw[key][:], compression="gzip")


def read_optional_scalar(h5: h5py.File, key: str | None, index: int, default: float = np.nan) -> float:
    if key and key in h5:
        return float(h5[key][index])
    return float(default)


def _cuda_iq_only_enabled(args: argparse.Namespace, cfg: DRCHOCConfig) -> bool:
    """Whether the exact I/Q-only compensation path can run in CUDA batches."""

    requested_cuda = str(args.device).lower().startswith("cuda")
    return bool(
        requested_cuda
        and torch.cuda.is_available()
        and args.save_iq_comp
        and args.skip_hoc
        and args.skip_v_hoc
    )


def _write_cuda_iq_only_batches(
    raw: h5py.File,
    mf: h5py.File,
    out: h5py.File,
    args: argparse.Namespace,
    cfg: DRCHOCConfig,
    hoc_dim: int,
) -> tuple[int, int]:
    """Write compensated I/Q in CUDA batches when HOC/meta HOC are unused.

    This path implements frame-start constant and quadratic phase compensation
    equivalent to :func:`compensate_full_doppler`, but avoids Python-frame
    overhead and per-row compressed HDF5 writes. It is intentionally narrow:
    HOC reconstruction still uses the scalar reference implementation.
    """

    n, _, num_samples = raw["iq"].shape
    device = torch.device(args.device)
    dtype = torch.float32
    t = torch.arange(num_samples, device=device, dtype=dtype) / float(cfg.sample_rate_hz)
    if cfg.centered_time:
        t = t - torch.mean(t)
    t_squared = t.square()
    observation_time = (
        float(cfg.observation_time_s)
        if cfg.observation_time_s is not None
        else float(num_samples) / float(cfg.sample_rate_hz)
    )
    batch_size = max(int(args.batch_size), 1)
    iterator = range(0, int(n), batch_size)
    if not args.no_progress:
        iterator = tqdm(iterator, total=(int(n) + batch_size - 1) // batch_size, desc="External-mu I/Q CUDA")

    num_valid = 0
    num_failed = 0
    for start in iterator:
        stop = min(start + batch_size, int(n))
        raw_iq = np.asarray(raw["iq"][start:stop], dtype=np.float32)
        mu_hat = np.asarray(mf[args.mu_key][start:stop], dtype=np.float32)
        external_valid = np.isfinite(mu_hat)
        if args.valid_key and args.valid_key in mf:
            external_valid &= np.asarray(mf[args.valid_key][start:stop], dtype=bool)
        score = (
            np.asarray(mf[args.score_key][start:stop], dtype=np.float32)
            if args.score_key and args.score_key in mf
            else np.zeros(stop - start, dtype=np.float32)
        )
        score = np.where(np.isfinite(score), score, 0.0).astype(np.float32)
        if args.fd0_key and args.fd0_key in mf:
            fd0_hat = np.asarray(mf[args.fd0_key][start:stop], dtype=np.float32)
        elif cfg.compensate_fd0 and "fd0" in raw:
            fd0_hat = np.asarray(raw["fd0"][start:stop], dtype=np.float32)
        else:
            fd0_hat = np.zeros(stop - start, dtype=np.float32)
        if args.fd0_key:
            external_valid &= np.isfinite(fd0_hat)

        iq_tensor = torch.from_numpy(raw_iq).to(device=device, dtype=dtype, non_blocking=True)
        x = torch.complex(iq_tensor[:, 0, :], iq_tensor[:, 1, :])
        mu_tensor = torch.from_numpy(np.nan_to_num(mu_hat, nan=0.0)).to(device=device, dtype=dtype)
        fd0_tensor = torch.from_numpy(np.nan_to_num(fd0_hat, nan=0.0)).to(device=device, dtype=dtype)
        phase_angle = (
            2.0 * torch.pi * fd0_tensor[:, None] * t[None, :]
            + torch.pi * mu_tensor[:, None] * t_squared[None, :]
        )
        phase = torch.exp(-1j * phase_angle)
        x_comp = x * phase
        comp_iq = torch.stack((x_comp.real, x_comp.imag), dim=1).cpu().numpy().astype(np.float32, copy=False)
        if not np.all(external_valid):
            comp_iq[~external_valid] = raw_iq[~external_valid]

        sl = slice(start, stop)
        out["iq_comp"][sl] = comp_iq
        out["h_drc"][sl] = 0.0
        out["mu_hat"][sl] = mu_hat
        out["S_peak"][sl] = score
        out["selected_k"][sl] = int(args.selected_k)
        out["external_mu_valid"][sl] = external_valid
        out["external_score"][sl] = score
        out["valid_flag"][sl] = external_valid
        if "external_fd0_hat" in out:
            out["external_fd0_hat"][sl] = fd0_hat
        out["V_hoc"][sl] = np.where(external_valid, 0.0, np.nan).astype(np.float32)
        gamma_hat = np.pi * np.abs(mu_hat) * observation_time**2
        out["gamma_hat"][sl] = np.where(external_valid, gamma_hat, np.nan).astype(np.float32)
        if "mu" in raw:
            mu_true = np.asarray(raw["mu"][sl], dtype=np.float32)
            mu_error = mu_hat - mu_true
            gamma_res = np.pi * np.abs(mu_error) * observation_time**2
            out["mu_error"][sl] = np.where(external_valid, mu_error, np.nan).astype(np.float32)
            out["gamma_res"][sl] = np.where(external_valid, gamma_res, np.nan).astype(np.float32)
        if "fd0_error" in out and "fd0" in raw:
            fd0_error = fd0_hat - np.asarray(raw["fd0"][sl], dtype=np.float32)
            out["fd0_error"][sl] = np.where(external_valid, fd0_error, np.nan).astype(np.float32)
        num_valid += int(np.count_nonzero(external_valid))
        num_failed += int(external_valid.size - np.count_nonzero(external_valid))

    return num_valid, num_failed


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}. Use --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    cfg = build_config(args, args.raw_data)
    logger = setup_logger("leo_drc_dualnet", output.with_suffix(".log"), reset=True)
    logger.info(f"External-mu HOC config: {cfg.to_dict()}")
    logger.info(f"mu_feature_data={args.mu_feature_data}, mu_key={args.mu_key}")

    feature_names = hoc_feature_names(cfg.hoc_orders, cfg.hoc_representation)
    hoc_dim = len(feature_names)

    with h5py.File(args.raw_data, "r") as raw, h5py.File(args.mu_feature_data, "r") as mf:
        if args.mu_key not in mf:
            raise SystemExit(f"mu key not found in {args.mu_feature_data}: {args.mu_key}")
        if args.score_key and args.score_key not in mf:
            logger.warning(f"score key not found: {args.score_key}; S_peak will be filled with 0.")
        if args.valid_key and args.valid_key not in mf:
            logger.warning(f"valid key not found: {args.valid_key}; all finite mu values will be treated as valid.")

        n = int(raw["iq"].shape[0])
        if len(mf[args.mu_key]) != n:
            raise SystemExit(f"mu length mismatch: raw has {n}, {args.mu_key} has {len(mf[args.mu_key])}")

        with h5py.File(output, "w") as out:
            # These arrays are populated one frame at a time. Contiguous
            # storage avoids repeatedly recompressing a default chunk that
            # spans hundreds of not-yet-written rows.
            out.create_dataset("h_drc", shape=(n, hoc_dim), dtype="float32")
            out.create_dataset("mu_hat", shape=(n,), dtype="float32")
            out.create_dataset("gamma_hat", shape=(n,), dtype="float32")
            out.create_dataset("S_peak", shape=(n,), dtype="float32")
            out.create_dataset("V_hoc", shape=(n,), dtype="float32")
            out.create_dataset("selected_k", shape=(n,), dtype="int64")
            out.create_dataset("valid_flag", shape=(n,), dtype="bool")
            out.create_dataset("external_mu_valid", shape=(n,), dtype="bool")
            out.create_dataset("external_score", shape=(n,), dtype="float32")
            if args.save_iq_comp:
                iq_shape = tuple(int(value) for value in raw["iq"].shape)
                iq_kwargs: dict[str, object] = {"shape": iq_shape, "dtype": "float32"}
                if args.iq_compression != "none":
                    iq_kwargs.update({"chunks": (1, iq_shape[1], iq_shape[2]), "compression": args.iq_compression})
                out.create_dataset("iq_comp", **iq_kwargs)
            if args.fd0_key:
                out.create_dataset("external_fd0_hat", shape=(n,), dtype="float32")
                if "fd0" in raw:
                    out.create_dataset("fd0_error", shape=(n,), dtype="float32")
            if "mu" in raw:
                out.create_dataset("gamma_res", shape=(n,), dtype="float32")
                out.create_dataset("mu_error", shape=(n,), dtype="float32")

            out.create_dataset("feature_names", data=np.asarray([s.encode("utf-8") for s in feature_names], dtype="S64"))
            copy_meta(
                raw,
                out,
                (
                    "track_id",
                    "frame_id",
                    "frame_time_s",
                    "fd0",
                    "snr_db",
                    "label",
                    "label_names",
                    "modulation",
                    "fs",
                    "fc",
                    "T",
                    "timing_offset_samples",
                ),
            )
            out.attrs["config"] = str(cfg.to_dict())
            out.attrs["raw_data"] = args.raw_data
            out.attrs["mu_feature_data"] = args.mu_feature_data
            out.attrs["mu_key"] = args.mu_key

            cuda_iq_only = _cuda_iq_only_enabled(args, cfg)
            if cuda_iq_only:
                logger.info(
                    "Using CUDA I/Q-only external-mu compensation: "
                    f"batch_size={args.batch_size}, iq_compression={args.iq_compression}."
                )
                num_valid, num_failed = _write_cuda_iq_only_batches(raw, mf, out, args, cfg, hoc_dim)
            else:
                if str(args.device).lower().startswith("cuda") and not torch.cuda.is_available():
                    logger.warning("CUDA requested for external-mu compensation but is unavailable; using scalar reference path.")
                elif str(args.device).lower().startswith("cuda"):
                    logger.info("CUDA I/Q-only fast path is inapplicable; using scalar reference path.")
                iterator = range(n)
                if not args.no_progress:
                    iterator = tqdm(iterator, desc="External-mu HOC")

                num_valid = 0
                num_failed = 0
                for i in iterator:
                    mu_hat = float(mf[args.mu_key][i])
                    external_valid = bool(np.isfinite(mu_hat))
                    if args.valid_key and args.valid_key in mf:
                        external_valid = external_valid and bool(mf[args.valid_key][i])

                    score = read_optional_scalar(mf, args.score_key, i, default=0.0) if args.score_key else 0.0
                    fd0_hat = 0.0
                    if args.fd0_key and args.fd0_key in mf:
                        fd0_hat = float(mf[args.fd0_key][i])
                        out["external_fd0_hat"][i] = fd0_hat
                    elif cfg.compensate_fd0 and "fd0" in raw:
                        fd0_hat = float(raw["fd0"][i])

                    out["mu_hat"][i] = mu_hat
                    out["S_peak"][i] = score if np.isfinite(score) else 0.0
                    out["selected_k"][i] = int(args.selected_k)
                    out["external_mu_valid"][i] = external_valid
                    out["external_score"][i] = score if np.isfinite(score) else np.nan

                    if not external_valid:
                        if args.save_iq_comp:
                            out["iq_comp"][i] = raw["iq"][i]
                        out["h_drc"][i] = np.zeros(hoc_dim, dtype=np.float32)
                        out["gamma_hat"][i] = np.nan
                        out["V_hoc"][i] = np.nan
                        out["valid_flag"][i] = False
                        if "mu" in raw:
                            out["gamma_res"][i] = np.nan
                            out["mu_error"][i] = np.nan
                        if "fd0_error" in out:
                            out["fd0_error"][i] = np.nan
                        num_failed += 1
                        continue

                    try:
                        r_in = to_complex(raw["iq"][i])
                        if cfg.compensate_fd0 and abs(fd0_hat) > 0:
                            r_base = compensate_constant_doppler(r_in, cfg.sample_rate_hz, fd0_hat, cfg.centered_time)
                        else:
                            r_base = r_in

                        r_comp = conjugate_quadratic_compensation(r_base, cfg.sample_rate_hz, mu_hat, cfg.centered_time)
                        if args.save_iq_comp:
                            out["iq_comp"][i] = complex_to_iq(r_comp, layout="channel_first")
                        h_drc = (
                            np.zeros(hoc_dim, dtype=np.float32)
                            if args.skip_hoc
                            else extract_hoc_features_fast(
                                r_comp,
                                orders=cfg.hoc_orders,
                                representation=cfg.hoc_representation,
                            )
                        )
                        v_hoc = (
                            0.0
                            if args.skip_v_hoc
                            else compute_V_hoc_fast(
                                r_comp,
                                num_subwindows=cfg.num_subwindows,
                                orders=cfg.hoc_orders,
                                representation=cfg.hoc_representation,
                            )
                        )

                        T = cfg.observation_time_s if cfg.observation_time_s is not None else len(r_comp) / cfg.sample_rate_hz
                        gamma_hat = float(compute_gamma(mu_hat, T))
                        out["h_drc"][i] = h_drc.astype(np.float32)
                        out["gamma_hat"][i] = gamma_hat
                        out["V_hoc"][i] = float(v_hoc)
                        out["valid_flag"][i] = True
                        if "mu" in raw:
                            mu_true = float(raw["mu"][i])
                            out["gamma_res"][i] = residual_gamma(mu_true, mu_hat, T)
                            out["mu_error"][i] = float(mu_hat - mu_true)
                        if "fd0_error" in out and "fd0" in raw:
                            out["fd0_error"][i] = float(fd0_hat - raw["fd0"][i])
                        num_valid += 1
                    except Exception as exc:
                        if args.save_iq_comp:
                            out["iq_comp"][i] = raw["iq"][i]
                        out["h_drc"][i] = np.zeros(hoc_dim, dtype=np.float32)
                        out["gamma_hat"][i] = np.nan
                        out["V_hoc"][i] = np.nan
                        out["valid_flag"][i] = False
                        if "mu" in raw:
                            out["gamma_res"][i] = np.nan
                            out["mu_error"][i] = np.nan
                        if "fd0_error" in out:
                            out["fd0_error"][i] = np.nan
                        num_failed += 1
                        logger.warning(f"Failed at sample {i}: {exc}")

    summary = {
        "raw_data": args.raw_data,
        "mu_feature_data": args.mu_feature_data,
        "output": str(output),
        "num_samples": int(n),
        "valid_count": int(num_valid),
        "failed_count": int(num_failed),
        "hoc_dim": int(hoc_dim),
        "mu_key": args.mu_key,
        "fd0_key": args.fd0_key,
        "score_key": args.score_key,
        "save_iq_comp": bool(args.save_iq_comp),
        "skip_v_hoc": bool(args.skip_v_hoc),
        "skip_hoc": bool(args.skip_hoc),
        "device": str(args.device),
        "batch_size": int(args.batch_size),
        "iq_compression": str(args.iq_compression),
        "config": cfg.to_dict(),
    }
    with h5py.File(args.raw_data, "r") as raw:
        has_mu = "mu" in raw
    if has_mu:
        with h5py.File(output, "r") as out:
            e = out["mu_error"][:]
            e = e[np.isfinite(e)]
            summary["mu_mae"] = float(np.mean(np.abs(e))) if len(e) else float("nan")
            summary["mu_rmse"] = float(np.sqrt(np.mean(e**2))) if len(e) else float("nan")
            if "fd0_error" in out:
                e_fd0 = out["fd0_error"][:]
                e_fd0 = e_fd0[np.isfinite(e_fd0)]
                summary["fd0_mae"] = float(np.mean(np.abs(e_fd0))) if len(e_fd0) else float("nan")
                summary["fd0_rmse"] = float(np.sqrt(np.mean(e_fd0**2))) if len(e_fd0) else float("nan")
    save_json(summary, output.with_suffix(".summary.json"))
    print(summary)
    logger.info(f"Saved external-mu HOC features to {output}")


if __name__ == "__main__":
    main()
