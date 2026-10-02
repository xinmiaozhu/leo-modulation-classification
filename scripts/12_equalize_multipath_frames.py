#!/usr/bin/env python
"""Apply oracle-tap or comb-pilot LS regularized frequency-domain equalization."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.signal.pulse_shape import pulse_shape_symbols


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-data", required=True)
    p.add_argument("--compensated-features", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["oracle", "pilot_ls"], required=True)
    p.add_argument("--channel-length", type=int, default=8)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--equalizer-regularization", type=float, default=1e-3)
    p.add_argument("--sps", type=int, default=8)
    p.add_argument("--rrc-beta", type=float, default=0.35)
    p.add_argument("--rrc-span", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def pilot_only_waveform(
    n_samples: int,
    pilot_indices: np.ndarray,
    pilot_symbols: np.ndarray,
    *,
    sps: int,
    beta: float,
    span: int,
) -> np.ndarray:
    n_symbols = n_samples // sps
    symbols = np.zeros(n_symbols, dtype=np.complex128)
    valid = (pilot_indices >= 0) & (pilot_indices < n_symbols)
    symbols[pilot_indices[valid]] = pilot_symbols[valid]
    x = pulse_shape_symbols(symbols, sps=sps, beta=beta, span=span, mode="same")
    if x.size < n_samples:
        x = np.pad(x, (0, n_samples - x.size))
    return np.asarray(x[:n_samples], dtype=np.complex128)


def estimate_taps_ls(y: np.ndarray, pilot_waveform: np.ndarray, length: int, ridge: float) -> np.ndarray:
    n = y.size
    design = np.zeros((n, length), dtype=np.complex128)
    for lag in range(length):
        if lag == 0:
            design[:, lag] = pilot_waveform
        else:
            design[lag:, lag] = pilot_waveform[:-lag]
    gram = design.conj().T @ design
    rhs = design.conj().T @ y
    return np.linalg.solve(gram + float(ridge) * np.trace(gram).real / max(length, 1) * np.eye(length), rhs)


def equalize(y: np.ndarray, taps: np.ndarray, regularization: float) -> np.ndarray:
    n = y.size
    nfft = 1 << int(np.ceil(np.log2(n + max(taps.size - 1, 0))))
    response = np.fft.fft(taps, nfft)
    inverse = np.conj(response) / (np.abs(response) ** 2 + float(regularization))
    return np.fft.ifft(np.fft.fft(y, nfft) * inverse)[:n]


def main() -> None:
    args = parse_args()
    source = Path(args.compensated_features)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output}; use --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)

    with h5py.File(args.raw_data, "r") as raw, h5py.File(output, "r+") as feature:
        if "channel_taps" not in raw:
            raise KeyError("Raw dataset does not contain channel_taps; regenerate with tap provenance.")
        n = int(raw["iq"].shape[0])
        length = int(args.channel_length)
        if "equalizer_taps" in feature:
            del feature["equalizer_taps"]
        if "equalizer_tap_nmse" in feature:
            del feature["equalizer_tap_nmse"]
        estimated_all = feature.create_dataset("equalizer_taps", shape=(n, length), dtype="complex64", compression="gzip")
        nmse_all = feature.create_dataset("equalizer_tap_nmse", shape=(n,), dtype="float32", compression="gzip")
        for i in range(n):
            y_iq = np.asarray(feature["iq_comp"][i], dtype=np.float64)
            y = y_iq[0] + 1j * y_iq[1]
            true = np.asarray(raw["channel_taps"][i], dtype=np.complex128)[:length]
            if true.size < length:
                true = np.pad(true, (0, length - true.size))
            if args.mode == "oracle":
                taps = true
            else:
                pilot = pilot_only_waveform(
                    y.size,
                    np.asarray(raw["pilot_indices"][i], dtype=np.int64),
                    np.asarray(raw["pilot_symbols"][i], dtype=np.complex128),
                    sps=int(args.sps), beta=float(args.rrc_beta), span=int(args.rrc_span),
                )
                taps = estimate_taps_ls(y, pilot, length, float(args.ridge))
            y_eq = equalize(y, taps, float(args.equalizer_regularization))
            feature["iq_comp"][i, 0] = np.real(y_eq).astype(np.float32)
            feature["iq_comp"][i, 1] = np.imag(y_eq).astype(np.float32)
            estimated_all[i] = taps.astype(np.complex64)
            denom = float(np.sum(np.abs(true) ** 2)) + 1e-12
            nmse_all[i] = float(np.sum(np.abs(taps - true) ** 2) / denom)
        provenance = {
            "mode": args.mode,
            "channel_length": length,
            "ridge": float(args.ridge),
            "equalizer_regularization": float(args.equalizer_regularization),
            "source": str(source),
            "raw_data": args.raw_data,
        }
        feature.attrs["equalizer_json"] = json.dumps(provenance)
        feature.flush()
    print(f"Saved {args.mode} equalized frames: {output}")


if __name__ == "__main__":
    main()
