#!/usr/bin/env python
"""Precompute NASA-HOC baseline features for the paper-aligned comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
from tqdm import tqdm

from src.drc_hoc.paper_features import NASA_FEATURE_NAMES, nasa_feature_vector
from src.signal.pulse_shape import rrc_filter
from src.utils.io import save_json
from src.utils.math_utils import to_complex


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute faithful NASA-HOC paper features.")
    p.add_argument("--raw-data", required=True)
    p.add_argument("--comp-feature-data", required=True)
    p.add_argument("--symbol-feature-data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples-per-symbol", type=int, default=8)
    p.add_argument("--rrc-beta", type=float, default=0.35)
    p.add_argument("--rrc-span", type=int, default=8)
    p.add_argument("--symbol-trim", type=int, default=8)
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--min-symbols", type=int, default=32)
    p.add_argument(
        "--chunk-rows",
        type=int,
        default=0,
        help="Rows read from compressed iq_comp at once; 0 follows its HDF5 row chunk.",
    )
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _copy_dataset(source: h5py.File, output: h5py.File, key: str) -> None:
    if key in source and key not in output and isinstance(source[key], h5py.Dataset):
        output.create_dataset(key, data=source[key][:], compression="gzip")


def _extract_symbols(
    compensated_iq: np.ndarray,
    timing: int,
    gain: complex,
    pilot_indices: np.ndarray,
    taps: np.ndarray,
    sps: int,
    trim: int,
    max_symbols: int,
) -> np.ndarray:
    matched = np.convolve(to_complex(compensated_iq), taps, mode="same")
    num_symbols = len(matched) // sps
    symbol_index = np.arange(num_symbols, dtype=np.int64)
    sample_index = symbol_index * sps + int(timing)
    valid = sample_index < len(matched)
    if trim > 0:
        valid &= (symbol_index >= trim) & (symbol_index < num_symbols - trim)

    pilots = np.asarray(pilot_indices, dtype=np.int64).reshape(-1)
    if pilots.size:
        valid &= ~np.isin(symbol_index, pilots)
    symbols = matched[sample_index[valid]] / (gain + 1e-12)
    if max_symbols > 0 and symbols.size > max_symbols:
        keep = np.linspace(0, symbols.size - 1, max_symbols, dtype=np.int64)
        symbols = symbols[keep]
    return symbols.astype(np.complex128)


def _create_output(
    path: Path,
    n: int,
    raw: h5py.File,
    comp: h5py.File,
) -> h5py.File:
    out = h5py.File(path, "w")
    out.create_dataset("h_drc", shape=(n, 10), dtype="float32", compression="gzip")
    out.create_dataset("paper_feature_valid", shape=(n,), dtype="bool", compression="gzip")
    out.create_dataset("symbol_count_paper", shape=(n,), dtype="int64", compression="gzip")
    out.create_dataset("processed", shape=(n,), dtype="bool", compression="gzip")
    out.create_dataset(
        "feature_names",
        data=np.asarray([name.encode("utf-8") for name in NASA_FEATURE_NAMES], dtype="S64"),
    )
    for key in (
        "label",
        "label_names",
        "modulation",
        "snr_db",
        "gamma",
        "mu",
        "fd0",
        "domain_id",
        "track_id",
        "frame_id",
        "frame_time_s",
        "fs",
        "fc",
        "T",
    ):
        _copy_dataset(raw, out, key)
    for key in ("gamma_hat", "S_peak", "V_hoc", "mu_hat", "selected_k"):
        _copy_dataset(comp, out, key)
    out.attrs["complete"] = False
    return out


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and args.overwrite:
        output.unlink()

    taps = rrc_filter(
        beta=float(args.rrc_beta),
        span=int(args.rrc_span),
        sps=max(int(args.samples_per_symbol), 1),
    )
    sps = max(int(args.samples_per_symbol), 1)

    with (
        h5py.File(args.raw_data, "r") as raw,
        h5py.File(args.comp_feature_data, "r") as comp,
        h5py.File(args.symbol_feature_data, "r") as symbol,
    ):
        n = int(raw["iq"].shape[0])
        if len(comp["h_drc"]) != n or len(symbol["pilot_evm"]) != n:
            raise ValueError("Raw, compensated-HOC and symbol feature files must be row aligned.")
        if "iq_comp" not in comp:
            raise KeyError("comp-feature-data must contain iq_comp.")

        if output.exists():
            out = h5py.File(output, "r+")
            if bool(out.attrs.get("complete", False)):
                print(f"Skip complete feature file: {output}")
                out.close()
                return
            if len(out["h_drc"]) != n:
                out.close()
                raise ValueError("Incomplete output length does not match current input data.")
        else:
            out = _create_output(output, n, raw, comp)
            out.flush()

        try:
            processed = np.asarray(out["processed"][:], dtype=bool)
            pending_count = int(np.count_nonzero(~processed))
            source_chunk_rows = int(comp["iq_comp"].chunks[0]) if comp["iq_comp"].chunks else 128
            chunk_rows = int(args.chunk_rows) if int(args.chunk_rows) > 0 else source_chunk_rows

            timing_all = np.asarray(symbol["timing_phase"][:], dtype=np.int64)
            gain_all = (
                np.asarray(symbol["complex_gain_real"][:], dtype=np.float64)
                + 1j * np.asarray(symbol["complex_gain_imag"][:], dtype=np.float64)
            )
            pilot_evm_all = np.asarray(symbol["pilot_evm"][:], dtype=np.float64)
            symbol_valid_all = np.asarray(symbol["symbol_valid"][:], dtype=bool)
            pilot_indices_all = np.asarray(raw["pilot_indices"][:], dtype=np.int64)

            progress = tqdm(
                total=pending_count,
                desc="Paper NASA-HOC features",
                disable=args.no_progress,
            )
            for start in range(0, n, chunk_rows):
                end = min(start + chunk_rows, n)
                block_pending = ~processed[start:end]
                if not np.any(block_pending):
                    continue

                # iq_comp is gzip-compressed in large row chunks.  Reading the
                # contiguous block once avoids decompressing the same chunk for
                # every individual frame.
                iq_block = np.asarray(comp["iq_comp"][start:end], dtype=np.float32)
                feature_block = np.asarray(out["h_drc"][start:end], dtype=np.float32)
                valid_block = np.asarray(out["paper_feature_valid"][start:end], dtype=bool)
                count_block = np.asarray(out["symbol_count_paper"][start:end], dtype=np.int64)
                processed_block = processed[start:end].copy()

                for local in np.flatnonzero(block_pending):
                    i = start + int(local)
                    symbols = _extract_symbols(
                        iq_block[local],
                        timing=max(int(timing_all[i]), 0),
                        gain=complex(gain_all[i]),
                        pilot_indices=pilot_indices_all[i],
                        taps=taps,
                        sps=sps,
                        trim=max(int(args.symbol_trim), 0),
                        max_symbols=max(int(args.max_symbols), 0),
                    )
                    valid = bool(
                        symbol_valid_all[i]
                        and symbols.size >= max(int(args.min_symbols), 1)
                    )
                    if valid:
                        features = nasa_feature_vector(symbols, float(pilot_evm_all[i]))
                    else:
                        features = np.zeros(10, dtype=np.float32)

                    feature_block[local] = features
                    valid_block[local] = valid
                    count_block[local] = int(symbols.size)
                    processed_block[local] = True

                out["h_drc"][start:end] = feature_block
                out["paper_feature_valid"][start:end] = valid_block
                out["symbol_count_paper"][start:end] = count_block
                out["processed"][start:end] = processed_block
                processed[start:end] = processed_block
                out.flush()
                progress.update(int(np.count_nonzero(block_pending)))
            progress.close()

            out.attrs["complete"] = bool(np.all(out["processed"][:]))
            out.flush()
        finally:
            out.close()

    summary = {
        "raw_data": args.raw_data,
        "comp_feature_data": args.comp_feature_data,
        "symbol_feature_data": args.symbol_feature_data,
        "output": str(output),
        "num_samples": n,
        "samples_per_symbol": sps,
        "rrc_beta": float(args.rrc_beta),
        "rrc_span": int(args.rrc_span),
        "symbol_trim": int(args.symbol_trim),
        "max_symbols": int(args.max_symbols),
        "snr_input": "pilot_esn0_est_db=-10*log10(pilot_evm_mse), clipped to [-20, 40] dB",
        "uses_true_snr": False,
        "uses_true_modulation": False,
    }
    save_json(summary, output.with_suffix(".json"))
    print(summary)


if __name__ == "__main__":
    main()
