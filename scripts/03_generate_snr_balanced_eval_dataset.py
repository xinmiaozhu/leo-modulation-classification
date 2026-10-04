#!/usr/bin/env python
"""Generate an SNR-balanced LEO dataset.

The normal trajectory dataset is useful for training and track-level
generalization tests, but SNR-specific confusion matrices need enough samples
for every modulation inside each SNR bin.  This script creates a controlled
set with fixed SNR points and equal samples per modulation.  By default it
keeps the original evaluation-only behavior; optional train/val/test counts
create real balanced splits for model training and plotting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import tqdm

from src.signal.signal_generator import LEOSignalGenerator, SignalSpec
from src.utils.io import save_h5, save_json
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate an SNR-balanced dataset.")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--splits-output", type=str, required=True)
    p.add_argument("--summary-output", type=str, default=None)
    p.add_argument(
        "--modulations",
        type=str,
        nargs="+",
        default=["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "16APSK", "32APSK"],
    )
    p.add_argument("--snr-db", type=float, nargs="+", default=[0.0, 5.0])
    p.add_argument("--samples-per-mod-snr", type=int, default=100)
    p.add_argument(
        "--test-only",
        action="store_true",
        help=(
            "Create an evaluation-only data set whose split file has empty "
            "train/val indices and places every generated sample in test. "
            "This is safer for an independent final test set than the legacy "
            "evaluation mode, which aliases the samples into train and test."
        ),
    )
    p.add_argument(
        "--train-samples-per-mod-snr",
        type=int,
        default=None,
        help=(
            "Samples per modulation and SNR assigned to train. If omitted "
            "with all split-specific counts omitted, the script preserves "
            "the legacy evaluation-only split."
        ),
    )
    p.add_argument(
        "--val-samples-per-mod-snr",
        type=int,
        default=None,
        help="Samples per modulation and SNR assigned to val.",
    )
    p.add_argument(
        "--test-samples-per-mod-snr",
        type=int,
        default=None,
        help="Samples per modulation and SNR assigned to test.",
    )
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--num-symbols", type=int, default=1024)
    p.add_argument("--samples-per-symbol", type=int, default=8)
    p.add_argument("--sample-rate-hz", type=float, default=200000.0)
    p.add_argument("--carrier-frequency-hz", type=float, default=30.0e9)
    p.add_argument("--target-num-samples", type=int, default=8192)
    p.add_argument("--rrc-beta", type=float, default=0.35)
    p.add_argument("--rrc-span", type=int, default=8)
    p.add_argument("--mu-min", type=float, default=-8160.0)
    p.add_argument("--mu-max", type=float, default=-180.0)
    p.add_argument("--fd0-hz", type=float, default=0.0)
    p.add_argument("--fd0-min-hz", type=float, default=None)
    p.add_argument("--fd0-max-hz", type=float, default=None)
    p.add_argument("--fractional-timing-min-samples", type=float, default=0.0)
    p.add_argument("--fractional-timing-max-samples", type=float, default=0.0)
    p.add_argument("--rician-k-db", type=float, default=10.0)
    p.add_argument("--phase-noise-std-rad", type=float, default=0.0)
    p.add_argument("--phase-noise-mode", choices=["iid", "random_walk"], default="random_walk")
    p.add_argument("--channel-type", type=str, default="awgn")
    p.add_argument("--multipath-delays", type=int, nargs="+", default=[0, 3, 7])
    p.add_argument("--multipath-gains-db", type=float, nargs="+", default=[0, -6, -10])
    p.add_argument("--multipath-doppler-hz", type=float, nargs="+", default=[])
    p.add_argument("--paired-seeds", action="store_true",
                   help="Independent per-frame symbol, channel and noise streams for paired channel comparisons.")
    p.add_argument("--pilot-pattern", type=str, default="comb")
    p.add_argument("--pilot-interval-symbols", type=int, default=16)
    p.add_argument("--pilot-guard-symbols", type=int, default=0)
    p.add_argument("--pilot-modulation", type=str, default="QPSK")
    p.add_argument("--pilot-symbol-mode", type=str, default="pn")
    p.add_argument("--pilot-seed", type=int, default=2026)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _resolve_split_plan(args: argparse.Namespace) -> tuple[list[tuple[str, int]], bool]:
    if args.test_only:
        explicit_counts = any(
            value is not None
            for value in (
                args.train_samples_per_mod_snr,
                args.val_samples_per_mod_snr,
                args.test_samples_per_mod_snr,
            )
        )
        if explicit_counts:
            raise SystemExit("--test-only cannot be combined with split-specific sample counts.")
        n = int(args.samples_per_mod_snr)
        if n <= 0:
            raise SystemExit("--samples-per-mod-snr must be positive.")
        return [("test", n)], True

    explicit = any(
        value is not None
        for value in (
            args.train_samples_per_mod_snr,
            args.val_samples_per_mod_snr,
            args.test_samples_per_mod_snr,
        )
    )
    if not explicit:
        n = int(args.samples_per_mod_snr)
        if n <= 0:
            raise SystemExit("--samples-per-mod-snr must be positive.")
        return [("eval", n)], False

    plan = [
        ("train", int(args.train_samples_per_mod_snr or 0)),
        ("val", int(args.val_samples_per_mod_snr or 0)),
        ("test", int(args.test_samples_per_mod_snr or 0)),
    ]
    if any(count < 0 for _, count in plan):
        raise SystemExit("Split-specific sample counts must be non-negative.")
    if sum(count for _, count in plan) <= 0:
        raise SystemExit("At least one split-specific sample count must be positive.")
    if int(args.train_samples_per_mod_snr or 0) <= 0:
        raise SystemExit("--train-samples-per-mod-snr must be positive in split mode.")
    if int(args.val_samples_per_mod_snr or 0) <= 0:
        raise SystemExit("--val-samples-per-mod-snr must be positive in split mode.")
    return [(name, count) for name, count in plan if count > 0], True


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    splits_output = Path(args.splits_output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}. Use --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    splits_output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    modulations = [m.upper() for m in args.modulations]
    snr_values = [float(x) for x in args.snr_db]
    generator = LEOSignalGenerator(modulations=modulations, rng=rng)
    split_plan, split_mode = _resolve_split_plan(args)

    total = len(modulations) * len(snr_values) * sum(count for _, count in split_plan)
    iterator = []
    for split_name, split_count in split_plan:
        for mod in modulations:
            for snr in snr_values:
                for rep in range(int(split_count)):
                    iterator.append((split_name, mod, snr, rep))
    if not args.no_progress:
        iterator = tqdm(iterator, desc="Generating SNR-balanced dataset")

    iq_list: list[np.ndarray] = []
    label_list: list[int] = []
    snr_list: list[float] = []
    fd0_list: list[float] = []
    fractional_timing_list: list[float] = []
    mu_list: list[float] = []
    gamma_list: list[float] = []
    modulation_list: list[bytes] = []
    domain_list: list[int] = []
    altitude_list: list[float] = []
    inclination_list: list[float] = []
    fs_list: list[float] = []
    fc_list: list[float] = []
    t_list: list[float] = []
    track_id_list: list[int] = []
    frame_id_list: list[int] = []
    frame_time_list: list[float] = []
    track_frame_count_list: list[int] = []
    track_mu_center_list: list[float] = []
    track_mu_edge_list: list[float] = []
    track_profile_center_list: list[float] = []
    track_profile_width_list: list[float] = []
    pilot_indices_list: list[np.ndarray] = []
    pilot_sample_indices_list: list[np.ndarray] = []
    pilot_symbols_list: list[np.ndarray] = []
    pilot_pattern_list: list[bytes] = []
    num_pilots_list: list[int] = []
    split_name_list: list[bytes] = []
    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    channel_taps_list: list[np.ndarray] = []
    channel_delays_list: list[np.ndarray] = []
    channel_doppler_list: list[np.ndarray] = []
    pair_keys: list[list[int]] = []

    for sample_id, (split_name, modulation, snr_db, rep) in enumerate(iterator):
        key = [args.seed, {"train": 1, "val": 2, "test": 3, "eval": 4}[split_name],
               modulations.index(modulation), snr_values.index(snr_db), rep]
        pair_keys.append(key)
        channel_rng = noise_rng = None
        if args.paired_seeds:
            streams = np.random.SeedSequence(key).spawn(4)
            rng = np.random.default_rng(streams[0])
            generator.rng = np.random.default_rng(streams[1])
            channel_rng, noise_rng = (np.random.default_rng(s) for s in streams[2:])
        mu = float(rng.uniform(args.mu_min, args.mu_max))
        fd0_min = float(args.fd0_hz if args.fd0_min_hz is None else args.fd0_min_hz)
        fd0_max = float(args.fd0_hz if args.fd0_max_hz is None else args.fd0_max_hz)
        fd0 = float(rng.uniform(min(fd0_min, fd0_max), max(fd0_min, fd0_max)))
        fractional_timing = float(
            rng.uniform(
                min(args.fractional_timing_min_samples, args.fractional_timing_max_samples),
                max(args.fractional_timing_min_samples, args.fractional_timing_max_samples),
            )
        )
        spec = SignalSpec(
            modulation=modulation,
            num_symbols=int(args.num_symbols),
            samples_per_symbol=int(args.samples_per_symbol),
            sample_rate_hz=float(args.sample_rate_hz),
            carrier_frequency_hz=float(args.carrier_frequency_hz),
            snr_db=snr_db,
            fd0_hz=fd0,
            mu_hz_per_s=mu,
            rrc_beta=float(args.rrc_beta),
            rrc_span=int(args.rrc_span),
            channel_type=str(args.channel_type),
            rician_k_db=float(args.rician_k_db),
            multipath_delays=tuple(args.multipath_delays),
            multipath_gains_db=tuple(args.multipath_gains_db),
            multipath_doppler_hz=tuple(args.multipath_doppler_hz),
            phase_noise_std_rad=float(args.phase_noise_std_rad),
            phase_noise_mode=str(args.phase_noise_mode),
            fractional_timing_offset_samples=fractional_timing,
            compensate_constant_doppler=False,
            target_num_samples=int(args.target_num_samples),
            pilot_enabled=True,
            pilot_pattern=str(args.pilot_pattern),
            pilot_interval_symbols=int(args.pilot_interval_symbols),
            pilot_guard_symbols=int(args.pilot_guard_symbols),
            pilot_modulation=str(args.pilot_modulation),
            pilot_symbol_mode=str(args.pilot_symbol_mode),
            pilot_seed=int(args.pilot_seed),
        )
        sample = generator.generate(spec, channel_rng=channel_rng, noise_rng=noise_rng)

        iq_list.append(sample.iq.astype(np.float32))
        label_list.append(int(sample.label))
        snr_list.append(float(sample.info["snr_db"]))
        fd0_list.append(float(sample.info["fd0_hz"]))
        fractional_timing_list.append(float(sample.info["fractional_timing_offset_samples"]))
        mu_list.append(float(sample.info["mu_hz_per_s"]))
        gamma_list.append(float(sample.info["gamma"]))
        modulation_list.append(sample.modulation.encode("utf-8"))
        domain_list.append(0)
        altitude_list.append(600.0)
        inclination_list.append(53.0)
        fs_list.append(float(sample.info["sample_rate_hz"]))
        fc_list.append(float(sample.info["carrier_frequency_hz"]))
        t_list.append(float(sample.info["observation_time_s"]))
        track_id_list.append(sample_id)
        frame_id_list.append(0)
        frame_time_list.append(0.0)
        track_frame_count_list.append(1)
        track_mu_center_list.append(mu)
        track_mu_edge_list.append(mu)
        track_profile_center_list.append(0.0)
        track_profile_width_list.append(0.0)
        pilot_indices_list.append(sample.pilot_indices.astype(np.int64))
        pilot_sample_indices_list.append((sample.pilot_indices * int(args.samples_per_symbol)).astype(np.int64))
        pilot_symbols_list.append(sample.pilot_symbols.astype(np.complex64))
        pilot_pattern_list.append(str(args.pilot_pattern).encode("utf-8"))
        num_pilots_list.append(int(len(sample.pilot_indices)))
        split_name_list.append(str(split_name).encode("utf-8"))
        channel_taps_list.append(
            np.asarray(sample.info.get("channel_taps", [1.0 + 0.0j]), dtype=np.complex64)
        )
        channel_delays_list.append(
            np.asarray(sample.info.get("channel_delays", [0]), dtype=np.int64)
        )
        channel_doppler_list.append(np.asarray(sample.info["channel_doppler_hz"], dtype=float))
        if split_mode:
            split_indices[str(split_name)].append(sample_id)

    max_channel_length = max(int(x.size) for x in channel_taps_list)
    channel_taps = np.zeros((total, max_channel_length), dtype=np.complex64)
    channel_tap_valid = np.zeros((total, max_channel_length), dtype=bool)
    for row, taps in enumerate(channel_taps_list):
        channel_taps[row, : taps.size] = taps
        channel_tap_valid[row, : taps.size] = np.abs(taps) > 0.0

    arrays = {
        "iq": np.stack(iq_list, axis=0).astype(np.float32),
        "label": np.asarray(label_list, dtype=np.int64),
        "snr_db": np.asarray(snr_list, dtype=np.float32),
        "fd0": np.asarray(fd0_list, dtype=np.float32),
        "fractional_timing_offset_samples": np.asarray(
            [float(x) for x in fractional_timing_list], dtype=np.float32
        ),
        "mu": np.asarray(mu_list, dtype=np.float32),
        "gamma": np.asarray(gamma_list, dtype=np.float32),
        "modulation": np.asarray(modulation_list, dtype="S16"),
        "domain_id": np.asarray(domain_list, dtype=np.int64),
        "orbit_altitude_km": np.asarray(altitude_list, dtype=np.float32),
        "orbit_inclination_deg": np.asarray(inclination_list, dtype=np.float32),
        "fs": np.asarray(fs_list, dtype=np.float32),
        "fc": np.asarray(fc_list, dtype=np.float32),
        "T": np.asarray(t_list, dtype=np.float32),
        "track_id": np.asarray(track_id_list, dtype=np.int64),
        "frame_id": np.asarray(frame_id_list, dtype=np.int64),
        "frame_time_s": np.asarray(frame_time_list, dtype=np.float32),
        "track_frame_count": np.asarray(track_frame_count_list, dtype=np.int64),
        "track_mu_center_hz_per_s": np.asarray(track_mu_center_list, dtype=np.float32),
        "track_mu_edge_hz_per_s": np.asarray(track_mu_edge_list, dtype=np.float32),
        "track_profile_center": np.asarray(track_profile_center_list, dtype=np.float32),
        "track_profile_width": np.asarray(track_profile_width_list, dtype=np.float32),
        "pilot_indices": np.stack(pilot_indices_list, axis=0).astype(np.int64),
        "pilot_sample_indices": np.stack(pilot_sample_indices_list, axis=0).astype(np.int64),
        "pilot_symbols": np.stack(pilot_symbols_list, axis=0).astype(np.complex64),
        "pilot_pattern": np.asarray(pilot_pattern_list, dtype="S16"),
        "num_pilots": np.asarray(num_pilots_list, dtype=np.int64),
        "dataset_split": np.asarray(split_name_list, dtype="S16"),
        "phase_noise_std_rad": np.full(total, float(args.phase_noise_std_rad), dtype=np.float32),
        "channel_type": np.asarray([str(args.channel_type).encode("utf-8")] * total, dtype="S32"),
        "channel_doppler_hz": np.stack(channel_doppler_list),
        "pair_key": np.asarray(pair_keys, dtype=np.int64),
        "channel_taps": channel_taps,
        "channel_tap_valid": channel_tap_valid,
        "channel_delays": np.stack(channel_delays_list, axis=0).astype(np.int64),
        "label_names": np.asarray([m.encode("utf-8") for m in modulations], dtype="S16"),
    }

    summary = {
        "generation_args": vars(args),
        "paired_seeds": bool(args.paired_seeds),
        "snr_convention": "per-frame received clean power / expected noise power",
        "dataset_type": "snr_balanced_split" if split_mode else "snr_balanced_eval",
        "num_samples": int(total),
        "modulations": modulations,
        "snr_db": snr_values,
        "samples_per_mod_snr": int(args.samples_per_mod_snr),
        "split_samples_per_mod_snr": {name: int(count) for name, count in split_plan},
        "iq_shape": list(arrays["iq"].shape),
        "mu_min_hz_per_s": float(np.min(arrays["mu"])),
        "mu_max_hz_per_s": float(np.max(arrays["mu"])),
        "num_pilots": int(np.max(arrays["num_pilots"])),
        "impairments": {
            "fd0_range_hz": [
                float(args.fd0_hz if args.fd0_min_hz is None else args.fd0_min_hz),
                float(args.fd0_hz if args.fd0_max_hz is None else args.fd0_max_hz),
            ],
            "fractional_timing_range_samples": [
                float(args.fractional_timing_min_samples),
                float(args.fractional_timing_max_samples),
            ],
            "channel_type": str(args.channel_type),
            "rician_k_db": float(args.rician_k_db),
            "phase_noise_std_rad": float(args.phase_noise_std_rad),
            "phase_noise_mode": str(args.phase_noise_mode),
        },
        "output_h5": str(output),
        "splits_output": str(splits_output),
    }
    if split_mode:
        summary["split_counts"] = {
            name: int(len(split_indices.get(name, [])))
            for name in ("train", "val", "test")
        }
    payload = dict(arrays)
    payload["meta"] = {
        "summary_json": json.dumps(summary, ensure_ascii=False),
        "dataset_type": summary["dataset_type"],
    }
    save_h5(output, payload)

    if split_mode:
        splits = {
            name: np.asarray(split_indices.get(name, []), dtype=np.int64)
            for name in ("train", "val", "test")
        }
    else:
        all_idx = np.arange(total, dtype=np.int64)
        splits = {
            "train": all_idx,
            "val": np.asarray([], dtype=np.int64),
            "test": all_idx,
        }
    np.savez_compressed(splits_output, **splits)
    save_json(summary, args.summary_output or output.with_suffix(".summary.json"))

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
