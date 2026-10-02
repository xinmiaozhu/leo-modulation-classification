#!/usr/bin/env python
"""Benchmark network-only and end-to-end complexity on real test frames.

The benchmark uses one CPU thread for comparable model latency. CUDA latency
is also reported for neural models, but the primary Pareto latency is the
fully comparable CPU end-to-end value. The end-to-end path contains the
pilot-aided Doppler-rate estimate, quadratic-phase compensation, each
method's input construction, and classifier inference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import pandas as pd
import torch

from src.datasets.feature_dataset import _preprocess_iq
from src.drc_hoc.compensation import compensate_full_doppler
from src.drc_hoc.paper_features import nasa_feature_vector
from src.drc_hoc.pilot_estimator import PilotMuEstimator, PilotMuEstimatorConfig
from src.models.drc_triplenet import DRCTripleNet
from src.models.paper_baselines import build_paper_baseline
from src.signal.modulation import get_constellation
from src.signal.pulse_shape import rrc_filter
from src.utils.config import load_config
from src.utils.math_utils import complex_to_iq, to_complex


METHODS = {
    "c2_cnn2": ("C1", "C1 CNN2", "paper_cnn2", "configs/model/paper_cnn2.yaml"),
    "c3_mcnet": ("C2", "C2 MCNet", "paper_mcnet", "configs/model/paper_mcnet.yaml"),
    "c4_cnn_lstm_dual": (
        "C3",
        "C3 CNN-LSTM",
        "paper_cnn_lstm_dual",
        "configs/model/paper_cnn_lstm_dual.yaml",
    ),
    "c5_satellite_cnn": (
        "C4",
        "C4 IQCNet",
        "paper_satellite_cnn",
        "configs/model/paper_satellite_cnn.yaml",
    ),
    "c6_nasa_hoc_nn": (
        "C5",
        "C5 NASA HOC-NN",
        "paper_nasa_hoc_nn",
        "configs/model/paper_nasa_hoc_nn.yaml",
    ),
    "c7_starnet": (
        "C6",
        "C6 STARNet",
        "paper_starnet",
        "configs/model/paper_starnet.yaml",
    ),
    "proposed": ("Proposed", "Proposed I/Q--constellation", "drc_triplenet", None),
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark the paper comparison models.")
    p.add_argument("--raw-data", required=True)
    p.add_argument("--feature-data", required=True)
    p.add_argument("--paper-feature-data", required=True)
    p.add_argument("--symbol-feature-data", required=True)
    p.add_argument("--summary", required=True, help="Independent-test accuracy summary CSV.")
    p.add_argument("--checkpoint-root", default="outputs/checkpoints/paper_baselines_faithful")
    p.add_argument(
        "--proposed-checkpoint",
        default="outputs/checkpoints/exact_mixture/protocol_a/proposed/seed_41/best.pt",
    )
    p.add_argument(
        "--starnet-checkpoint",
        default="outputs/checkpoints/starnet_seeds/protocol_a/seed_41/best.pt",
    )
    p.add_argument(
        "--mcnet-checkpoint",
        default=None,
        help="Optional formal MCNet checkpoint; otherwise use --checkpoint-root.",
    )
    p.add_argument(
        "--accuracy-override",
        nargs="*",
        default=(),
        metavar="METHOD=ACCURACY",
        help="Override summary accuracies, e.g. proposed=0.90948 c7_starnet=0.82519.",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--frontend-samples", type=int, default=77)
    p.add_argument("--cpu-warmup", type=int, default=5)
    p.add_argument("--cpu-repeats", type=int, default=30)
    p.add_argument("--cuda-warmup", type=int, default=20)
    p.add_argument("--cuda-repeats", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--candidate-score-mode",
        choices=["exact_mixture", "logistic"],
        default="exact_mixture",
        help="Candidate score used when timing on-the-fly I/Q+descriptor construction.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _load_script(name: str, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / "scripts" / name)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import script module: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config_dict(path: str) -> dict[str, Any]:
    cfg = load_config(path)
    values = cfg.model if "model" in cfg else cfg
    return values.to_dict() if hasattr(values, "to_dict") else dict(values)


def _state_bytes(state: dict[str, torch.Tensor]) -> int:
    return int(
        sum(value.numel() * value.element_size() for value in state.values() if torch.is_tensor(value))
    )


def _checkpoint_model(
    method: str,
    checkpoint_root: Path,
    proposed_checkpoint: Path,
    starnet_checkpoint: Path,
    mcnet_checkpoint: Path | None,
) -> tuple[Any, int, int]:
    _, _, model_name, config_path = METHODS[method]
    if method == "proposed":
        checkpoint_path = proposed_checkpoint
    elif method == "c7_starnet":
        checkpoint_path = starnet_checkpoint
    elif method == "c3_mcnet" and mcnet_checkpoint is not None:
        checkpoint_path = mcnet_checkpoint
    else:
        checkpoint_path = checkpoint_root / method / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    if method == "proposed":
        run_config = dict(checkpoint.get("config", {}))
        cfg = dict(run_config.get("model_config", {}))
        cfg.setdefault("hoc_dim", 21)
        cfg.setdefault("evm_dim", int(run_config.get("evm_dim", 48)))
        cfg.setdefault("num_classes", 7)
        model = DRCTripleNet(**cfg)
    else:
        cfg = _config_dict(str(config_path))
        feature_dim = int(cfg.pop("feature_dim", 128))
        dropout = float(cfg.pop("dropout", 0.2))
        cfg.setdefault("feature_dim", feature_dim)
        cfg.setdefault("dropout", dropout)
        model = build_paper_baseline(
            str(model_name),
            num_classes=7,
            hoc_dim=10 if method == "c6_nasa_hoc_nn" else None,
            **cfg,
        )
    model.load_state_dict(state, strict=True)
    model.eval()
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    return model, parameter_count, _state_bytes(state)


def _checkpoint_config(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return dict(checkpoint.get("config", {}))


def _forward_callable(
    method: str,
    model: Any,
    iq: np.ndarray | None,
    hoc: np.ndarray | None,
    evm: np.ndarray | None,
    device: torch.device,
) -> Callable[[], Any]:
    model = model.to(device).eval()
    iq_t = torch.from_numpy(np.asarray(iq, dtype=np.float32)[None]).to(device) if iq is not None else None
    hoc_t = (
        torch.from_numpy(np.asarray(hoc, dtype=np.float32)[None]).to(device)
        if hoc is not None
        else None
    )
    evm_t = (
        torch.from_numpy(np.asarray(evm, dtype=np.float32)[None]).to(device)
        if evm is not None
        else None
    )

    if method == "c6_nasa_hoc_nn":
        return lambda: model(None, hoc_t, None)
    if method == "proposed":
        return lambda: model(iq_t, None, None, evm_features=evm_t)
    return lambda: model(iq_t)


def _time_cpu(fn: Callable[[], Any], warmup: int, repeats: int) -> np.ndarray:
    for _ in range(max(warmup, 0)):
        fn()
    values = np.empty(max(repeats, 1), dtype=np.float64)
    with torch.inference_mode():
        for i in range(len(values)):
            start = time.perf_counter_ns()
            fn()
            values[i] = (time.perf_counter_ns() - start) / 1.0e6
    return values


def _time_cuda(fn: Callable[[], Any], warmup: int, repeats: int) -> np.ndarray:
    for _ in range(max(warmup, 0)):
        fn()
    torch.cuda.synchronize()
    values = np.empty(max(repeats, 1), dtype=np.float64)
    with torch.inference_mode():
        for i in range(len(values)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            values[i] = float(start.elapsed_time(end))
    return values


def _percentiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_median_ms": float(np.median(values)),
        f"{prefix}_mean_ms": float(np.mean(values)),
        f"{prefix}_std_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        f"{prefix}_p90_ms": float(np.quantile(values, 0.90)),
        f"{prefix}_p95_ms": float(np.quantile(values, 0.95)),
        f"{prefix}_p99_ms": float(np.quantile(values, 0.99)),
        f"{prefix}_max_ms": float(np.max(values)),
    }


def _estimate_neural_macs(model: torch.nn.Module, fn: Callable[[], Any]) -> int:
    """Estimate batch-one multiply-accumulate operations with forward hooks."""

    total = 0
    handles: list[Any] = []

    def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal total
        out = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(out):
            return
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
            kernel = int(np.prod(module.kernel_size))
            per_output = int(module.in_channels // module.groups) * kernel
            total += int(out.numel()) * per_output
        elif isinstance(module, torch.nn.Linear):
            total += int(out.numel()) * int(module.in_features)
        elif isinstance(module, (torch.nn.LSTM, torch.nn.GRU)):
            x = inputs[0]
            if not torch.is_tensor(x) or x.ndim < 3:
                return
            batch = int(x.shape[0] if module.batch_first else x.shape[1])
            steps = int(x.shape[1] if module.batch_first else x.shape[0])
            directions = 2 if module.bidirectional else 1
            hidden = int(module.hidden_size)
            input_size = int(module.input_size)
            gates = 4 if isinstance(module, torch.nn.LSTM) else 3
            for layer in range(int(module.num_layers)):
                layer_input = input_size if layer == 0 else hidden * directions
                total += (
                    batch
                    * steps
                    * directions
                    * gates
                    * hidden
                    * (layer_input + hidden + 1)
                )

    for module in model.modules():
        if isinstance(
            module,
            (
                torch.nn.Conv1d,
                torch.nn.Conv2d,
                torch.nn.Linear,
                torch.nn.LSTM,
                torch.nn.GRU,
            ),
        ):
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            fn()
    finally:
        for handle in handles:
            handle.remove()
    return int(total)


def _cuda_peak_increment(fn: Callable[[], Any], device: torch.device) -> int:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = int(torch.cuda.memory_allocated(device))
    with torch.inference_mode():
        fn()
    torch.cuda.synchronize(device)
    return max(0, int(torch.cuda.max_memory_allocated(device)) - baseline)


def _standardize(values: np.ndarray, stats: dict[str, Any] | None) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if not stats:
        return x
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return np.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


def _stratified_frontend_indices(
    raw: h5py.File,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = np.asarray(raw["label"][:], dtype=np.int64)
    snr_db = np.asarray(raw["snr_db"][:], dtype=np.float64)
    groups = [
        np.flatnonzero((labels == label) & (snr_db == snr))
        for snr in np.unique(snr_db)
        for label in np.unique(labels)
    ]
    groups = [group for group in groups if len(group)]
    target = min(max(int(sample_count), 1), len(labels))
    if target < len(groups):
        positions = np.linspace(0, len(groups) - 1, target, dtype=int)
        groups = [groups[position] for position in positions]

    chosen = [int(rng.choice(group)) for group in groups]
    if len(chosen) < target:
        remaining = np.setdiff1d(
            np.arange(len(labels), dtype=np.int64),
            np.asarray(chosen, dtype=np.int64),
            assume_unique=False,
        )
        chosen.extend(
            int(value)
            for value in rng.choice(remaining, size=target - len(chosen), replace=False)
        )
    return np.sort(np.asarray(chosen[:target], dtype=np.int64))


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        print(f"Skip existing: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    checkpoint_root = Path(args.checkpoint_root)
    proposed_checkpoint = Path(args.proposed_checkpoint)
    starnet_checkpoint = Path(args.starnet_checkpoint)
    mcnet_checkpoint = Path(args.mcnet_checkpoint) if args.mcnet_checkpoint else None
    summary = pd.read_csv(args.summary)
    accuracy = dict(zip(summary["method"], summary["accuracy"], strict=True))
    for item in args.accuracy_override:
        if "=" not in item:
            raise ValueError(f"Invalid --accuracy-override {item!r}; expected METHOD=ACCURACY")
        method, value = item.split("=", 1)
        if method not in METHODS:
            raise ValueError(f"Unknown accuracy-override method: {method}")
        accuracy[method] = float(value)
    missing_accuracy = sorted(set(METHODS) - set(accuracy))
    if missing_accuracy:
        raise ValueError(
            "Missing accuracy for methods " + ", ".join(missing_accuracy)
            + "; supply --accuracy-override."
        )

    sym = _load_script("10_precompute_symbol_constellation.py", "complexity_symbol")
    modulations = ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "16APSK", "32APSK")
    constellations = {
        name: get_constellation(name).astype(np.complex128) for name in modulations
    }
    phase_grid = np.asarray([0.0], dtype=np.float64)
    taps = rrc_filter(beta=0.35, span=8, sps=8)
    estimator = PilotMuEstimator(
        PilotMuEstimatorConfig(
            sample_rate_hz=200000.0,
            samples_per_symbol=8,
            rrc_beta=0.35,
            rrc_span=8,
            timing_phases=(0,),
            mu_min=-8160.0,
            mu_max=-180.0,
            coarse_step_hz_per_s=40.0,
            fine_radius_hz_per_s=120.0,
            fine_step_hz_per_s=5.0,
            fd0_min_hz=0.0,
            fd0_max_hz=0.0,
            fd0_step_hz=100.0,
            pilot_weighting="coherent",
            min_pilots=8,
        )
    )
    proposed_cfg = _checkpoint_config(proposed_checkpoint)
    evm_stats = proposed_cfg.get("evm_stats")
    evm_feature_indices = np.asarray(
        proposed_cfg.get("evm_feature_indices", np.arange(48)), dtype=np.int64
    )

    with h5py.File(args.raw_data, "r") as raw, h5py.File(
        args.feature_data, "r"
    ) as feat, h5py.File(args.paper_feature_data, "r") as paper, h5py.File(
        args.symbol_feature_data, "r"
    ) as symbol:
        chosen = _stratified_frontend_indices(raw, args.frontend_samples, rng)
        sample_index = int(chosen[len(chosen) // 2])

        precomputed: dict[str, tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]] = {
            "c2_cnn2": (
                _preprocess_iq(feat["iq_comp"][sample_index], normalize="zscore"),
                None,
                None,
            ),
            "c3_mcnet": (
                _preprocess_iq(feat["iq_comp"][sample_index], normalize="zscore"),
                None,
                None,
            ),
            "c4_cnn_lstm_dual": (
                _preprocess_iq(feat["iq_comp"][sample_index], normalize="power"),
                None,
                None,
            ),
            "c5_satellite_cnn": (
                _preprocess_iq(feat["iq_comp"][sample_index], normalize="zscore"),
                None,
                None,
            ),
            "c6_nasa_hoc_nn": (None, np.asarray(paper["h_drc"][sample_index]), None),
            "c7_starnet": (
                _preprocess_iq(feat["iq_comp"][sample_index], normalize="zscore"),
                None,
                None,
            ),
            "proposed": (
                _preprocess_iq(feat["iq_comp"][sample_index], normalize="zscore"),
                None,
                _standardize(
                    np.asarray(symbol["evm_features"][sample_index])[evm_feature_indices],
                    evm_stats,
                ),
            ),
        }

        common_times: list[float] = []
        method_feature_times: dict[str, list[float]] = {key: [] for key in METHODS}
        for index in chosen:
            raw_iq = np.asarray(raw["iq"][index], dtype=np.float32)
            start = time.perf_counter_ns()
            estimate = estimator.estimate(
                raw_iq,
                np.asarray(raw["pilot_indices"][index]),
                np.asarray(raw["pilot_symbols"][index]),
            )
            compensated = compensate_full_doppler(
                to_complex(raw_iq),
                sample_rate_hz=200000.0,
                fd0_hat_hz=estimate.fd0_hat_hz,
                mu_hat_hz_per_s=estimate.mu_hat_hz_per_s,
                centered=False,
            )
            common_times.append((time.perf_counter_ns() - start) / 1.0e6)
            iq_comp = complex_to_iq(compensated).astype(np.float32)

            for method in METHODS:
                start = time.perf_counter_ns()
                if method in {"c2_cnn2", "c3_mcnet", "c5_satellite_cnn", "c7_starnet"}:
                    _preprocess_iq(iq_comp, normalize="zscore")
                elif method == "c4_cnn_lstm_dual":
                    _preprocess_iq(iq_comp, normalize="power")
                else:
                    matched = np.convolve(compensated, taps, mode="same")
                    gain, pilot_evm_in_sample, _ = sym._estimate_gain_and_pilot_evm(
                        matched,
                        np.asarray(raw["pilot_indices"][index]),
                        np.asarray(raw["pilot_symbols"][index]),
                        timing=0,
                        sps=8,
                    )
                    pilot_evm_crossfit, _ = sym._estimate_crossfit_pilot_evm(
                        matched,
                        np.asarray(raw["pilot_indices"][index]),
                        np.asarray(raw["pilot_symbols"][index]),
                        timing=0,
                        sps=8,
                    )
                    pilot_evm = (
                        pilot_evm_crossfit
                        if np.isfinite(pilot_evm_crossfit)
                        else pilot_evm_in_sample
                    )
                    pilot_snr_db = sym._pilot_sample_snr_db(
                        pilot_evm,
                        samples_per_symbol=8,
                        offset_db=None,
                    )
                    symbols = sym._extract_equalized_symbols(
                        matched,
                        gain,
                        0,
                        8,
                        np.asarray(raw["pilot_indices"][index]),
                        symbol_trim=8,
                        max_symbols=512,
                    )
                    symbols = sym._prepare_symbols(
                        symbols, center=True, power_normalize=True
                    )
                    if method == "c6_nasa_hoc_nn":
                        nasa_feature_vector(symbols, pilot_evm)
                    else:
                        evm_raw, phase_best = sym._score_evm_all_candidates(
                            symbols, modulations, constellations, phase_grid
                        )
                        evm_features, _ = sym._evm_feature_vector(
                            evm_raw,
                            modulations,
                            pilot_evm,
                            symbols,
                            constellations,
                            phase_best
                            if phase_best is not None
                            else [0.0] * len(modulations),
                            snr_db=pilot_snr_db,
                            complexity_penalty_lambda=0.01,
                            complexity_penalty_snr_threshold=-3.0,
                            complexity_penalty_snr_scale=4.0,
                            apsk_ring_penalty_lambda=0.012,
                            apsk_phase_weight=1.0,
                            apsk_occupancy_weight=0.25,
                            candidate_score_mode=args.candidate_score_mode,
                        )
                        _standardize(
                            np.asarray(evm_features)[evm_feature_indices], evm_stats
                        )
                        _preprocess_iq(iq_comp, normalize="zscore")
                method_feature_times[method].append(
                    (time.perf_counter_ns() - start) / 1.0e6
                )

    rows: list[dict[str, Any]] = []
    common = np.asarray(common_times, dtype=np.float64)
    for method, (method_id, display_label, _, _) in METHODS.items():
        model, parameter_count, model_bytes = _checkpoint_model(
            method,
            checkpoint_root,
            proposed_checkpoint,
            starnet_checkpoint,
            mcnet_checkpoint,
        )
        iq, hoc, evm = precomputed[method]
        cpu_fn = _forward_callable(method, model, iq, hoc, evm, torch.device("cpu"))
        cpu_values = _time_cpu(cpu_fn, args.cpu_warmup, args.cpu_repeats)
        macs = float(_estimate_neural_macs(model, cpu_fn))

        cuda_values = np.asarray([np.nan], dtype=np.float64)
        cuda_peak_increment = np.nan
        if args.device.startswith("cuda") and torch.cuda.is_available():
            cuda_device = torch.device(args.device)
            cuda_fn = _forward_callable(method, model, iq, hoc, evm, cuda_device)
            cuda_peak_increment = float(_cuda_peak_increment(cuda_fn, cuda_device))
            cuda_values = _time_cuda(cuda_fn, args.cuda_warmup, args.cuda_repeats)
            model.to("cpu")
            torch.cuda.empty_cache()

        feature_values = np.asarray(method_feature_times[method], dtype=np.float64)
        end_to_end = common + feature_values + float(np.median(cpu_values))
        row: dict[str, Any] = {
            "method_id": method_id,
            "method": method,
            "display_label": display_label,
            "accuracy": float(accuracy[method]),
            "accuracy_percent": 100.0 * float(accuracy[method]),
            "parameter_count": parameter_count,
            "parameter_count_m": parameter_count / 1.0e6,
            "inference_model_bytes": model_bytes,
            "inference_model_mib": model_bytes / (1024.0**2),
            "estimated_macs": macs,
            "estimated_mmacs": macs / 1.0e6 if np.isfinite(macs) else np.nan,
            "estimated_flops": 2.0 * macs if np.isfinite(macs) else np.nan,
            "cuda_peak_increment_bytes": cuda_peak_increment,
            "cuda_peak_increment_mib": (
                cuda_peak_increment / (1024.0**2)
                if np.isfinite(cuda_peak_increment)
                else np.nan
            ),
            "frontend_sample_count": int(len(common)),
        }
        row.update(_percentiles(cpu_values, "network_cpu"))
        if np.all(np.isfinite(cuda_values)):
            row.update(_percentiles(cuda_values, "network_cuda"))
        else:
            row.update(
                {
                    "network_cuda_median_ms": np.nan,
                    "network_cuda_mean_ms": np.nan,
                    "network_cuda_std_ms": np.nan,
                    "network_cuda_p90_ms": np.nan,
                    "network_cuda_p95_ms": np.nan,
                    "network_cuda_p99_ms": np.nan,
                    "network_cuda_max_ms": np.nan,
                }
            )
        row.update(_percentiles(common, "common_frontend_cpu"))
        row.update(_percentiles(feature_values, "method_feature_cpu"))
        row.update(_percentiles(end_to_end, "end_to_end_cpu"))
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    metadata = {
        "output": str(output),
        "raw_data": args.raw_data,
        "summary": args.summary,
        "proposed_checkpoint": str(proposed_checkpoint),
        "starnet_checkpoint": str(starnet_checkpoint),
        "mcnet_checkpoint": str(mcnet_checkpoint) if mcnet_checkpoint is not None else None,
        "accuracy_overrides": list(args.accuracy_override),
        "cpu_threads": 1,
        "device": args.device,
        "frontend_sample_count": int(len(chosen)),
        "frontend_sampling": "stratified by exact (snr_db, label) groups",
        "cpu_warmup": int(args.cpu_warmup),
        "cpu_repeats": int(args.cpu_repeats),
        "cuda_warmup": int(args.cuda_warmup),
        "cuda_repeats": int(args.cuda_repeats),
        "seed": int(args.seed),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(torch.device(args.device))
            if args.device.startswith("cuda") and torch.cuda.is_available()
            else None
        ),
        "network_input_dtype": "float32",
        "frontend_numeric_dtype": "mixed float32/float64/complex128",
        "batch_size": 1,
        "latency_policy": (
            "Primary end-to-end latency is single-thread CPU and includes pilot "
            "mu estimation, compensation, method-specific input construction, "
            "and batch-one classifier inference. CUDA network-only latency is "
            "reported separately for neural models."
        ),
        "frontend_indices": chosen.tolist(),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        result[
            [
                "method_id",
                "accuracy_percent",
                "parameter_count",
                "network_cpu_median_ms",
                "network_cuda_median_ms",
                "end_to_end_cpu_median_ms",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
