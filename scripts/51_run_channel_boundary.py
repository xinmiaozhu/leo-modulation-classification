#!/usr/bin/env python
"""Paired frequency selectivity, path Doppler, and train/test mismatch controls.

All artifacts are isolated under --output-dir. Successful subprocesses have
command-fingerprinted completion markers; interrupted steps can be rerun.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.signal.multipath_equalization import compensated_path_coefficients


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_progress(out, **values):
    path = out / "run_progress.json"
    temporary = path.with_suffix(".tmp")
    payload = {"updated_utc": datetime.now(timezone.utc).isoformat(), "runner_pid": os.getpid(), **values}
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_step(name, script, arguments, out, threads):
    command = [sys.executable, "-c",
               f"import runpy,sys,torch; torch.set_num_threads({threads}); p=sys.argv.pop(1); runpy.run_path(p,run_name='__main__')",
               str(ROOT / "scripts" / script), *map(str, arguments)]
    # Upstream regeneration must invalidate downstream steps, even when the
    # command line has not changed. Avoid rereading multi-GB HDF5 inputs here.
    input_flags = {"--raw-data", "--feature-data", "--symbol-feature-data", "--splits",
                   "--checkpoint", "--model-config", "--train-config", "--config",
                   "--compensated-features", "--mu-feature-data", "--input-iq-feature-data"}
    inputs = {}
    for index, value in enumerate(arguments[:-1]):
        if str(value) in input_flags:
            path = Path(arguments[index + 1]).resolve()
            stat = path.stat()
            inputs[str(path)] = [stat.st_size, stat.st_mtime_ns]
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in ROOT.glob("src/**/*.py")}
    fingerprint = hashlib.sha256(json.dumps([command, inputs, source_hashes], sort_keys=True).encode()
                                 + (ROOT / "scripts" / script).read_bytes()).hexdigest()
    marker = out / "logs" / f"{name}.done.json"
    outputs = []
    for index, value in enumerate(arguments[:-1]):
        if str(value) in {"--output", "--summary-output", "--splits-output"}:
            outputs.append(Path(arguments[index + 1]))
        elif str(value) == "--output-dir":
            outputs.extend(Path(arguments[index + 1]) / file for file in ("best.pt", "train_result.json"))
    if (marker.exists() and json.loads(marker.read_text())["fingerprint"] == fingerprint
            and all(path.exists() for path in outputs)):
        return
    print(f"Running {name}", flush=True)
    log = marker.with_suffix(".log")
    pending = marker.with_suffix(".pending.json")
    executed_command = list(command)
    if script == "13_train_model.py" and "--output-dir" in arguments:
        last = Path(arguments[arguments.index("--output-dir") + 1]) / "last.pt"
        if (last.exists() and pending.exists()
                and json.loads(pending.read_text())["fingerprint"] == fingerprint):
            executed_command += ["--resume", str(last)]
    pending.write_text(json.dumps({"fingerprint": fingerprint, "command": command}, indent=2), encoding="utf-8")
    started = time.monotonic()
    write_progress(out, state="running", step=name, log=str(log), command=executed_command)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(executed_command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env)
    if completed.returncode:
        write_progress(out, state="failed", step=name, log=str(log), returncode=completed.returncode)
        raise RuntimeError(f"{name} failed: {log}\n{log.read_text(encoding='utf-8', errors='replace')[-5000:]}")
    marker.write_text(json.dumps({"fingerprint": fingerprint, "command": command,
                                  "executed_command": executed_command, "elapsed_seconds": time.monotonic()-started,
                                  "input_file_stats": inputs, "source_sha256": source_hashes}, indent=2), encoding="utf-8")
    write_progress(out, state="step_completed", step=name, elapsed_seconds=time.monotonic()-started)


def generation_split_arguments(config, condition):
    if config.get("evaluation_only_nontraining_conditions", False) and condition not in config["training_conditions"]:
        return ["--test-only", "--samples-per-mod-snr", config["test_per_class_snr"]]
    return ["--train-samples-per-mod-snr", config["train_per_class_snr"],
            "--val-samples-per-mod-snr", config["val_per_class_snr"],
            "--test-samples-per-mod-snr", config["test_per_class_snr"]]


def locations(out, condition):
    source_record = out / "prepared_source.json"
    if source_record.exists():
        directory = Path(json.loads(source_record.read_text())["directory"]) / condition
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    else:
        directory = out / condition
    directory.mkdir(exist_ok=True)
    return directory, directory / "raw.h5", directory / "splits.npz"


def prepare(config, out, device):
    for name, profile in config["profiles"].items():
        folder, raw, splits = locations(out, name)
        power = 10 ** (np.asarray(profile["gains_db"]) / 10)
        gains = np.asarray(profile["gains_db"]) - 10 * np.log10(power.sum())
        channel = "time_varying_multipath" if any(profile["doppler_hz"]) else "multipath"
        args = ["--output", raw, "--splits-output", splits, "--snr-db", *config["snr_db"],
                *generation_split_arguments(config, name),
                "--seed", config["data_seed"], "--paired-seeds", "--channel-type", channel,
                "--multipath-delays", *profile["delays"], "--multipath-gains-db", *gains,
                "--multipath-doppler-hz", *profile["doppler_hz"], "--no-progress", "--overwrite"]
        run_step(f"{name}_data", "03_generate_snr_balanced_eval_dataset.py", args, out, config["threads"])
        pilot = folder / "pilot.h5"
        run_step(f"{name}_pilot", "06_precompute_pilot_mu.py", [
            "--raw-data", raw, "--output", pilot, "--estimator-type", "coherent_grid",
            "--pilot-weighting", "coherent", "--device", device, "--no-progress", "--overwrite",
        ], out, config["threads"])
        iq = folder / "unequalized.h5"
        run_step(f"{name}_comp", "09_precompute_external_mu_hoc.py", [
            "--raw-data", raw, "--mu-feature-data", pilot, "--fd0-key", "pilot_fd0_hat",
            "--output", iq, "--save-iq-comp", "--skip-hoc", "--skip-v-hoc", "--no-progress", "--overwrite",
        ], out, config["threads"])
        # Same assumed support for every realizable equalizer, fixed before testing.
        for mode in ("unequalized", "pilot_ls", "oracle_operator"):
            feature = folder / f"{mode}.h5"
            if mode != "unequalized":
                run_step(f"{name}_{mode}", "12_equalize_multipath_frames.py", [
                    "--raw-data", raw, "--compensated-features", iq, "--output", feature,
                    "--mode", mode, "--channel-length", 25, "--overwrite",
                ], out, config["threads"])
            run_step(f"{name}_{mode}_symbol", "10_precompute_symbol_constellation.py", [
                "--raw-data", raw, "--mu-feature-data", pilot, "--input-iq-feature-data", feature,
                "--output", folder / f"{mode}_symbol.h5", "--candidate-score-mode", "exact_mixture",
                "--snr-source", "pilot_evm", "--no-progress", "--overwrite",
            ], out, config["threads"])


def model_arguments(folder, raw, splits, mode):
    return ["--raw-data", raw, "--feature-data", folder / f"{mode}.h5",
            "--symbol-feature-data", folder / f"{mode}_symbol.h5", "--splits", splits,
            "--model", "drc_triplenet", "--model-config", ROOT / "configs/model/triplenet_iq_evm.yaml",
            "--iq-source", "comp", "--iq-representation", "iq", "--iq-normalize", "zscore",
            "--num-workers", 0]


def train(config, out, device):
    for source in config["training_conditions"]:
        folder, raw, splits = locations(out, source)
        for seed in config["model_seeds"]:
            checkpoint = out / "checkpoints" / f"{source}_{seed}"
            args = model_arguments(folder, raw, splits, "unequalized") + [
                "--train-config", ROOT / "configs/train/train_default.yaml", "--output-dir", checkpoint,
                "--seed", seed, "--device", device, "--epochs", config["epochs"],
                "--batch-size", config["batch_size"], "--cache-iq",
            ]
            run_step(f"train_{source}_{seed}", "13_train_model.py", args, out, config["threads"])


def evaluate(config, out, device):
    for source in config["training_conditions"]:
        for seed in config["model_seeds"]:
            checkpoint = out / "checkpoints" / f"{source}_{seed}" / "best.pt"
            for target in config["profiles"]:
                folder, raw, splits = locations(out, target)
                modes = ("unequalized", "pilot_ls", "oracle_operator") if source == "flat" else ("unequalized",)
                for mode in modes:
                    name = f"{source}_{seed}__{target}__{mode}"
                    args = model_arguments(folder, raw, splits, mode) + [
                        "--checkpoint", checkpoint, "--split", "test", "--device", device,
                        "--batch-size", config["batch_size"], "--output", out / "predictions" / f"{name}.csv",
                        "--summary-output", out / "predictions" / f"{name}.summary.json",
                    ]
                    run_step(f"eval_{name}", "14_evaluate_model.py", args, out, config["threads"])


def paired_interval(delta, draws, rng):
    values = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(draws)])
    return (100 * np.quantile(values, [.025, .975])).tolist()


def summarize(config, out):
    rows, diagnostics, comparisons, neural = [], [], [], []
    rng = np.random.default_rng(927)
    reference_keys = reference_labels = None
    exact = {}
    for name, profile in config["profiles"].items():
        folder, raw_path, split_path = locations(out, name)
        split = np.load(split_path)
        train_idx, val_idx, test = (split[k] for k in ("train", "val", "test"))
        if any(np.intersect1d(a, b).size for a, b in ((train_idx, val_idx), (train_idx, test), (val_idx, test))):
            raise ValueError("Overlapping train/validation/test indices")
        with h5py.File(raw_path) as raw, h5py.File(folder / "pilot.h5") as pilot:
            keys, labels, snr = raw["pair_key"][test], raw["label"][test], raw["snr_db"][test]
            if reference_keys is None:
                reference_keys, reference_labels = keys, labels
            np.testing.assert_array_equal(reference_keys, keys)
            np.testing.assert_array_equal(reference_labels, labels)
            power = 10 ** (np.asarray(profile["gains_db"]) / 10)
            power /= power.sum()
            delays = np.asarray(profile["delays"])
            rms_samples = np.sqrt(np.sum(power * (delays - np.sum(power * delays))**2))
            valid = pilot["pilot_valid"][test].astype(bool)
            errors = pilot["pilot_mu_hat"][test] - raw["mu"][test]
            diagnostics.append({"condition": name, "test_n": len(test),
                "rms_delay_symbols": rms_samples / 8, "max_delay_symbols": max(delays) / 8,
                "max_abs_path_doppler_times_frame": max(abs(np.asarray(profile["doppler_hz"]))) * 8192 / 200000,
                "pilot_valid_fraction": valid.mean(),
                "mu_rmse_valid": np.sqrt(np.mean(errors[valid]**2)) if valid.any() else np.nan,
                "mu_abs_error_p95_valid": np.quantile(abs(errors[valid]), .95) if valid.any() else np.nan})
        for mode in ("unequalized", "pilot_ls", "oracle_operator"):
            with h5py.File(folder / f"{mode}_symbol.h5") as symbol:
                pred = np.argmin(symbol["all_candidate_penalized_evm"][test], axis=1)
                correct = pred == labels
                exact[name, mode] = correct
                evm, valid = symbol["pilot_evm"][test], symbol["symbol_valid"][test]
            for value in ["all", *config["snr_db"]]:
                mask = np.ones(len(test), dtype=bool) if value == "all" else snr == value
                rows.append({"condition": name, "receiver": mode, "snr_db": value,
                             "n": int(mask.sum()), "accuracy": float(correct[mask].mean()),
                             "pilot_evm_median": float(np.median(evm[mask])),
                             "symbol_valid_fraction": float(valid[mask].mean())})
                if mode != "unequalized":
                    delta = correct[mask].astype(float) - exact[name, "unequalized"][mask]
                    lo, hi = paired_interval(delta, config["bootstrap_draws"], rng)
                    comparisons.append({"condition": name, "receiver": mode, "snr_db": value,
                                        "gain_pp": 100 * delta.mean(), "ci95_low_pp": lo, "ci95_high_pp": hi})
            if mode == "oracle_operator":
                with h5py.File(folder / f"{mode}.h5") as feature:
                    diagnostics[-1]["oracle_iteration_limit_fraction"] = float(np.mean(feature["oracle_solver_stop"][test] == 7))
            if mode == "pilot_ls":
                errors = []
                with h5py.File(folder / f"{mode}.h5") as feature, h5py.File(raw_path) as raw:
                    for index in test:
                        delays = raw["channel_delays"][index]
                        c = compensated_path_coefficients(raw["channel_taps"][index], delays,
                            raw["channel_doppler_hz"][index], raw["iq"].shape[-1], float(raw["fs"][index]),
                            float(feature["mu_hat"][index]), 0.)
                        estimated = feature["equalizer_taps"][index]
                        energy_true = np.sum(abs(c)**2)
                        energy_est = c.shape[1] * np.sum(abs(estimated)**2)
                        cross = sum(np.conj(estimated[d]) * path.sum() for d, path in zip(delays, c) if d < len(estimated))
                        errors.append(max(0., 1 - abs(cross)**2 / max(energy_true*energy_est, 1e-30)))
                diagnostics[-1]["pilot_ls_gain_aligned_trajectory_nmse_median"] = float(np.median(errors))
    pd.DataFrame(rows).to_csv(out / "exact_mixture.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "paired_equalization_gains.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / "channel_diagnostics.csv", index=False)
    model_correct = {}
    for path in sorted((out / "predictions").glob("*.csv")):
        # The evaluator also emits auxiliary grouped CSVs; only exact task filenames count.
        pieces = path.stem.split("__")
        if len(pieces) != 3 or pieces[2] not in ("unequalized", "pilot_ls", "oracle_operator"):
            continue
        source_seed, target, mode = pieces
        source, seed = source_seed.rsplit("_", 1)
        table = pd.read_csv(path)
        np.testing.assert_array_equal(table["label"], reference_labels)
        model_correct[source, seed, target, mode] = table["correct"].to_numpy(float)
        for value in ["all", *config["snr_db"]]:
            subset = table if value == "all" else table.loc[table["snr_db"] == value]
            neural.append({"train_condition": source, "seed": int(seed), "test_condition": target,
                           "receiver": mode, "snr_db": value, "n": len(subset),
                           "accuracy": subset["correct"].mean()})
    pd.DataFrame(neural).to_csv(out / "neural_accuracy.csv", index=False)
    if neural:
        seed_summary = pd.DataFrame(neural).groupby(
            ["train_condition", "test_condition", "receiver", "snr_db"], sort=False,
        ).accuracy.agg(["mean", "std", "count"]).reset_index()
        seed_summary.rename(columns={"count": "model_seed_count"}).to_csv(out / "neural_seed_summary.csv", index=False)
    mismatch = []
    for (source, seed, target, mode), correct in model_correct.items():
        if source == "flat":
            continue
        base = model_correct["flat", seed, target, mode]
        delta = correct-base
        lo, hi = paired_interval(delta, config["bootstrap_draws"], rng)
        mismatch.append({"train_condition": source, "seed": int(seed), "test_condition": target,
                         "gain_over_flat_training_pp": 100 * delta.mean(),
                         "ci95_low_pp": lo, "ci95_high_pp": hi})
    pd.DataFrame(mismatch).to_csv(out / "paired_training_gains.csv", index=False)
    training = []
    for source in config["training_conditions"]:
        for seed in config["model_seeds"]:
            path = out / "checkpoints" / f"{source}_{seed}" / "history.csv"
            if path.exists():
                history = pd.read_csv(path)
                best = history.loc[history.val_acc.idxmax()]
                training.append({"train_condition": source, "seed": seed, "epochs_run": len(history),
                                 "best_validation_epoch": int(best.epoch), "best_validation_accuracy": best.val_acc})
    pd.DataFrame(training).to_csv(out / "training_summary.csv", index=False)
    plot_results(rows, neural, out)
    expected = len(config["model_seeds"]) * len(config["profiles"]) * (3 + len(config["training_conditions"])-1)
    (out / "status.json").write_text(json.dumps({
        "description": config["description"], "paired_test_keys_verified": True,
        "neural_evaluations": len(model_correct), "expected_neural_evaluations": expected,
        "complete": len(model_correct) == expected and all(
            item["oracle_iteration_limit_fraction"] == 0 for item in diagnostics),
        "test_frames_per_condition": len(reference_labels), "config": config,
    }, indent=2), encoding="utf-8")
    report = ["# Channel boundary results", "", config["description"], "",
              f"Test frames per condition: {len(reference_labels)}; SNR points: {config['snr_db']}.",
              f"Training: {config['model_seeds']} seeds, at most {config['epochs']} epochs; validation-only checkpoint selection.", "",
              "## Exact-mixture accuracy (%)", "",
              "| Channel | Unequalized | Pilot-LS | Oracle operator |", "|---|---:|---:|---:|"]
    for condition in config["profiles"]:
        values = [100 * exact[condition, mode].mean() for mode in ("unequalized", "pilot_ls", "oracle_operator")]
        report.append(f"| {condition} | " + " | ".join(f"{value:.2f}" for value in values) + " |")
    report += ["", "## Neural train/test mismatch (%, unequalized)", ""]
    if neural:
        table = pd.DataFrame(neural)
        table = table[(table.snr_db == "all") & (table.receiver == "unequalized")]
        conditions = list(config["profiles"])
        report += ["| Train / Test | " + " | ".join(conditions) + " |", "|---|" + "---:|" * len(conditions)]
        for source in config["training_conditions"]:
            cells = []
            for target in conditions:
                values = 100 * table[(table.train_condition == source) & (table.test_condition == target)].accuracy
                cells.append(f"{values.mean():.2f}" + (f" ± {values.std():.2f}" if len(values) > 1 else ""))
            report.append(f"| {source} | " + " | ".join(cells) + " |")
    report += ["", "## Interpretation limits", "",
               "Results apply to the recorded sample sizes and training budget; convergence must be assessed from training histories. Neural ± values, if present, are standard deviations across model seeds.",
               "The time-varying channel has deterministic per-path Doppler, not a full stochastic fading spectrum.",
               "Pilot-LS uses fixed 25-sample support and no within-frame tracking. Oracle uses true channel trajectories but retains estimated-carrier error.",
               "SNR is normalized by each received clean frame's power. Bootstrap intervals are conditional on trained checkpoints.",
               "See paired_equalization_gains.csv, paired_training_gains.csv and channel_diagnostics.csv for intervals and failure diagnostics.", ""]
    (out / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")


def plot_results(rows, neural, out):
    # Keep the runner and standalone paper export on the same single-panel style.
    from src.plotting.channel_boundary import plot_channel_boundary
    plot_channel_boundary(pd.DataFrame(neural), out / "channel_boundary.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/channel_boundary_screen.json")
    parser.add_argument("--output-dir", default="outputs/channel_boundary_screen")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-prepared-from", help="Read immutable prepared data from an existing run; use train/eval/summary stages only.")
    parser.add_argument("--stages", nargs="+", choices=["prepare", "train", "eval", "summary"],
                        default=["prepare", "train", "eval", "summary"])
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for directory in ("logs", "predictions"):
        (out / directory).mkdir(exist_ok=True)
    source_record = out / "prepared_source.json"
    if args.reuse_prepared_from:
        source = Path(args.reuse_prepared_from).resolve()
        original = json.loads((source / "manifest.json").read_text())["config"]
        data_keys = ("snr_db", "train_per_class_snr", "val_per_class_snr", "test_per_class_snr", "data_seed", "profiles")
        if (any(original[key] != config[key] for key in data_keys)
                or original.get("evaluation_only_nontraining_conditions", False) != config.get("evaluation_only_nontraining_conditions", False)
                or original["training_conditions"] != config["training_conditions"]):
            raise ValueError("Reused prepared data must have exactly the requested generation settings.")
        payload = {"directory": str(source), "manifest_sha256": digest(source / "manifest.json")}
        if source_record.exists() and json.loads(source_record.read_text()) != payload:
            raise ValueError("Cannot change the source of an existing prepared-data run.")
        source_record.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if source_record.exists() and "prepare" in args.stages:
        raise ValueError("Reused prepared data are read-only; select train/eval/summary stages.")
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["config"] != config:
        raise ValueError("Use a new output directory for changed experimental settings.")
    if not manifest.exists():
        files = [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"), ROOT / "configs/model/triplenet_iq_evm.yaml",
                 ROOT / "configs/train/train_default.yaml"]
        manifest.write_text(json.dumps({"config": config, "device": args.device,
            "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in files},
            "carrier": "coherent_grid; common CFO=0, mu uniform [-8160,-180] Hz/s",
            "channel_order": "common carrier then sparse multipath then received-power AWGN",
            "oracle": "finite-frame true time-varying channel; estimated carrier retained",
            "noise_pairing": "same unit complex Gaussian draws, scaled to received-power SNR",
        }, indent=2), encoding="utf-8")
    with (out / "invocations.jsonl").open("a", encoding="utf-8") as history:
        history.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
            "stages": args.stages, "device": args.device, "runner_sha256": digest(__file__)}) + "\n")
    write_progress(out, state="starting", stages=args.stages)
    try:
        if "prepare" in args.stages:
            prepare(config, out, args.device)
        if "train" in args.stages:
            train(config, out, args.device)
        if "eval" in args.stages:
            evaluate(config, out, args.device)
        if "summary" in args.stages:
            summarize(config, out)
    except Exception as exc:
        previous = json.loads((out / "run_progress.json").read_text())
        previous.pop("updated_utc", None)
        previous.pop("runner_pid", None)
        previous.update(state="failed", error=str(exc))
        write_progress(out, **previous)
        raise
    write_progress(out, state="requested_stages_completed", stages=args.stages)
    print(f"Finished requested stages: {out}", flush=True)


if __name__ == "__main__":
    main()
