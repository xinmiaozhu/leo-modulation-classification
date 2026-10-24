#!/usr/bin/env python
"""Assemble the Protocol-B front-end attribution table from frozen artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/processed/leo_7mods_joint_practical.h5"
SPLITS = ROOT / "data/splits/leo_7mods_joint_practical_splits.npz"
RESULT_ROOT = ROOT / "outputs/results/frontend_attribution"

ESTIMATORS = {
    "No compensation": (None, "", ""),
    "Coherent, equal budget": (
        ROOT / "data/features/frontend_attribution/protocol_b_pilot_coherent_equal_budget.h5",
        "pilot_mu_hat",
        "pilot_fd0_hat",
    ),
    "Coherent coarse-to-fine": (
        ROOT / "data/features/leo_7mods_joint_practical_pilot_joint.h5",
        "pilot_mu_hat",
        "pilot_fd0_hat",
    ),
    "DFRFT-equivalent acquisition": (
        ROOT / "data/features/frontend_attribution/protocol_b_pilot_dfrft_only.h5",
        "pilot_mu_hat",
        "pilot_fd0_hat",
    ),
    "DFRFT + coherent refinement": (
        ROOT / "data/features/hybrid_dfrft/protocol_b_pilot_hybrid.h5",
        "pilot_mu_hat",
        "pilot_fd0_hat",
    ),
    "Oracle compensation": (
        ROOT / "data/features/joint_practical_gamma_oracle_mu.h5",
        "mu_hat",
        "fd0_hat",
    ),
}

FROZEN_SUMMARIES = {
    "No compensation": "no_comp_seed41.summary.json",
    "Coherent, equal budget": "coherent_equal_budget_seed41.summary.json",
    "Coherent coarse-to-fine": "coherent_grid_seed41.summary.json",
    "DFRFT-equivalent acquisition": "dfrft_only_seed41.summary.json",
    "DFRFT + coherent refinement": None,
    "Oracle compensation": "oracle_seed41.summary.json",
}

CLASSIFIER_INPUTS = {
    "No compensation": (
        "data/features/frontend_attribution/protocol_b_no_comp_iqcomp.h5",
        "data/features/frontend_attribution/protocol_b_no_comp_symbol_exact.h5",
    ),
    "Coherent, equal budget": (
        "data/features/frontend_attribution/protocol_b_coherent_equal_budget_iqcomp.h5",
        "data/features/frontend_attribution/protocol_b_coherent_equal_budget_symbol_exact.h5",
    ),
    "Coherent coarse-to-fine": (
        "data/features/leo_7mods_joint_practical_iqcomp.h5",
        "data/features/exact_mixture/protocol_b_symbol_exact.h5",
    ),
    "DFRFT-equivalent acquisition": (
        "data/features/frontend_attribution/protocol_b_dfrft_only_iqcomp.h5",
        "data/features/frontend_attribution/protocol_b_dfrft_only_symbol_exact.h5",
    ),
    "DFRFT + coherent refinement": (
        "data/features/hybrid_dfrft/protocol_b_iqcomp.h5",
        "data/features/hybrid_dfrft/protocol_b_symbol_exact.h5",
    ),
    "Oracle compensation": (
        "data/features/frontend_attribution/protocol_b_oracle_iqcomp.h5",
        "data/features/frontend_attribution/protocol_b_oracle_symbol_exact.h5",
    ),
}

LATENCY_NAMES = {
    "Coherent, equal budget": "coherent_equal_budget",
    "Coherent coarse-to-fine": "coherent_current",
    "DFRFT-equivalent acquisition": "dfrft_only",
    "DFRFT + coherent refinement": "hybrid_dfrft_coherent",
}

VALIDATION_LATENCY_NAMES = {
    "Coherent, equal budget": "coherent_80x25",
    "Coherent coarse-to-fine": "coherent_40x25",
    "DFRFT-equivalent acquisition": "dfrft_only",
    "DFRFT + coherent refinement": "hybrid_dfrft_coherent",
}

CANDIDATES = {
    "No compensation": 0,
    "Coherent, equal budget": 7534,
    "Coherent coarse-to-fine": 12034,
    "DFRFT-equivalent acquisition": 51 * 2048,
    "DFRFT + coherent refinement": 51 * 2048 + 4941,
    "Oracle compensation": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(RESULT_ROOT.relative_to(ROOT)))
    parser.add_argument(
        "--timing-source",
        choices=("validation", "test"),
        default="validation",
        help="Use the isolated validation timing screen by default; the long test run may overlap other jobs.",
    )
    return parser.parse_args()


def load_accuracy(name: str) -> float | None:
    filename = FROZEN_SUMMARIES[name]
    if filename is None:
        path = ROOT / "outputs/results/hybrid_dfrft/protocol_b/proposed/seed_41_test.summary.json"
    else:
        path = RESULT_ROOT / "frozen_classifier" / filename
    if not path.exists():
        return None
    return 100.0 * float(json.loads(path.read_text(encoding="utf-8"))["accuracy"])


def main() -> None:
    args = parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    test = np.asarray(np.load(SPLITS)["test"], dtype=np.int64)
    timing_path = RESULT_ROOT / "independent_test/summary.json"
    timings = {}
    timing_scope = "independent-test subset"
    if args.timing_source == "test" and timing_path.exists():
        timings = json.loads(timing_path.read_text(encoding="utf-8"))["summary"]
    else:
        timing_scope = "validation subset"
        validation_timing_path = RESULT_ROOT / "validation/summary.json"
        timings = json.loads(validation_timing_path.read_text(encoding="utf-8"))["summary"]

    rows: list[dict[str, object]] = []
    with h5py.File(RAW, "r") as raw:
        truth_mu = np.asarray(raw["mu"][test], dtype=np.float64)
        truth_cfo = np.asarray(raw["fd0"][test], dtype=np.float64)
        duration = np.asarray(raw["T"][test], dtype=np.float64)
        for name, (path, mu_key, cfo_key) in ESTIMATORS.items():
            if name == "No compensation":
                mu_hat = np.zeros_like(truth_mu)
                cfo_hat = np.zeros_like(truth_cfo)
            else:
                if path is None or not path.exists():
                    raise FileNotFoundError(path)
                with h5py.File(path, "r") as source:
                    mu_hat = np.asarray(source[mu_key][test], dtype=np.float64)
                    cfo_hat = np.asarray(source[cfo_key][test], dtype=np.float64)
            em = np.abs(truth_mu - mu_hat)
            ef = np.abs(truth_cfo - cfo_hat)
            eta = 2.0 * np.pi * ef * duration + np.pi * em * duration**2
            latency_name = (
                LATENCY_NAMES.get(name)
                if timing_scope == "independent-test subset"
                else VALIDATION_LATENCY_NAMES.get(name)
            )
            latency = None
            if latency_name in timings:
                latency = float(timings[latency_name]["median_cpu_ms"])
            elif name == "No compensation":
                latency = 0.0
            rows.append(
                {
                    "front_end": name,
                    "scope": "full independent test (7700 frames)",
                    "search_scores_per_frame": CANDIDATES[name],
                    "rate_p95_hz_per_s": float(np.quantile(em, 0.95)),
                    "cfo_p95_hz": float(np.quantile(ef, 0.95)),
                    "eta_p95": float(np.quantile(eta, 0.95)),
                    "frozen_seed41_accuracy_percent": load_accuracy(name),
                    "median_cpu_ms": latency,
                    "median_gpu_ms": None,
                }
            )

    exhaustive_path = RESULT_ROOT / "exhaustive_subset/summary.json"
    subset_comparison: dict[str, object] = {}
    if exhaustive_path.exists():
        exact = json.loads(exhaustive_path.read_text(encoding="utf-8"))
        rows.insert(
            3,
            {
                "front_end": "Global coherent 5 Hz/s x 1 Hz",
                "scope": "fixed stratified test subset (77 frames)",
                "search_scores_per_frame": int(exact["joint_candidates_per_frame"]),
                "rate_p95_hz_per_s": float(exact["rate_p95_hz_per_s"]),
                "cfo_p95_hz": float(exact["cfo_p95_hz"]),
                "eta_p95": float(exact["eta_p95"]),
                "frozen_seed41_accuracy_percent": None,
                "median_cpu_ms": None,
                "median_gpu_ms": float(exact["median_gpu_ms_per_frame"]),
            },
        )
        subset_index = pd.read_csv(
            RESULT_ROOT / "exhaustive_subset/frame_results.csv"
        )["index"].to_numpy(np.int64)
        with h5py.File(RAW, "r") as raw:
            subset_mu = np.asarray(raw["mu"][subset_index], dtype=np.float64)
            subset_cfo = np.asarray(raw["fd0"][subset_index], dtype=np.float64)
            subset_duration = np.asarray(raw["T"][subset_index], dtype=np.float64)
            for name, (path, mu_key, cfo_key) in ESTIMATORS.items():
                if name in {"No compensation", "Oracle compensation"}:
                    continue
                assert path is not None
                with h5py.File(path, "r") as source:
                    em = np.abs(subset_mu - np.asarray(source[mu_key][subset_index]))
                    ef = np.abs(subset_cfo - np.asarray(source[cfo_key][subset_index]))
                eta = 2.0 * np.pi * ef * subset_duration + np.pi * em * subset_duration**2
                subset_comparison[name] = {
                    "frames": int(subset_index.size),
                    "rate_p95_hz_per_s": float(np.quantile(em, 0.95)),
                    "cfo_p95_hz": float(np.quantile(ef, 0.95)),
                    "eta_p95": float(np.quantile(eta, 0.95)),
                }
        subset_comparison["Global coherent 5 Hz/s x 1 Hz"] = {
            "frames": int(exact["frames"]),
            "rate_p95_hz_per_s": float(exact["rate_p95_hz_per_s"]),
            "cfo_p95_hz": float(exact["cfo_p95_hz"]),
            "eta_p95": float(exact["eta_p95"]),
        }

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "task_table.csv", index=False)
    hybrid_predictions = pd.read_csv(
        ROOT / "outputs/results/hybrid_dfrft/protocol_b/proposed/seed_41_test.csv"
    ).set_index("index")
    hybrid_correct = hybrid_predictions["correct"].astype(float)
    rng = np.random.default_rng(20260905)
    paired: dict[str, object] = {}
    for name, filename in FROZEN_SUMMARIES.items():
        if filename is None:
            continue
        path = RESULT_ROOT / "frozen_classifier" / filename.replace(".summary.json", ".csv")
        if not path.exists():
            continue
        candidate = pd.read_csv(path).set_index("index").loc[hybrid_correct.index]
        delta = 100.0 * (hybrid_correct.to_numpy() - candidate["correct"].to_numpy(float))
        draw = rng.integers(0, delta.size, size=(5000, delta.size))
        boot = delta[draw].mean(axis=1)
        paired[name] = {
            "hybrid_minus_candidate_pp": float(delta.mean()),
            "paired_bootstrap_95_ci_pp": [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ],
            "hybrid_only_correct": int(np.count_nonzero(delta > 0.0)),
            "candidate_only_correct": int(np.count_nonzero(delta < 0.0)),
            "bootstrap_replicates": 5000,
        }

    payload = {
        "table": rows,
        "paired_accuracy_vs_hybrid": paired,
        "matched_exhaustive_subset": subset_comparison,
        "artifact_lineage": {
            "raw_data": str(RAW.relative_to(ROOT)).replace("\\", "/"),
            "splits": str(SPLITS.relative_to(ROOT)).replace("\\", "/"),
            "frozen_checkpoint": "outputs/checkpoints/hybrid_dfrft/protocol_b/proposed/seed_41/best.pt",
            "classifier_inputs": {
                name: {"iq": paths[0], "descriptor": paths[1]}
                for name, paths in CLASSIFIER_INPUTS.items()
            },
        },
        "accuracy_intervention": (
            "One hybrid-trained seed-41 checkpoint is frozen; only the front-end "
            "I/Q and 48-D descriptor artifacts are replaced."
        ),
        "timing_policy": (
            f"Full-set methods use scalar CPU medians on the {timing_scope}. The exhaustive "
            "row is a CUDA subset timing and is not compared numerically with CPU timing."
        ),
    }
    (output / "task_table.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(frame.to_string(index=False))


def frft_comparison_main():
    """Generate receiver features, rebuild classifiers, and report the new campaign."""
    import sys
    import time
    import importlib.util
    import hashlib
    import subprocess
    from threadpoolctl import threadpool_limits
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    threadpool_limits(limits=1)
    p = argparse.ArgumentParser()
    p.add_argument('--frft-comparison', action='store_true')
    p.add_argument('--stage', choices=['features', 'train', 'summary', 'all', 'estimation-report'], default='all')
    p.add_argument('--wait-for-estimates', action='store_true')
    p.add_argument('--wait-for-baselines', action='store_true')
    p.add_argument('--output-dir', default='outputs/results/frft_dcft_comparison')
    args = p.parse_args()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'estimation-report':
        frft_estimation_report(out)
        return
    methods = ('coherent_budget', 'coherent_two_stage', 'dcft_capture', 'frft_capture', 'dcft_refined', 'frft_refined')
    seeds = (41, 73, 107, 149, 211)
    split = np.load(SPLITS)
    test_ix = np.sort(split['test'])
    trainval = np.sort(np.r_[split['train'], split['val']])
    all_ix = np.sort(np.r_[trainval, test_ix])

    def load_script(name):
        spec = importlib.util.spec_from_file_location('campaign_' + name[:2], ROOT / 'scripts' / name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    feature_module = load_script('10_precompute_symbol_constellation.py')
    runner = load_script('16_run_hybrid_dfrft_seeds.py')
    from src.drc_hoc.compensation import compensate_full_doppler
    from src.utils.math_utils import to_complex, complex_to_iq
    with h5py.File(ROOT / 'data/features/hybrid_dfrft/protocol_b_symbol_exact.h5', 'r') as original:
        descriptor_config = json.loads(original.attrs['config_json'])
    (out / 'descriptor_config.json').write_text(json.dumps(descriptor_config, indent=2))
    assert descriptor_config['timing_phases'] == [0] and descriptor_config['snr_source'] == 'pilot_evm'
    taps = feature_module.rrc_filter(beta=descriptor_config['rrc_beta'], span=descriptor_config['rrc_span'], sps=8)
    mods = tuple(descriptor_config['modulations'])
    constellations = {m: feature_module.get_constellation(m).astype(np.complex128) for m in mods}
    fdir = ROOT / 'data/features/frft_dcft_comparison'
    fdir.mkdir(parents=True, exist_ok=True)
    feature_observations = {}

    def descriptor(comp, pi, ps):
        # The final original pipeline reads stored float32 compensated I/Q.
        comp = to_complex(complex_to_iq(comp, layout='channel_first').astype(np.float32))
        matched = np.convolve(comp, taps, mode='same')
        gain, pe, count = feature_module._estimate_gain_and_pilot_evm(matched, pi, ps, 0, 8)
        crossfit, _ = feature_module._estimate_crossfit_pilot_evm(matched, pi, ps, 0, 8)
        pe = crossfit if np.isfinite(crossfit) else pe
        symbols = feature_module._extract_equalized_symbols(matched, gain, 0, 8, pi, descriptor_config['symbol_trim'], descriptor_config['max_symbols'])
        symbols = feature_module._prepare_symbols(symbols, descriptor_config['center'], descriptor_config['power_normalize'])
        evm, phases = feature_module._score_evm_all_candidates(symbols, mods, constellations, np.array([0.]))
        values, names = feature_module._evm_feature_vector(evm, mods, pe, symbols, constellations, phases,
                    snr_db=feature_module._pilot_sample_snr_db(pe, samples_per_symbol=8, offset_db=descriptor_config['pilot_snr_offset_db']),
                    complexity_penalty_lambda=descriptor_config['complexity_penalty_lambda'], complexity_penalty_snr_threshold=-3.,
                    complexity_penalty_snr_scale=4., apsk_ring_penalty_lambda=descriptor_config['apsk_ring_penalty_lambda'],
                    apsk_phase_weight=1., apsk_occupancy_weight=.25, candidate_score_mode='exact_mixture')
        if len(values) != 48 or not np.isfinite(values).all():
            raise ValueError('Invalid full descriptor')
        return values, names

    def verify_feature_pipeline():
        # Audit against existing final DCFT files BEFORE generating any FrFT input.
        checks = []
        with h5py.File(RAW, 'r') as raw, \
             h5py.File(ROOT / 'data/features/hybrid_dfrft/protocol_b_pilot_hybrid.h5', 'r') as est, \
             h5py.File(ROOT / 'data/features/hybrid_dfrft/protocol_b_symbol_exact.h5', 'r') as sym, \
             h5py.File(ROOT / 'data/features/hybrid_dfrft/protocol_b_iqcomp.h5', 'r') as iq:
            for i in all_ix[np.linspace(0, len(all_ix)-1, 77, dtype=int)]:
                comp = compensate_full_doppler(to_complex(raw['iq'][i]), sample_rate_hz=200000,
                        fd0_hat_hz=float(est['pilot_fd0_hat'][i]), mu_hat_hz_per_s=float(est['pilot_mu_hat'][i]), centered=False)
                values, _ = descriptor(comp, raw['pilot_indices'][i], raw['pilot_symbols'][i])
                stored = sym['evm_features'][i]
                iq_new = complex_to_iq(comp, layout='channel_first').astype(np.float32)
                np.testing.assert_allclose(values, stored, rtol=2e-4, atol=2e-5)
                # Original IQ was produced by CUDA float32 phase arithmetic;
                # CPU complex128 compensation differs at rounding scale.
                np.testing.assert_allclose(iq_new, iq['iq_comp'][i], rtol=2e-4, atol=1e-4)
                checks.append(dict(index=int(i), descriptor_max_error=float(np.max(np.abs(values-stored))),
                                   iq_max_error=float(np.max(np.abs(iq_new-iq['iq_comp'][i])))))
        pd.DataFrame(checks).to_csv(out / 'feature_pipeline_audit.csv', index=False)

    def read_estimates(method):
        path = out / 'test' / (method + '.csv')
        frame = pd.read_csv(path)
        assert len(frame) == 7700 and set(frame['index']) == set(test_ix)
        if method in ('dcft_refined','frft_refined'):
            more = pd.read_csv(out / 'training_estimates' / (method+'.csv'))
            assert len(more) == len(trainval) and set(more['index']) == set(trainval)
            frame = pd.concat([frame, more])
        return frame.sort_values('index')

    def generate_features(method):
        estimates = read_estimates(method)
        dest = fdir / (method + '.h5')
        with h5py.File(RAW, 'r') as raw, h5py.File(dest, 'a') as h:
            n, _, width = raw['iq'].shape
            if not feature_observations:
                print('Loading shared IQ/pilots for feature generation',flush=True)
                for key in ('iq','pilot_indices','pilot_symbols'):
                    feature_observations[key] = raw[key][:]
            if 'done' not in h:
                h.create_dataset('iq_comp', shape=(n, 2, width), dtype='f4', chunks=(1,2,width), fillvalue=np.nan)
                h.create_dataset('evm_features', shape=(n,48), dtype='f4', fillvalue=np.nan)
                h.create_dataset('h_drc', shape=(n,21), dtype='f4')
                for key in ('gamma_hat','S_peak','V_hoc','mu_hat'):
                    h.create_dataset(key, shape=(n,), dtype='f4')
                h.create_dataset('done', shape=(n,), dtype='bool')
                h.attrs['method'] = method
                h.attrs['configuration_sha256'] = (out / 'frozen_selection.sha256').read_text()
            assert h.attrs['configuration_sha256'] == (out / 'frozen_selection.sha256').read_text()
            for pos, row in enumerate(estimates.itertuples()):
                i = int(row.index)
                if h['done'][i]:
                    continue
                comp = compensate_full_doppler(to_complex(feature_observations['iq'][i]), sample_rate_hz=200000,
                        fd0_hat_hz=row.cfo_hat_hz, mu_hat_hz_per_s=row.mu_hat_hz_per_s, centered=False)
                values, names = descriptor(comp, feature_observations['pilot_indices'][i], feature_observations['pilot_symbols'][i])
                h['iq_comp'][i] = complex_to_iq(comp, layout='channel_first')
                h['evm_features'][i] = values
                h['mu_hat'][i] = row.mu_hat_hz_per_s
                if 'evm_feature_names' not in h:
                    h.create_dataset('evm_feature_names', data=np.asarray(names,dtype=h5py.string_dtype('utf-8')))
                h['done'][i] = True
                if (pos+1) % 200 == 0:
                    h.flush()
                    print(f'features/{method}: {pos+1}/{len(estimates)}', flush=True)
        return dest

    def complete_estimates():
        try:
            for method in methods:
                read_estimates(method)
            return True
        except (FileNotFoundError, AssertionError, pd.errors.EmptyDataError):
            return False

    if args.stage in ('features','all'):
        verify_feature_pipeline()
        if args.wait_for_estimates:
            while not complete_estimates():
                print('Waiting for complete frozen test/training estimates', flush=True)
                time.sleep(60)
        if not complete_estimates():
            raise RuntimeError('Estimation stage incomplete')
        for method in methods:
            generate_features(method)
        feature_observations.clear()

    def run_logged(command):
        with (out / 'commands.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(command) + '\n')
        runner.run(command, False)

    def evaluate(checkpoint, feature, symbol, dest):
        if dest.exists() and dest.with_suffix('.summary.json').exists():
            frame = pd.read_csv(dest)
            assert len(frame) == 7700 and set(frame['index']) == set(test_ix)
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        run_logged([str(ROOT / '.venv-cuda/Scripts/python.exe'), 'scripts/13_evaluate_model.py',
                    '--checkpoint', str(checkpoint), '--raw-data', str(RAW), '--feature-data', str(feature),
                    '--symbol-feature-data', str(symbol), '--splits', str(SPLITS), '--split','test',
                    '--model','drc_dualnet', '--output',str(dest), '--summary-output',str(dest.with_suffix('.summary.json')),
                    '--batch-size','256','--num-workers','0','--device','cuda','--strict','--load-into-memory'])

    if args.stage in ('train','all'):
        if args.wait_for_baselines:
            while True:
                # Inspect actual Python workers; never race the existing GPU run.
                cmd = "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object { $_.CommandLine -match '16_run_hybrid_dfrft_seeds.py|12_train_model.py|13_evaluate_model.py' } | Select-Object -ExpandProperty ProcessId"
                busy = subprocess.check_output(['powershell','-NoProfile','-Command',cmd], text=True).strip()
                if not busy:
                    break
                print('Waiting for previous GPU baseline campaign: ' + busy.replace('\n',','), flush=True)
                time.sleep(60)
        for method in ('dcft_refined','frft_refined'):
            feature = fdir / (method+'.h5')
            symbol = feature
            with h5py.File(feature, 'r') as h:
                assert h['done'][:].all()
            for seed in seeds:
                checkpoint_dir = ROOT / 'outputs/checkpoints/frft_dcft_comparison' / method / f'seed_{seed}'
                best = checkpoint_dir / 'best.pt'
                if not (best.exists() and (checkpoint_dir / 'train_result.json').exists()):
                    command = [str(ROOT / '.venv-cuda/Scripts/python.exe'),'scripts/12_train_model.py',
                         '--raw-data',str(RAW),'--feature-data',str(feature),'--symbol-feature-data',str(symbol),
                         '--splits',str(SPLITS),'--model','drc_dualnet','--model-config','configs/model/dualnet_iq_evm.yaml',
                         '--train-config','configs/train/train_long.yaml','--output-dir',str(checkpoint_dir),
                         '--seed',str(seed),'--device','cuda','--num-workers','0','--iq-source','comp',
                         '--iq-representation','iq','--iq-normalize','zscore','--cache-iq','--no-hoc-standardize']
                    if (checkpoint_dir / 'last.pt').exists():
                        command.extend(['--resume',str(checkpoint_dir / 'last.pt')])
                    run_logged(command)
                evaluate(best, feature, symbol, out / 'matched_training' / f'{method}_seed{seed}.csv')
                if method == 'dcft_refined' and seed == 41:
                    import torch
                    checkpoint_config = torch.load(best, map_location='cpu', weights_only=False)['config']
                    assert checkpoint_config['seed'] == 41 and checkpoint_config['standardize_evm']
                    assert len(checkpoint_config['evm_stats']['mean']) == 48
                    (out / 'frozen_normalization.json').write_text(json.dumps(checkpoint_config['evm_stats'],indent=2))
                    (out / 'frozen_checkpoint.json').write_text(json.dumps(dict(path=str(best),
                         sha256=hashlib.sha256(best.read_bytes()).hexdigest(), status='rebuilt_frozen_reference',
                         note='Original checkpoint unavailable; user authorized original-configuration rebuild. All six frontends use checkpoint statistics without refitting.'), indent=2))
                    for front in methods:
                        with h5py.File(fdir / (front+'.h5'), 'r') as h:
                            assert h['done'][test_ix].all()
                        evaluate(best, fdir / (front+'.h5'), fdir / (front+'.h5'), out / 'frozen_classifier' / (front+'.csv'))
    if args.stage in ('summary','all'):
        run_logged([str(ROOT / '.venv/Scripts/python.exe'), 'scripts/26_screen_frontend_attribution.py',
                    '--frft-comparison','--stage','timing','--output-dir',str(out)])
        frft_campaign_summary(out, methods, seeds, test_ix)


def frft_estimation_report(out):
    """Full-test estimation figure, independently available before training ends."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = pd.read_csv(out / 'estimation_summary.csv', index_col=0)
    order = ['coherent_budget','coherent_two_stage','dcft_capture','frft_capture','dcft_refined','frft_refined']
    summary = summary.loc[order]
    assert (summary['frames'] == 7700).all()
    labels = ['Coherent\nbudget-selected','Coherent\ncoarse-to-fine','DCFT\ncapture','FrFT\ncapture',
              'DCFT +\nrefinement','FrFT +\nrefinement']
    colors = ['#888888','#aaaaaa','#377eb8','#ff8c36','#377eb8','#ff8c36']
    fig, axes = plt.subplots(1,3,figsize=(12,3.9))
    panels = [('rate_rmse_hz_per_s','Rate RMSE (Hz/s)'),('cfo_rmse_hz','CFO RMSE (Hz)'),('B_phi_p95',r'$B_\phi$ P95 (rad)')]
    for ax,(column,ylabel) in zip(axes,panels):
        values = summary[column].to_numpy()
        bars = ax.bar(np.arange(6),values,color=colors,width=.72)
        ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=3,fontsize=8)
        ax.set_xticks(np.arange(6),labels,rotation=35,ha='right',fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0,values.max()*1.18)
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for ext in ('pdf','png'):
        fig.savefig(out / ('estimation_errors.'+ext),dpi=220)
    plt.close(fig)
    print(summary[['rate_mae_hz_per_s','rate_rmse_hz_per_s','rate_p95_hz_per_s',
                   'cfo_mae_hz','cfo_rmse_hz','cfo_p95_hz','B_phi_p95']].to_string(),flush=True)


def stratified_paired_accuracy_interval(first, second, labels, snr, replicates=5000, seed=20261018):
    """Resample paired frame differences within fixed label-by-SNR strata."""
    first, second, labels, snr = [np.asarray(x) for x in (first,second,labels,snr)]
    if not (first.ndim == second.ndim == labels.ndim == snr.ndim == 1
            and len(first) == len(second) == len(labels) == len(snr) and len(first) > 0):
        raise ValueError('Bootstrap inputs must be nonempty aligned vectors')
    if not np.isin(first,[0,1]).all() or not np.isin(second,[0,1]).all():
        raise ValueError('Expected per-frame binary correctness')
    delta = first.astype(float)-second.astype(float)
    rng = np.random.default_rng(seed)
    draws = np.zeros(replicates)
    for label in np.unique(labels):
        for level in np.unique(snr):
            cell = np.flatnonzero((labels==label)&(snr==level))
            if len(cell):
                draws += delta[cell[rng.integers(0,len(cell),size=(replicates,len(cell)))]].sum(axis=1)
    draws *= 100/len(delta)
    return dict(gain_pp=float(100*delta.mean()),ci95_low=float(np.quantile(draws,.025)),
                ci95_high=float(np.quantile(draws,.975)),bootstrap_replicates=replicates,strata='label x SNR')


def frft_campaign_summary(out, methods, seeds, test_ix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frft_estimation_report(out)
    def predictions(path):
        frame = pd.read_csv(path).sort_values('index')
        assert len(frame) == 7700 and np.array_equal(frame['index'], test_ix)
        return frame
    frozen = {m: predictions(out / 'frozen_classifier' / (m+'.csv')) for m in methods}
    # Prediction CSV schema is checked explicitly; never silently use a wrong column.
    first = frozen[methods[0]]
    label_key = next(k for k in ('y_true','label','target') if k in first)
    pred_key = next(k for k in ('y_pred','prediction','pred') if k in first)
    with h5py.File(RAW, 'r') as raw:
        labels, snr = raw['label'][test_ix], raw['snr_db'][test_ix]
    correct = {}
    grouped = []
    for name, frame in frozen.items():
        assert np.array_equal(frame[label_key], labels)
        correct[name] = (frame[pred_key].to_numpy() == labels).astype(float)
        for level in np.unique(snr):
            mask = snr == level
            grouped.append(dict(method=name,snr_db=level,n=int(mask.sum()),accuracy_pct=100*correct[name][mask].mean()))
    pd.DataFrame(grouped).to_csv(out / 'accuracy_by_snr.csv', index=False)
    summary = pd.read_csv(out / 'estimation_summary.csv', index_col=0)
    summary['frozen_accuracy_pct'] = pd.Series({m:100*c.mean() for m,c in correct.items()})
    summary.to_csv(out / 'main_comparison.csv')
    contrasts = [('dcft_refined','frft_refined'),('dcft_capture','frft_capture'),
                 ('dcft_refined','dcft_capture'),('frft_refined','frft_capture'),
                 ('dcft_refined','coherent_budget'),('frft_refined','coherent_budget')]
    ci_rows = []
    for a,b in contrasts:
        ci_rows.append(dict(first=a,second=b,**stratified_paired_accuracy_interval(correct[a],correct[b],labels,snr)))
    pd.DataFrame(ci_rows).to_csv(out / 'stratified_paired_bootstrap.csv', index=False)
    training = []
    for seed in seeds:
        row = dict(seed=seed)
        for method in ('dcft_refined','frft_refined'):
            f = predictions(out / 'matched_training' / f'{method}_seed{seed}.csv')
            assert np.array_equal(f[label_key],labels)
            row[method] = 100*np.mean(f[pred_key].to_numpy()==labels)
        row['dcft_minus_frft_pp'] = row['dcft_refined']-row['frft_refined']
        training.append(row)
    train = pd.DataFrame(training).set_index('seed')
    train.to_csv(out / 'matched_seed_accuracy.csv')
    train.agg(['mean','std']).T.to_csv(out / 'matched_training_summary.csv')
    fig, ax = plt.subplots(figsize=(7,4.2))
    for method, part in pd.DataFrame(grouped).groupby('method', sort=False):
        ax.plot(part.snr_db,part.accuracy_pct,marker='o',label=method.replace('_',' '))
    ax.set(xlabel='SNR (dB)',ylabel='Test accuracy (%)')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('pdf','png'):
        fig.savefig(out / ('frozen_accuracy_vs_snr.'+ext), dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(9,3.8))
    for col, ax, ylabel in [('B_phi_p95',axes[0],r'$B_\phi$ P95 (rad)'),('median_cpu_ms',axes[1],'Median estimation time (ms)')]:
        ax.bar(np.arange(len(summary)),summary[col])
        ax.set_xticks(np.arange(len(summary)),[m.replace('_','\n') for m in summary.index],fontsize=7)
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    for ext in ('pdf','png'):
        fig.savefig(out / ('estimation_comparison.'+ext),dpi=200)
    plt.close(fig)
    (out / 'COMPLETED.json').write_text(json.dumps(dict(test_frames=7700,methods=6,matched_training_seeds=list(seeds),
                 frozen_reference='rebuilt original-configuration DCFT seed 41',bootstrap_replicates=5000),indent=2))
    print(summary.to_string(),flush=True)
    print(train.agg(['mean','std']).to_string(),flush=True)


if __name__ == "__main__":
    import sys
    if '--frft-comparison' in sys.argv:
        frft_comparison_main()
    else:
        main()
