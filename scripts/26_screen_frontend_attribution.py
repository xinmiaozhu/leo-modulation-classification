#!/usr/bin/env python
"""Attribute Protocol-B carrier recovery to acquisition and refinement stages.

Validation mode compares a frozen bank of coherent coarse-to-fine searches and
selects the most accurate candidate whose median CPU latency is within 10% of
the hybrid front end.  Test mode loads that frozen choice.  The same stratified
frames are used for every method, and DFRFT-only stops before coherent local
refinement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import pandas as pd

from src.drc_hoc.hybrid_dfrft_estimator import HybridDFRFTConfig, HybridDFRFTPilotEstimator
from src.drc_hoc.pilot_estimator import PilotMuEstimator, PilotMuEstimatorConfig, PilotMuResult


COHERENT_CANDIDATES = {
    "coherent_40x25": (40.0, 25.0),
    "coherent_80x25": (80.0, 25.0),
    "coherent_80x50": (80.0, 50.0),
    "coherent_120x25": (120.0, 25.0),
    "coherent_120x50": (120.0, 50.0),
    "coherent_160x25": (160.0, 25.0),
    "coherent_160x50": (160.0, 50.0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-data", default="data/processed/leo_7mods_joint_practical.h5")
    p.add_argument("--splits", default="data/splits/leo_7mods_joint_practical_splits.npz")
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--samples-per-class-snr", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--latency-frames", type=int, default=77)
    p.add_argument("--budget-latency-ratio", type=float, default=1.10)
    p.add_argument("--selection-json")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def base_config() -> PilotMuEstimatorConfig:
    return PilotMuEstimatorConfig(
        sample_rate_hz=200_000.0,
        samples_per_symbol=8,
        rrc_beta=0.35,
        rrc_span=8,
        timing_phases=(0,),
        mu_min=-8160.0,
        mu_max=-180.0,
        coarse_step_hz_per_s=40.0,
        fine_radius_hz_per_s=120.0,
        fine_step_hz_per_s=5.0,
        fd0_min_hz=-550.0,
        fd0_max_hz=550.0,
        fd0_step_hz=25.0,
        fine_fd0_radius_hz=30.0,
        fine_fd0_step_hz=1.0,
        pilot_weighting="coherent",
    )


def hybrid_config() -> HybridDFRFTConfig:
    return HybridDFRFTConfig(
        coarse_mu_step_hz_per_s=160.0,
        fft_size=2048,
        fine_mu_radius_hz_per_s=200.0,
        fine_mu_step_hz_per_s=5.0,
        fine_fd0_radius_hz=30.0,
        fine_fd0_step_hz=1.0,
    )


class DFRFTOnlyEstimator:
    def __init__(self, pilot: PilotMuEstimatorConfig, hybrid: HybridDFRFTConfig) -> None:
        self.estimator = HybridDFRFTPilotEstimator(pilot, hybrid)

    def estimate(self, iq: np.ndarray, pilot_indices: np.ndarray, pilot_symbols: np.ndarray) -> PilotMuResult:
        best = None
        for timing in self.estimator.pilot.timing_phases:
            z, t = self.estimator.pilot._extract_pilot_observations(
                iq, pilot_indices, pilot_symbols, int(timing)
            )
            payload = self.estimator._chirp_focus(z, t)
            if best is None or float(payload[3]) > float(best[0][3]):
                best = (payload, int(timing), int(z.size))
        assert best is not None
        payload, timing, count = best
        mu, _, fd0, score, margin, _, _ = payload
        return PilotMuResult(
            float(mu), float(fd0), float(score), float(margin), timing, count,
            bool(np.isfinite(score)),
        )


def stratified_indices(candidates: np.ndarray, labels: np.ndarray, snr: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in np.unique(labels[candidates]):
        for level in np.unique(snr[candidates]):
            cell = candidates[(labels[candidates] == label) & np.isclose(snr[candidates], level)]
            if cell.size < count:
                raise ValueError(f"label={label}, SNR={level}: requested {count}, available {cell.size}")
            selected.extend(rng.choice(cell, count, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    em = frame["abs_mu_error_hz_per_s"].to_numpy(float)
    ef = frame["abs_cfo_error_hz"].to_numpy(float)
    eta = frame["eta"].to_numpy(float)
    return {
        "frames": int(len(frame)),
        "rate_rmse_hz_per_s": float(np.sqrt(np.mean(em**2))),
        "rate_p95_hz_per_s": float(np.quantile(em, 0.95)),
        "cfo_rmse_hz": float(np.sqrt(np.mean(ef**2))),
        "cfo_p95_hz": float(np.quantile(ef, 0.95)),
        "eta_p90": float(np.quantile(eta, 0.90)),
        "eta_p95": float(np.quantile(eta, 0.95)),
    }


def main() -> None:
    args = parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    split_file = np.load(ROOT / args.splits)
    candidates = np.asarray(split_file[args.split], dtype=np.int64)
    with h5py.File(ROOT / args.raw_data, "r") as raw:
        labels = np.asarray(raw["label"][:], dtype=np.int64)
        snr = np.asarray(raw["snr_db"][:], dtype=np.float64)
        chosen = stratified_indices(candidates, labels, snr, args.samples_per_class_snr, args.seed)
        payloads = [
            (
                int(i),
                np.asarray(raw["iq"][i], dtype=np.float32),
                np.asarray(raw["pilot_indices"][i], dtype=np.int64),
                np.asarray(raw["pilot_symbols"][i], dtype=np.complex64),
                float(raw["mu"][i]),
                float(raw["fd0"][i]),
                float(raw["T"][i]),
                int(labels[i]),
                float(snr[i]),
            )
            for i in chosen
        ]

    base = base_config()
    estimators: dict[str, object] = {
        "dfrft_only": DFRFTOnlyEstimator(base, hybrid_config()),
        "hybrid_dfrft_coherent": HybridDFRFTPilotEstimator(base, hybrid_config()),
    }
    if args.split == "val":
        for name, (mu_step, cfo_step) in COHERENT_CANDIDATES.items():
            estimators[name] = PilotMuEstimator(
                replace(base, coarse_step_hz_per_s=mu_step, fd0_step_hz=cfo_step)
            )
        estimators["coherent_centered_40x25"] = PilotMuEstimator(replace(base, center_time=True))
    else:
        if not args.selection_json:
            raise ValueError("--selection-json is required for test mode")
        selected_cfg = json.loads((ROOT / args.selection_json).read_text(encoding="utf-8"))
        name = str(selected_cfg["selected_equal_budget_method"])
        mu_step, cfo_step = selected_cfg["coherent_candidates"][name]
        estimators["coherent_current"] = PilotMuEstimator(base)
        estimators["coherent_equal_budget"] = PilotMuEstimator(
            replace(base, coarse_step_hz_per_s=float(mu_step), fd0_step_hz=float(cfo_step))
        )
        estimators["coherent_centered_40x25"] = PilotMuEstimator(replace(base, center_time=True))

    rows: list[dict[str, object]] = []
    latency: dict[str, float] = {}
    latency_count = min(max(args.latency_frames, 1), len(payloads))
    for name, estimator in estimators.items():
        start_samples: list[float] = []
        estimator.estimate(*payloads[0][1:4])
        for position, item in enumerate(payloads):
            index, iq, pilot_indices, pilot_symbols, truth_mu, truth_cfo, duration, label, level = item
            start = time.perf_counter()
            result = estimator.estimate(iq, pilot_indices, pilot_symbols)
            elapsed_ms = 1000.0 * (time.perf_counter() - start)
            if position < latency_count:
                start_samples.append(elapsed_ms)
            if hasattr(result, "estimate"):
                result = result.estimate
            em = truth_mu - float(result.mu_hat_hz_per_s)
            ef = truth_cfo - float(result.fd0_hat_hz)
            rows.append({
                "index": index,
                "method": name,
                "label": label,
                "snr_db": level,
                "mu_hat_hz_per_s": float(result.mu_hat_hz_per_s),
                "cfo_hat_hz": float(result.fd0_hat_hz),
                "abs_mu_error_hz_per_s": abs(em),
                "abs_cfo_error_hz": abs(ef),
                "eta": 2.0 * np.pi * abs(ef) * duration + np.pi * abs(em) * duration**2,
                "valid": bool(result.valid),
            })
        latency[name] = float(np.median(start_samples))

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "frame_results.csv", index=False)
    summary = {name: metrics(part) for name, part in frame.groupby("method", sort=False)}
    for name in summary:
        summary[name]["median_cpu_ms"] = latency[name]

    selection = name if args.split == "test" else None
    if args.split == "val":
        hybrid_ms = latency["hybrid_dfrft_coherent"]
        eligible = [
            name for name in COHERENT_CANDIDATES
            if latency[name] <= args.budget_latency_ratio * hybrid_ms
        ]
        if not eligible:
            eligible = [min(COHERENT_CANDIDATES, key=lambda name: abs(latency[name] - hybrid_ms))]
        selection = min(eligible, key=lambda name: summary[name]["eta_p95"])
        selection_payload = {
            "selection_split": "validation",
            "latency_budget_ratio": float(args.budget_latency_ratio),
            "hybrid_median_cpu_ms": hybrid_ms,
            "selected_equal_budget_method": selection,
            "coherent_candidates": {k: list(v) for k, v in COHERENT_CANDIDATES.items()},
            "eligible_methods": eligible,
            "candidate_summary": {k: summary[k] for k in COHERENT_CANDIDATES},
        }
        (output / "equal_budget_selection.json").write_text(
            json.dumps(selection_payload, indent=2) + "\n", encoding="utf-8"
        )

    result = {
        "split": args.split,
        "num_frames": len(payloads),
        "samples_per_class_snr": args.samples_per_class_snr,
        "selection_seed": args.seed,
        "summary": summary,
        "selected_equal_budget_method": selection,
        "information_policy": {
            "known_pilots_only": True,
            "identical_frames_across_methods": True,
            "test_not_used_for_equal_budget_selection": True,
        },
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary).T.to_string())
    if selection:
        print(f"SELECTED_EQUAL_BUDGET={selection}")


def comparison_main():
    """Resumable, separate DCFT/FrFT campaign; historical outputs are untouched."""
    import os
    import hashlib
    from dataclasses import asdict
    from threadpoolctl import threadpool_limits
    from src.drc_hoc.hybrid_dfrft_estimator import FrFTPilotEstimator, FrFTCaptureConfig

    p = argparse.ArgumentParser()
    p.add_argument('--frft-comparison', action='store_true')
    p.add_argument('--stage', choices=['select', 'test', 'train-estimates', 'all-estimates', 'timing'], default='all-estimates')
    p.add_argument('--output-dir', default='outputs/results/frft_dcft_comparison')
    args = p.parse_args()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    threadpool_limits(limits=1)
    raw_path = ROOT / 'data/processed/leo_7mods_joint_practical.h5'
    splits = np.load(ROOT / 'data/splits/leo_7mods_joint_practical_splits.npz')
    validation = np.sort(pd.read_csv(ROOT / 'outputs/results/frontend_attribution/validation/frame_results.csv')['index'].unique())
    assert len(validation) == 770 and set(validation) <= set(splits['val'])
    assert not set(validation) & set(splits['test'])
    plan_path = out / 'frozen_selection.json'
    observation_cache = {}

    def cached_observations(raw):
        if not observation_cache:
            # Original gzip chunks span 241 frames: repeated scalar reads would
            # decompress the same chunks thousands of times. Cache IQ/pilots
            # outside every timer; metadata are never passed to the estimator.
            print('Loading shared IQ/pilot cache outside estimator timing', flush=True)
            for key in ('iq','pilot_indices','pilot_symbols'):
                observation_cache[key] = raw[key][:]
        return observation_cache

    def capture(estimator):
        obj = DFRFTOnlyEstimator(base_config(), hybrid_config())
        obj.estimator = estimator
        return obj

    def frft(n, m, refine=True):
        obj = FrFTPilotEstimator(base_config(), hybrid_config(), FrFTCaptureConfig(n, m))
        return obj if refine else capture(obj)

    def summarize(frame):
        result = metrics(frame.rename(columns={'B_phi': 'eta'}))
        result.update(rate_mae_hz_per_s=float(frame.abs_mu_error_hz_per_s.mean()),
                      cfo_mae_hz=float(frame.abs_cfo_error_hz.mean()),
                      B_phi_p95=float(frame.B_phi.quantile(.95)))
        return result

    def evaluate(name, estimator, indices, folder):
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / (name + '.csv')
        existing = pd.read_csv(dest) if dest.exists() else pd.DataFrame()
        done = set(existing['index']) if len(existing) else set()
        if not done <= set(indices) or len(done) != len(existing):
            raise ValueError('Inconsistent resume indices: ' + str(dest))
        with h5py.File(raw_path, 'r') as raw:
            obs_cache = cached_observations(raw)
            for position, i in enumerate(indices):
                if int(i) in done:
                    continue
                iq, pi, ps = (obs_cache[key][i] for key in ('iq','pilot_indices','pilot_symbols'))
                result = estimator.estimate(iq, pi, ps)
                if hasattr(result, 'estimate'):
                    result = result.estimate
                if not result.valid:
                    raise ValueError(f'Invalid estimate {name} {i}')
                # Ground truth is read only after the receiver returned.
                em = float(result.mu_hat_hz_per_s) - float(raw['mu'][i])
                ef = float(result.fd0_hat_hz) - float(raw['fd0'][i])
                duration = float(raw['T'][i])
                row = dict(index=int(i), method=name, label=int(raw['label'][i]), snr_db=float(raw['snr_db'][i]),
                           mu_hat_hz_per_s=float(result.mu_hat_hz_per_s), cfo_hat_hz=float(result.fd0_hat_hz),
                           abs_mu_error_hz_per_s=abs(em), abs_cfo_error_hz=abs(ef),
                           B_phi=2*np.pi*abs(ef)*duration + np.pi*abs(em)*duration**2,
                           valid=True)
                pd.DataFrame([row]).to_csv(dest, mode='a', header=not dest.exists(), index=False)
                if (position + 1) % 100 == 0:
                    print(f'{folder.name}/{name}: {position+1}/{len(indices)}', flush=True)
        frame = pd.read_csv(dest)
        assert set(frame['index']) == set(indices)
        return summarize(frame)

    def benchmark(estimators, folder):
        """Same 77 stratified observations, five warmups, three randomized rounds."""
        with h5py.File(raw_path, 'r') as raw:
            ix = stratified_indices(validation, raw['label'][:], raw['snr_db'][:], 1, 20261018)
            observations = [(raw['iq'][i], raw['pilot_indices'][i], raw['pilot_symbols'][i]) for i in ix]
        for estimator in estimators.values():
            for obs in observations[:5]:
                estimator.estimate(*obs)
        rng = np.random.default_rng(20261018)
        rows = []
        for repeat in range(3):
            for position in rng.permutation(len(observations)):
                for name in rng.permutation(list(estimators)):
                    start = time.perf_counter_ns()
                    estimators[name].estimate(*observations[position])
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    rows.append(dict(method=name, index=int(ix[position]), repeat=repeat, cpu_ms=elapsed))
        frame = pd.DataFrame(rows)
        frame.to_csv(folder / 'timing_samples.csv', index=False)
        frame.groupby('method').cpu_ms.agg(['median', 'mean', 'std', 'count']).to_csv(folder / 'timing_summary.csv')
        return frame.groupby('method').cpu_ms.median().to_dict()

    if args.stage in ('select', 'all-estimates') and not plan_path.exists():
        # Run the independently checked kernel/unit/off-grid tests before validation.
        import subprocess
        subprocess.run([sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider',
                        'tests/test_hybrid_dfrft_estimator.py', '-q'], cwd=ROOT, check=True,
                       stdout=(out / 'verification.log').open('w'), stderr=subprocess.STDOUT)
        estimators = {'dcft_capture': capture(HybridDFRFTPilotEstimator(base_config(), hybrid_config())),
                      'dcft_refined': HybridDFRFTPilotEstimator(base_config(), hybrid_config())}
        for name, (mu, fd) in COHERENT_CANDIDATES.items():
            estimators[name] = PilotMuEstimator(replace(base_config(), coarse_step_hz_per_s=mu, fd0_step_hz=fd))
        frft_candidates = {f'frft_{n}_{m}': (n, m) for n in (17, 33, 65, 129) for m in (512, 1024, 2048)}
        for name, (n, m) in frft_candidates.items():
            estimators[name] = frft(n, m, False)
        valdir = out / 'validation'
        valdir.mkdir(exist_ok=True)
        latency = benchmark(estimators, valdir)
        scores = {name: evaluate(name, est, validation, valdir) for name, est in estimators.items()}
        def select(names, reference):
            ratios = {n: latency[n] / latency[reference] for n in names}
            eligible = [n for n in names if .9 <= ratios[n] <= 1.1]
            choice = min(eligible, key=lambda n: scores[n]['B_phi_p95']) if eligible else min(names, key=lambda n: abs(np.log(ratios[n])))
            return dict(selected=choice, matched_within_10_percent=bool(eligible), latency_ratio=ratios[choice],
                        target=reference, eligible=eligible, all_ratios=ratios)
        coherent_choice = select(list(COHERENT_CANDIDATES), 'dcft_refined')
        frft_choice = select(list(frft_candidates), 'dcft_capture')
        selection = dict(protocol='B', selection_split='validation', validation_indices=validation.tolist(),
                         pilot=asdict(base_config()), common_refiner=asdict(hybrid_config()),
                         coherent_choice=coherent_choice, frft_choice=frft_choice,
                         coherent_steps=COHERENT_CANDIDATES[coherent_choice['selected']],
                         frft_grid=frft_candidates[frft_choice['selected']], scores=scores, latency_ms=latency,
                         information_policy='Receiver uses IQ and known pilots only; truth and SNR only for scoring/stratification.',
                         timing_policy='Single CPU thread; 5 warmups; 77 stratified validation frames x 3 randomized rounds; includes extraction, capture and optional refinement; excludes IO/classification; concurrent job load recorded separately.',
                         frozen_reference='Rebuilt original-configuration DCFT seed 41, user authorized; original checkpoint unavailable.')
        serialized = json.dumps(selection, indent=2)
        plan_path.write_text(serialized, encoding='utf-8')
        (out / 'frozen_selection.sha256').write_text(hashlib.sha256(plan_path.read_bytes()).hexdigest())
        print('FROZEN ' + json.dumps([coherent_choice, frft_choice]), flush=True)
    if args.stage == 'select':
        return
    selection = json.loads(plan_path.read_text())
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == (out / 'frozen_selection.sha256').read_text()
    mu, fd = selection['coherent_steps']
    n, m = selection['frft_grid']
    estimators = dict(coherent_budget=PilotMuEstimator(replace(base_config(), coarse_step_hz_per_s=mu, fd0_step_hz=fd)),
                      coherent_two_stage=PilotMuEstimator(base_config()),
                      dcft_capture=capture(HybridDFRFTPilotEstimator(base_config(), hybrid_config())),
                      frft_capture=frft(n, m, False), dcft_refined=HybridDFRFTPilotEstimator(base_config(), hybrid_config()),
                      frft_refined=frft(n, m))
    if args.stage == 'timing':
        timingdir = out / 'final_timing'
        timingdir.mkdir(exist_ok=True)
        latency = benchmark(estimators, timingdir)
        summary = pd.read_csv(out / 'estimation_summary.csv', index_col=0)
        summary['median_cpu_ms'] = pd.Series(latency)
        summary.to_csv(out / 'estimation_summary.csv')
        ratios = dict(frft_capture_over_dcft_capture=latency['frft_capture']/latency['dcft_capture'],
                      frft_refined_over_dcft_refined=latency['frft_refined']/latency['dcft_refined'],
                      coherent_budget_over_dcft_refined=latency['coherent_budget']/latency['dcft_refined'])
        (timingdir / 'budget_audit.json').write_text(json.dumps({k:dict(ratio=v,matched_within_10_percent=.9<=v<=1.1) for k,v in ratios.items()},indent=2))
        return
    if args.stage in ('test', 'all-estimates'):
        testdir = out / 'test'
        testdir.mkdir(exist_ok=True)
        latency = benchmark(estimators, testdir)
        scores = {name: evaluate(name, est, np.sort(splits['test']), testdir) for name, est in estimators.items()}
        pd.DataFrame(scores).T.assign(median_cpu_ms=pd.Series(latency)).to_csv(out / 'estimation_summary.csv')
    if args.stage in ('train-estimates', 'all-estimates'):
        for name in ('dcft_refined','frft_refined'):
            evaluate(name, estimators[name], np.sort(np.r_[splits['train'], splits['val']]), out / 'training_estimates')


if __name__ == "__main__":
    if '--frft-comparison' in sys.argv:
        comparison_main()
    else:
        main()
