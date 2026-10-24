"""Paired Protocol-B observation-length campaign, separate from gamma calibration."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LENGTHS = (2048, 4096, 8192, 16384)
SEEDS = (41, 73, 107, 149, 211)
MODULATIONS = ('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '16APSK', '32APSK')
SNR = tuple(range(-10, 21, 3))
METHODS = {
    'proposed': ('drc_dualnet', 'configs/model/dualnet_iq_evm.yaml'),
    'iq_only': ('drc_dualnet', 'configs/model/dualnet_iq_only.yaml'),
    'iqformer': ('paper_iqformer', 'configs/model/paper_iqformer.yaml'),
}
NAME = 'frame_length_protocol_b'


def module(filename):
    spec = importlib.util.spec_from_file_location('frame_length_' + Path(filename).stem, ROOT / 'scripts' / filename)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def paths(n, name=NAME):
    data = ROOT / 'data' / 'experiments' / name
    result = ROOT / 'outputs/results' / name
    return dict(data=data, root=result, raw=data / f'raw_n{n}.h5', splits=data / 'splits.npz',
                pilot=data / f'pilot_n{n}.h5', feature=data / f'iqcomp_n{n}.h5',
                symbol=data / f'symbol_n{n}.h5', result=result / f'n{n}',
                checkpoint=ROOT / 'outputs/checkpoints' / name / f'n{n}')


def frontend_settings(n):
    # Preserve at least the paper resolution; long-frame phase quantization
    # is held constant by scaling rate and CFO steps with T^-2 and T^-1.
    ratio = max(1.0, n / 8192)
    return dict(coarse_mu_step=160 / ratio**2, fft_size=int(2048 * ratio),
                fine_mu_radius=200 / ratio**2, fine_mu_step=5 / ratio**2,
                fine_fd0_radius=30.0, fine_fd0_step=1 / ratio,
                unwrap_centered_cfo=True)


def protocol(smoke=False):
    return dict(version=1, protocol='B', lengths=list(LENGTHS), seeds=list(SEEDS),
                methods=list(METHODS), modulations=list(MODULATIONS), snr_db=list(SNR),
                data_seed=20261022, samples_per_class_snr={'train': 2 if smoke else 200,
                'val': 1 if smoke else 100, 'test': 1 if smoke else 100},
                fs_hz=200000, samples_per_symbol=8, pilot_interval_symbols=16,
                max_descriptor_symbols=512, mu_range=[-8160, -180], cfo_range=[-500, 500],
                rician_k_db=10, fractional_timing_samples=[-0.5, 0.5], phase_noise_std=0.0001,
                pairing='Identical prefixes of a common 16384-sample received waveform; split before slicing.',
                snr_definition='Expected noise power relative to the full parent received waveform; finite-prefix power fluctuates.',
                training='train_long.yaml; independent weights per N and seed; validation early stopping',
                diagnostic='Frozen proposed checkpoint with only carrier compensation changed; CD regenerated; original training normalization.',
                frontend={str(n): frontend_settings(n) for n in LENGTHS}, smoke=smoke)


def prepare_source(name=NAME, smoke=False):
    """Stream independent parent frames to disk; all lengths share exact prefixes."""
    from src.signal.signal_generator import LEOSignalGenerator, SignalSpec
    cfg = protocol(smoke)
    p = paths(16384, name)
    p['data'].mkdir(parents=True, exist_ok=True)
    config_path = p['root'] / 'protocol.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != cfg:
        raise ValueError('Existing campaign protocol differs; use a different campaign name.')
    write_json(config_path, cfg)
    source = p['data'] / 'source_16384.h5'
    plan = [(split, mi, si, rep) for split, count in cfg['samples_per_class_snr'].items()
            for mi in range(7) for si in range(len(SNR)) for rep in range(count)]
    total = len(plan)
    if not source.exists():
        temporary = source.with_suffix('.partial.h5')
        with h5py.File(temporary, 'a') as f:
            if 'iq' not in f:
                f.create_dataset('iq', shape=(total, 2, 16384), dtype='float32', chunks=(1, 2, 16384), compression='lzf')
                scalar_types = dict(label='i8', snr_db='f4', fd0='f4', fd0_frame_start='f4',
                    fractional_timing_offset_samples='f4', mu='f4', fs='f4', fc='f4',
                    track_id='i8', frame_id='i8', frame_time_s='f4', domain_id='i8',
                    orbit_altitude_km='f4', orbit_inclination_deg='f4', phase_noise_std_rad='f4',
                    modulation='S16', dataset_split='S16', pilot_pattern='S16', channel_type='S32')
                for key, dtype in scalar_types.items():
                    f.create_dataset(key, shape=(total,), dtype=dtype)
                f.create_dataset('pair_key', shape=(total, 5), dtype='i8')
                f.create_dataset('pilot_indices', shape=(total, 128), dtype='i8')
                f.create_dataset('pilot_sample_indices', shape=(total, 128), dtype='i8')
                f.create_dataset('pilot_symbols', shape=(total, 128), dtype='complex64')
                f.create_dataset('label_names', data=np.asarray(MODULATIONS, dtype='S16'))
                f.attrs['protocol_json'] = json.dumps(cfg)
                f.attrs['next_frame'] = 0
            if f['iq'].shape[0] != total or json.loads(f.attrs['protocol_json']) != cfg:
                raise ValueError('Partial source has a different protocol.')
            for i in range(int(f.attrs['next_frame']), total):
                split, mi, si, rep = plan[i]
                key = [cfg['data_seed'], {'train': 1, 'val': 2, 'test': 3}[split], mi, si, rep]
                streams = np.random.SeedSequence(key).spawn(4)
                rng, symbol_rng, channel_rng, noise_rng = [np.random.default_rng(s) for s in streams]
                mu, fd0, timing = rng.uniform(-8160, -180), rng.uniform(-500, 500), rng.uniform(-0.5, 0.5)
                gen = LEOSignalGenerator(modulations=list(MODULATIONS), rng=symbol_rng)
                spec = SignalSpec(modulation=MODULATIONS[mi], num_symbols=2048, samples_per_symbol=8,
                    sample_rate_hz=200000, carrier_frequency_hz=30e9, snr_db=SNR[si], fd0_hz=fd0,
                    mu_hz_per_s=mu, rrc_beta=0.35, rrc_span=8, channel_type='rician', rician_k_db=10,
                    phase_noise_std_rad=0.0001, phase_noise_mode='random_walk',
                    fractional_timing_offset_samples=timing, compensate_constant_doppler=False,
                    target_num_samples=16384, pilot_enabled=True, pilot_pattern='comb',
                    pilot_interval_symbols=16, pilot_guard_symbols=0, pilot_modulation='QPSK', pilot_symbol_mode='pn', pilot_seed=2026)
                sample = gen.generate(spec, channel_rng=channel_rng, noise_rng=noise_rng)
                f['iq'][i] = sample.iq
                values = dict(label=sample.label, snr_db=SNR[si], fd0=fd0,
                    fd0_frame_start=sample.info['fd0_frame_start_hz'], fractional_timing_offset_samples=timing,
                    mu=mu, fs=200000, fc=30e9, track_id=i, frame_id=0, frame_time_s=0, domain_id=0,
                    orbit_altitude_km=600, orbit_inclination_deg=53, phase_noise_std_rad=0.0001,
                    modulation=MODULATIONS[mi].encode(), dataset_split=split.encode(), pilot_pattern=b'comb',
                    channel_type=b'rician', pair_key=key, pilot_indices=sample.pilot_indices,
                    pilot_sample_indices=sample.pilot_indices * 8, pilot_symbols=sample.pilot_symbols)
                for k, value in values.items():
                    f[k][i] = value
                if (i + 1) % 100 == 0 or i + 1 == total:
                    f.attrs['next_frame'] = i + 1
                    f.flush()
                    print(f'Generated {i+1}/{total} paired source frames', flush=True)
            f.attrs['complete'] = True
        temporary.replace(source)
    with h5py.File(source, 'r') as f:
        if not f.attrs.get('complete') or len(f['iq']) != total:
            raise ValueError('Incomplete source')
        split_names = f['dataset_split'][:].astype(str)
        splits = {key: np.flatnonzero(split_names == key) for key in ('train', 'val', 'test')}
        np.savez_compressed(p['splits'], **splits)
        for n in LENGTHS:
            raw = paths(n, name)['raw']
            if raw.exists():
                continue
            with h5py.File(raw.with_suffix('.partial.h5'), 'w', libver='latest') as out:
                for key, ds in f.items():
                    shape = list(ds.shape)
                    if key == 'iq': shape[-1] = n
                    elif key in {'pilot_indices', 'pilot_sample_indices', 'pilot_symbols'}: shape[-1] = n // 128
                    layout = h5py.VirtualLayout(tuple(shape), dtype=ds.dtype)
                    src = h5py.VirtualSource(source.name, key, shape=ds.shape)
                    selection = tuple(slice(0, size) for size in shape)
                    layout[:] = src[selection]
                    out.create_virtual_dataset(key, layout)
                out.create_dataset('T', data=np.full(total, n / 200000, dtype='f4'))
                out.create_dataset('gamma', data=(np.pi * np.abs(f['mu'][:]) * (n / 200000)**2).astype('f4'))
                out.create_dataset('num_pilots', data=np.full(total, n // 128, dtype='i8'))
                out.attrs['source'] = source.name
                out.attrs['complete'] = True
            raw.with_suffix('.partial.h5').replace(raw)
    audit_data(name)


def audit_data(name=NAME):
    p = paths(8192, name)
    splits = dict(np.load(p['splits']))
    joined = np.concatenate(list(splits.values()))
    if len(np.unique(joined)) != len(joined):
        raise ValueError('Overlapping source-frame splits')
    with h5py.File(p['data'] / 'source_16384.h5', 'r') as source:
        for n in LENGTHS:
            with h5py.File(paths(n, name)['raw'], 'r') as f:
                assert f['iq'].shape == (len(joined), 2, n)
                np.testing.assert_array_equal(f['label'][:], source['label'][:])
                np.testing.assert_array_equal(f['pair_key'][:], source['pair_key'][:])
                np.testing.assert_allclose(f['T'][:], n / 200000)
                assert f['pilot_indices'].shape[1] == n // 128
                for split, indices in splits.items():
                    for index in (indices[0], indices[len(indices)//2], indices[-1]):
                        np.testing.assert_array_equal(f['iq'][index], source['iq'][index, :, :n])
                assert np.all(f['pilot_sample_indices'][:] < n)
    write_json(p['root'] / 'data_audit.json', dict(passed=True, lengths=list(LENGTHS),
        split_counts={k: len(v) for k, v in splits.items()}, paired_prefixes=True, source_groups_disjoint=True))


def check_orbit_approximation(name=NAME):
    root = paths(8192, name)['root']
    output = root / 'quadratic_phase_geometry_check.csv'
    if output.exists():
        return
    geometry = module('30_validate_leo_orbit_geometry.py')
    rows = []
    for n in LENGTHS:
        for elevation in (20, 30, 35, 50, 65, 80, 90):
            result = geometry._local_quadratic_model_error(carrier_hz=30e9, altitude_m=600e3,
                min_elevation_deg=10, max_elevation_deg=elevation, frame_duration_s=n/200000,
                fine_step_s=0.0001, start_step_s=0.1)
            rows.append(dict(n=n, **result))
        print(f'Checked circular-orbit phase approximation for N={n}', flush=True)
    pd.DataFrame(rows).to_csv(output, index=False)


def run_step(command, log, marker, status_root, **status):
    command = [str(x) for x in command]
    identity = hashlib.sha256(json.dumps(command).encode()).hexdigest()
    if marker.exists():
        if json.loads(marker.read_text())['command_hash'] != identity:
            raise ValueError(f'Completed command changed: {marker}')
        return
    write_json(status_root / 'status.json', dict(state='running', pid=os.getpid(), **status,
        command=command, log=str(log), updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', TQDM_DISABLE='1')
    with log.open('a', encoding='utf-8') as out:
        out.write('\n' + json.dumps(command) + '\n'); out.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        # Existing Windows teardown issue: accept only freshly verified completion.
        if (Path(command[1]).name != '12_train_model.py' or (result.returncode & 0xffffffff) != 0xC0000409
                or not module('16_run_hybrid_dfrft_seeds.py').completed_training_artifacts(command, started)):
            raise RuntimeError(f'Command failed ({result.returncode}); see {log}')
    write_json(marker, dict(command=command, command_hash=identity, elapsed_s=time.time()-started,
        completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))


def prepare_features(n, name, device):
    p = paths(n, name); setting = frontend_settings(n)
    commands = [
        ('pilot', [sys.executable, 'scripts/06_precompute_pilot_mu.py', '--raw-data', p['raw'], '--output', p['pilot'],
            '--estimator-type', 'hybrid_dfrft', '--pilot-weighting', 'coherent', '--timing-phases', '0',
            '--fd0-min', '-550', '--fd0-max', '550', '--fd0-step', '25', '--device', device, '--batch-size', '64',
            '--dfrft-coarse-mu-step', setting['coarse_mu_step'], '--dfrft-fft-size', setting['fft_size'],
            '--dfrft-fine-mu-radius', setting['fine_mu_radius'], '--dfrft-fine-mu-step', setting['fine_mu_step'],
            '--dfrft-fine-fd0-radius', setting['fine_fd0_radius'], '--dfrft-fine-fd0-step', setting['fine_fd0_step'],
            '--unwrap-centered-cfo', '--no-progress', '--overwrite']),
        ('iqcomp', [sys.executable, 'scripts/08_precompute_external_mu_hoc.py', '--raw-data', p['raw'],
            '--mu-feature-data', p['pilot'], '--output', p['feature'], '--fd0-key', 'pilot_fd0_hat',
            '--compensate-fd0', '--save-iq-comp', '--skip-hoc', '--skip-v-hoc', '--device', device, '--no-progress', '--overwrite']),
        ('symbol', [sys.executable, 'scripts/10_precompute_symbol_constellation.py', '--raw-data', p['raw'],
            '--mu-feature-data', p['pilot'], '--output', p['symbol'], '--fd0-key', 'pilot_fd0_hat',
            '--candidate-score-mode', 'exact_mixture', '--snr-source', 'pilot_evm', '--max-symbols', '512',
            '--no-progress', '--overwrite']),
    ]
    for step, command in commands:
        run_step(command, p['result'] / (step + '.log'), p['result'] / (step + '.done.json'), p['root'], stage=step, n=n)
    with h5py.File(p['pilot'], 'r') as f, h5py.File(p['symbol'], 'r') as symbols:
        if not np.all(f['pilot_valid'][:]) or not np.isfinite(symbols['evm_features'][:]).all():
            raise ValueError(f'Invalid receiver features for N={n}')
        if symbols['evm_features'].shape[1] != 48:
            raise ValueError('Expected the full 48-D descriptor')


def train_and_evaluate(n, method, seed, name, device, smoke=False):
    p = paths(n, name); model, config = METHODS[method]
    directory = p['checkpoint'] / method / f'seed_{seed}'
    output = p['result'] / method / f'seed_{seed}_test.csv'
    train = [sys.executable, 'scripts/12_train_model.py', '--raw-data', p['raw'], '--feature-data', p['feature'],
        '--splits', p['splits'], '--model', model, '--model-config', config,
        '--train-config', 'configs/train/train_long.yaml', '--output-dir', directory,
        '--seed', seed, '--device', device, '--num-workers', '0', '--iq-source', 'comp',
        '--iq-representation', 'iq', '--iq-normalize', 'zscore', '--cache-iq', '--no-hoc-standardize']
    if model == 'drc_dualnet': train += ['--symbol-feature-data', p['symbol']]
    if smoke: train += ['--epochs', '1', '--batch-size', '8']
    marker = directory / 'train.done.json'
    # Keep the canonical command hash stable across checkpoint resume.
    if (directory / 'last.pt').exists() and not marker.exists():
        resume = train + ['--resume', directory / 'last.pt']
        run_step(resume, directory / 'runner.log', directory / 'resume.done.json', p['root'], stage='train', n=n, method=method, seed=seed)
        identity = hashlib.sha256(json.dumps([str(x) for x in train]).encode()).hexdigest()
        write_json(marker, dict(command_hash=identity, resumed=True))
    else:
        run_step(train, directory / 'runner.log', marker, p['root'], stage='train', n=n, method=method, seed=seed)
    evaluate = [sys.executable, 'scripts/13_evaluate_model.py', '--checkpoint', directory / 'best.pt',
        '--raw-data', p['raw'], '--feature-data', p['feature'], '--splits', p['splits'], '--split', 'test',
        '--model', model, '--output', output, '--summary-output', output.with_suffix('.summary.json'),
        '--batch-size', '128', '--num-workers', '0', '--device', device, '--strict', '--load-into-memory']
    if model == 'drc_dualnet': evaluate += ['--symbol-feature-data', p['symbol']]
    run_step(evaluate, directory / 'evaluation.log', output.with_suffix('.done.json'), p['root'],
             stage='evaluate', n=n, method=method, seed=seed)
    validate_predictions(output, p)


def validate_predictions(output, p):
    table = pd.read_csv(output).sort_values('index').reset_index(drop=True)
    indices = np.sort(np.load(p['splits'])['test'])
    np.testing.assert_array_equal(table['index'], indices)
    with h5py.File(p['raw'], 'r') as f:
        np.testing.assert_array_equal(table.label, f['label'][indices])
        np.testing.assert_allclose(table.snr_db, f['snr_db'][indices], atol=1e-6, rtol=0)
    correct = (table.label == table.pred).astype(int)
    np.testing.assert_array_equal(table.correct, correct)
    meta = json.loads(output.with_suffix('.summary.json').read_text())
    assert meta['split'] == 'test' and np.isclose(meta['accuracy'], correct.mean())
    return table


def estimation_metrics(n, name):
    p = paths(n, name)
    indices = np.sort(np.load(p['splits'])['test'])
    with h5py.File(p['pilot'], 'r') as f:
        rate = np.asarray(f['mu_error'][indices], dtype=float)
        cfo = np.asarray(f['fd0_error'][indices], dtype=float)
        snr = f['snr_db'][indices]
    duration = n / 200000
    bound = 2 * np.pi * np.abs(cfo) * duration + np.pi * np.abs(rate) * duration**2
    table = pd.DataFrame(dict(index=indices, snr_db=snr, rate_error_hz_s=rate, cfo_error_hz=cfo, b_phi=bound))
    table.to_csv(p['result'] / 'estimation_test.csv', index=False)
    return dict(n=n, duration_ms=duration*1000, pilots=n//128,
        rate_p95_hz_s=float(np.quantile(np.abs(rate), .95)), cfo_p95_hz=float(np.quantile(np.abs(cfo), .95)),
        b_phi_p95_rad=float(np.quantile(bound, .95)))


def diagnostic_and_profile(n, name, device, seeds=SEEDS):
    """Rebuild oracle/no-comp descriptors on test frames; time full CPU receiver on validation."""
    import torch
    from src.datasets.feature_dataset import _preprocess_iq
    from src.drc_hoc.hybrid_dfrft_estimator import HybridDFRFTConfig, HybridDFRFTPilotEstimator
    from src.drc_hoc.pilot_estimator import PilotMuEstimatorConfig
    from src.signal.modulation import get_constellation
    from src.signal.pulse_shape import rrc_filter
    p = paths(n, name)
    helper = module('32_validate_feature_degradation.py')
    symbols = helper._load_symbol_helpers()
    taps = rrc_filter(beta=.35, span=8, sps=8)
    constellations = {m: get_constellation(m) for m in MODULATIONS}
    splits = np.load(p['splits']); test = np.sort(splits['test'])
    reference = p['checkpoint'] / 'proposed' / f'seed_{seeds[0]}' / 'best.pt'
    cpu_model, cfg, model_cfg = helper._load_fixed_classifier(reference, num_classes=7, hoc_dim=21, evm_dim=48, device=torch.device('cpu'))
    symbol_cfg = helper._resolve_symbol_config(cfg)
    def descriptor(frame, raw, index):
        return helper._build_evm_features_for_frame(frame, pilot_indices=raw['pilot_indices'][index],
            pilot_symbols=raw['pilot_symbols'][index], symbol_helpers=symbols, taps=taps, sps=8,
            modulations=MODULATIONS, constellations=constellations, config=symbol_cfg)[0]
    diagnostics = p['result'] / 'frozen_diagnostics.h5'
    if not diagnostics.exists():
        temporary = diagnostics.with_suffix('.partial.h5')
        with h5py.File(p['raw'], 'r') as raw, h5py.File(temporary, 'w') as out:
            out.create_dataset('indices', data=test)
            t = np.arange(n) / 200000
            for mode in ('no_comp', 'oracle'):
                group = out.create_group(mode)
                iq = group.create_dataset('iq', shape=(len(test), 2, n), dtype='f4', compression='lzf')
                evm = group.create_dataset('evm', shape=(len(test), 48), dtype='f4')
                for row, index in enumerate(test):
                    v = raw['iq'][index]; z = v[0].astype(float) + 1j*v[1].astype(float)
                    if mode == 'oracle':
                        z *= np.exp(-1j*(2*np.pi*float(raw['fd0'][index])*t + np.pi*float(raw['mu'][index])*t*t))
                    iq[row] = _preprocess_iq(np.stack((z.real,z.imag)).astype('f4'), 'iq', 'zscore')
                    evm[row] = descriptor(z, raw, index)
        temporary.replace(diagnostics)
    with h5py.File(diagnostics, 'r') as diag, h5py.File(p['raw'], 'r') as raw:
        for mode in ('no_comp', 'oracle'):
            iq, evm = diag[mode]['iq'][:], diag[mode]['evm'][:]
            for seed in seeds:
                output = p['result'] / 'proposed' / f'seed_{seed}_{mode}.csv'
                if output.exists(): continue
                model, run_cfg, mc = helper._load_fixed_classifier(p['checkpoint']/'proposed'/f'seed_{seed}'/'best.pt', num_classes=7, hoc_dim=21, evm_dim=48, device=torch.device(device))
                normalized = helper._standardize_evm(evm, run_cfg.get('evm_stats'))
                pred = helper._predict_fixed_classifier(model, mc, iq, normalized, device=torch.device(device), batch_size=64)
                labels = raw['label'][test]
                pd.DataFrame(dict(index=test, label=labels, pred=pred, snr_db=raw['snr_db'][test], correct=(pred==labels).astype(int))).to_csv(output,index=False)
                del model
            del iq, evm
    timing_path = p['result'] / 'cpu_receiver_timing.json'
    if timing_path.exists(): return
    torch.set_num_threads(1)
    setting = frontend_settings(n)
    estimator = HybridDFRFTPilotEstimator(PilotMuEstimatorConfig(sample_rate_hz=200000, samples_per_symbol=8,
        fd0_min_hz=-550, fd0_max_hz=550, pilot_weighting='coherent', phase_only=False, timing_phases=(0,)),
        HybridDFRFTConfig(coarse_mu_step_hz_per_s=setting['coarse_mu_step'], fft_size=setting['fft_size'],
        fine_mu_radius_hz_per_s=setting['fine_mu_radius'], fine_mu_step_hz_per_s=setting['fine_mu_step'],
        fine_fd0_radius_hz=setting['fine_fd0_radius'], fine_fd0_step_hz=setting['fine_fd0_step'], unwrap_centered_cfo=True))
    with h5py.File(p['raw'],'r') as raw:
        val = np.sort(splits['val']); labels=raw['label'][val]; snrs=raw['snr_db'][val]
        selected=[int(val[np.flatnonzero((labels==c)&(snrs==s))[0]]) for c in range(7) for s in SNR]
        # Input is in memory before the clock; times include estimation,
        # compensation, descriptor, normalization and batch-one CPU inference.
        frames=[(i,raw['iq'][i],raw['pilot_indices'][i],raw['pilot_symbols'][i]) for i in selected]
        time_axis=np.arange(n)/200000
        elapsed=[]; inference=[]
        with torch.inference_mode():
            for iteration in range(10+len(frames)*3):
                index,v,pi,ps=frames[iteration%len(frames)]
                start=time.perf_counter()
                est=estimator.estimate(v,pi,ps).estimate
                z=(v[0].astype(float)+1j*v[1].astype(float))*np.exp(-1j*(2*np.pi*est.fd0_hat_hz*time_axis+np.pi*est.mu_hat_hz_per_s*time_axis**2))
                features=helper._build_evm_features_for_frame(z,pilot_indices=pi,pilot_symbols=ps,symbol_helpers=symbols,
                    taps=taps,sps=8,modulations=MODULATIONS,constellations=constellations,config=symbol_cfg)[0]
                iq=torch.from_numpy(_preprocess_iq(np.stack((z.real,z.imag)).astype('f4'),'iq','zscore')[None])
                evm=torch.from_numpy(helper._standardize_evm(features,cfg.get('evm_stats'))[None])
                network_start=time.perf_counter(); cpu_model(iq,None,None,evm_features=evm)
                stop=time.perf_counter()
                if iteration>=10:
                    elapsed.append((stop-start)*1000); inference.append((stop-network_start)*1000)
    write_json(timing_path,dict(cpu_threads=1,frames=len(frames),repetitions=3,seed=seeds[0],
        e2e_median_ms=float(np.median(elapsed)),e2e_p95_ms=float(np.quantile(elapsed,.95)),
        inference_median_ms=float(np.median(inference)),includes_disk_io=False,includes_observation_wait=False,
        device='CPU',processor=os.environ.get('PROCESSOR_IDENTIFIER','unknown')))


def completeness(name=NAME, seeds=SEEDS):
    cells=[]
    for n in LENGTHS:
        p=paths(n,name)
        for method in METHODS:
            for seed in seeds:
                output=p['result']/method/f'seed_{seed}_test.csv'
                complete=output.exists() and output.with_suffix('.summary.json').exists() and output.with_suffix('.done.json').exists()
                cells.append(dict(n=n,method=method,seed=seed,complete=complete))
    payload=dict(complete=sum(c['complete'] for c in cells),expected=len(cells),cells=cells)
    write_json(paths(8192,name)['root']/'completeness.json',payload)
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default=NAME)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--stages',nargs='+',choices=['data','features','train','diagnostics','report'],
                        default=['data','features','train','diagnostics','report'])
    parser.add_argument('--smoke',action='store_true',help='Isolated one-epoch integration run; never publishes paper results.')
    args=parser.parse_args()
    if Path(args.name).name!=args.name or args.name in {'.','..'}: parser.error('Campaign name must be a single directory name.')
    if args.smoke and args.name==NAME: args.name=NAME+'_smoke'
    if not args.smoke and args.name.endswith('_smoke'): parser.error('Smoke name requires --smoke.')
    os.chdir(ROOT)
    root=paths(8192,args.name)['root']; root.mkdir(parents=True,exist_ok=True)
    lock=root/'campaign.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'Campaign lock exists: {lock}; check the recorded PID before removing a stale lock.')
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    seeds=(41,) if args.smoke else SEEDS
    try:
        if 'data' in args.stages: prepare_source(args.name,args.smoke)
        if 'features' in args.stages and not args.smoke:
            write_json(root/'status.json',dict(state='running',pid=os.getpid(),stage='geometry_check'))
            check_orbit_approximation(args.name)
        for n in LENGTHS:
            if 'features' in args.stages: prepare_features(n,args.name,args.device)
            if 'train' in args.stages:
                for seed in seeds:
                    for method in METHODS:
                        train_and_evaluate(n,method,seed,args.name,args.device,args.smoke)
                        completeness(args.name,seeds)
            if 'diagnostics' in args.stages:
                write_json(root/'status.json',dict(state='running',pid=os.getpid(),stage='diagnostics',n=n))
                diagnostic_and_profile(n,args.name,args.device,seeds)
        if 'report' in args.stages and not args.smoke:
            from src.experiments.frame_length_report import publish
            publish(args.name)
        write_json(root/'status.json',dict(state='complete' if 'report' in args.stages and not args.smoke else 'stages_complete',pid=os.getpid(),stages=args.stages,smoke=args.smoke,
                   updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    except BaseException as error:
        previous=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
        write_json(root/'status.json',dict(previous,state='failed',error=repr(error)))
        raise
    finally:
        lock.unlink(missing_ok=True)

