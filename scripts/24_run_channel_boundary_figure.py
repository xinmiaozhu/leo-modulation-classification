"""Run only the seven published curves and replace the figure after validation."""
from pathlib import Path
import argparse
import json
import runpy
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import pandas as pd
from src.plotting.channel_boundary import CURVES, aggregate_curves, plot_channel_boundary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiment/channel_boundary_figure.json')
    parser.add_argument('--output-dir', default='outputs/channel_boundary_flat_multipath')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-publish', action='store_true', help='Validate a small run without replacing paper artifacts')
    parser.add_argument('--checkpoint-source', help='Reuse a completed run; generate only test data and never train')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ('logs', 'checkpoints', 'predictions'):
        (out / name).mkdir(exist_ok=True)
    runner = runpy.run_path(str(ROOT / 'scripts/22_run_channel_boundary.py'))
    checkpoint_root = out
    checkpoint_record = None
    if args.checkpoint_source:
        checkpoint_root = Path(args.checkpoint_source).resolve()
        original = json.loads((checkpoint_root / 'manifest.json').read_text())['config']
        for key in ('profiles', 'snr_db', 'data_seed', 'model_seeds', 'training_conditions',
                    'train_per_class_snr', 'val_per_class_snr'):
            if config[key] != original[key]:
                raise ValueError(f'Checkpoint protocol mismatch: {key}')
        if not config.get('evaluation_only'):
            raise ValueError('Checkpoint reuse requires evaluation_only=true')
        hashes = {}
        for source in config['training_conditions']:
            for seed in config['model_seeds']:
                checkpoint = checkpoint_root / 'checkpoints' / f'{source}_{seed}' / 'best.pt'
                if not checkpoint.with_name('train_result.json').exists():
                    raise ValueError(f'Training incomplete: {checkpoint}')
                hashes[str(checkpoint)] = runner['digest'](checkpoint)
        checkpoint_record = dict(directory=str(checkpoint_root), sha256=hashes)
    elif config.get('evaluation_only'):
        raise ValueError('Evaluation-only runs require --checkpoint-source')
    manifest = out / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['config'] != config:
        raise ValueError('Changed protocol requires a new output directory')
    if manifest.exists() and json.loads(manifest.read_text()).get('checkpoint_source') != checkpoint_record:
        raise ValueError('Checkpoint source changed for existing run')
    if not manifest.exists():
        manifest.write_text(json.dumps({'config': config, 'device': args.device,
            'checkpoint_source': checkpoint_record,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'source_sha256': {str(p.relative_to(ROOT)): runner['digest'](p)
                for p in [*ROOT.glob('src/**/*.py'), *ROOT.glob('scripts/*.py')]},
            'flat_model': 'Three coincident specular paths; paired coefficients with reference',
            'snr': 'Per-frame received-power AWGN',
            'carrier': 'Estimated common carrier, explicit CFO compensation'}, indent=2))
    try:
        runner['prepare'](config, out, args.device)
        reference = None
        test_labels = {}
        for condition in config['profiles']:
            folder, raw_path, splits_path = runner['locations'](out, condition)
            split = np.load(splits_path)
            for a, b in [('train','val'),('train','test'),('val','test')]:
                assert not np.intersect1d(split[a], split[b]).size
            with h5py.File(raw_path) as raw:
                idx = split['test']
                current = (raw['pair_key'][idx], raw['label'][idx], raw['snr_db'][idx])
                if len(np.unique(current[0], axis=0)) != len(idx):
                    raise ValueError('Duplicate test sample keys')
                counts = pd.DataFrame({'label': current[1], 'snr': current[2]}).groupby(['label', 'snr']).size()
                if not (counts == config['test_per_class_snr']).all():
                    raise ValueError('Incorrect per-class/SNR test sample count')
                if reference is None:
                    reference = current
                for left, right in zip(reference, current):
                    np.testing.assert_array_equal(left, right)
                test_labels[condition] = current[1:]
        if not args.checkpoint_source:
            runner['train'](config, out, args.device)
        rows = []
        for source, target, *_ in CURVES:
            folder, raw, splits = runner['locations'](out, target)
            for seed in config['model_seeds']:
                name = f'{source}_{seed}__{target}__unequalized'
                output = out / 'predictions' / f'{name}.csv'
                arguments = runner['model_arguments'](folder, raw, splits, 'unequalized') + [
                    '--checkpoint', checkpoint_root / 'checkpoints' / f'{source}_{seed}' / 'best.pt',
                    '--split', 'test', '--device', args.device, '--batch-size', config['batch_size'],
                    '--output', output, '--summary-output', output.with_suffix('.summary.json')]
                runner['run_step'](f'eval_{name}', '14_evaluate_model.py', arguments, out, config['threads'])
                table = pd.read_csv(output)
                np.testing.assert_array_equal(table.label, test_labels[target][0])
                np.testing.assert_array_equal(table.snr_db, test_labels[target][1])
                for snr, group in table.groupby('snr_db'):
                    rows.append(dict(train_condition=source, test_condition=target, receiver='unequalized',
                        seed=seed, snr_db=snr, n=len(group), accuracy=group.correct.mean()))
        neural = pd.DataFrame(rows)
        aggregate_curves(neural)
        neural.to_csv(out / 'neural_accuracy.csv', index=False)
        name = 'channel_boundary_representative_accuracy_vs_snr'
        staged = out / f'{name}.pdf'
        plot_channel_boundary(neural, staged)
        metadata_path = staged.with_suffix('.json')
        metadata = json.loads(metadata_path.read_text())
        metadata.update(flat_model='Three coincident random-phase paths, relative powers [0,-6,-10] dB',
                        test_per_class_snr=config['test_per_class_snr'],
                        checkpoint_source=checkpoint_record,
                        snr_definition='Per-frame received SNR', experiment_manifest=str(manifest))
        metadata_path.write_text(json.dumps(metadata, indent=2))
        if args.no_publish:
            runner['write_progress'](out, state='complete', figure=str(staged), published=False)
            return
        destination = ROOT / 'outputs/figures/paper'
        backup = out / 'previous_figure'
        backup.mkdir(exist_ok=True)
        for suffix in ('.pdf', '.png', '.csv', '.json'):
            target = destination / (name + suffix)
            if target.exists() and not (backup / target.name).exists():
                shutil.copy2(target, backup / target.name)
            shutil.copy2(staged.with_suffix(suffix), target)
        runner['write_progress'](out, state='complete', figure=str(destination / (name + '.pdf')))
    except Exception as exc:
        runner['write_progress'](out, state='failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
