# Practical-extension protocol

The current paper includes the joint-impairment protocol below. Its receiver
uses DCFT-based chirp-focus acquisition with coherent local refinement.
The `hybrid_dfrft` estimator option and historical script names are retained
for compatibility; the training and evaluation runner is
[`scripts/17_run_hybrid_dfrft_seeds.py`](../scripts/17_run_hybrid_dfrft_seeds.py).

## Joint-impairment split

Generate class/SNR-balanced train, validation, and test data with simultaneous
residual CFO, fractional timing, flat Rician fading, and oscillator phase
noise:

```powershell
& .\.venv\Scripts\python.exe scripts\03_generate_snr_balanced_eval_dataset.py `
  --output data\processed\leo_7mods_joint_practical.h5 `
  --splits-output data\splits\leo_7mods_joint_practical_splits.npz `
  --snr-db -10 -7 -4 -1 2 5 8 11 14 17 20 `
  --train-samples-per-mod-snr 200 `
  --val-samples-per-mod-snr 100 `
  --test-samples-per-mod-snr 100 `
  --fd0-min-hz -500 --fd0-max-hz 500 `
  --fractional-timing-min-samples -0.5 `
  --fractional-timing-max-samples 0.5 `
  --channel-type rician --rician-k-db 10 `
  --phase-noise-std-rad 0.0001 --phase-noise-mode random_walk
```

Use `scripts/06_precompute_pilot_mu.py --estimator-type hybrid_dfrft` with joint CFO search bounds covering
the generated interval, then rebuild compensated I/Q and all-candidate EVM
features with scripts 08 and 10. Train on the new train split, select all
hyperparameters on validation only, and report the untouched test split. A
frozen AWGN checkpoint evaluation is a mismatch stress test, not a headline
robustness result.

## Exact-orbit pass holdout

Generate pass-structured data with:

```powershell
& .\.venv\Scripts\python.exe scripts\02_generate_track_dataset.py `
  --config configs\dataset\leo_orbital_elevation_ood_four_way.yaml `
  --output data\processed\leo_orbital_elevation_ood_four_way.h5 `
  --splits-output data\splits\leo_orbital_elevation_ood_four_way_splits.npz
```

The four-way split holds out complete maximum-elevation groups: 20/30/45 degrees
for training, 60 degrees for validation, 75 degrees for calibration, and
90 degrees for testing, as defined in
[`leo_orbital_elevation_ood_four_way.yaml`](../configs/dataset/leo_orbital_elevation_ood_four_way.yaml).
Use validation for model selection and keep calibration separate from training
and final test evaluation. The stored
`track_id`, `maximum_elevation_deg`, and `trajectory_mode` fields must be
audited before training. Report both frame-level accuracy and macro-average
accuracy over passes with a pass bootstrap confidence interval.

## Provenance rules

- Use one selected checkpoint for every channel condition in a comparison.
- Pair modulation, SNR, Doppler rate, and random seed across channel variants.
- Use equal test counts when quoting raw percentage-point differences.
- Preserve every evaluator `.summary.json`; it records the checkpoint path.
