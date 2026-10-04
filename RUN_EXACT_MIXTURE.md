# Protocol A: exact-mixture receiver

Run these PowerShell commands from the repository root after installing
`python -m pip install -e ".[dev]"` as described in [README](README.md).
The commands regenerate data and features; full training is computationally expensive.
Existing outputs are protected by the preprocessing scripts unless `--overwrite` is used.

## Data

The following example uses distinct generation seeds for train/validation and
independent test data. These are explicit regeneration settings, not a claim
of byte-for-byte identity with historical paper datasets. Exact historical
reproduction requires the original data, splits, and generation metadata.

```powershell
python scripts/03_generate_snr_balanced_eval_dataset.py --output data/processed/leo_7mods_snr_balanced_trainval_pilot.h5 --splits-output data/splits/leo_7mods_snr_balanced_trainval_pilot_splits.npz --snr-db -10 -7 -4 -1 2 5 8 11 14 17 20 --train-samples-per-mod-snr 200 --val-samples-per-mod-snr 100 --test-samples-per-mod-snr 0 --seed 2027
python scripts/03_generate_snr_balanced_eval_dataset.py --output data/processed/leo_7mods_snr_balanced_independent_test_pilot.h5 --splits-output data/splits/leo_7mods_snr_balanced_independent_test_pilot_splits.npz --snr-db -10 -7 -4 -1 2 5 8 11 14 17 20 --samples-per-mod-snr 100 --test-only --seed 2028
```

## Pilot estimation and features

Run all three commands for each dataset. The feature filenames retain historical
`hoc` naming for compatibility; the proposed model uses I/Q and the constellation
descriptor. Zero HOC placeholders below are unsuitable for HOC-based baselines.

```powershell
python scripts/06_precompute_pilot_mu.py --raw-data data/processed/leo_7mods_snr_balanced_trainval_pilot.h5 --output data/features/exact_mixture/protocol_a_trainval_pilot.h5 --estimator-type coherent_grid --pilot-weighting coherent
python scripts/09_precompute_external_mu_hoc.py --raw-data data/processed/leo_7mods_snr_balanced_trainval_pilot.h5 --mu-feature-data data/features/exact_mixture/protocol_a_trainval_pilot.h5 --output data/features/leo_7mods_snr_balanced_trainval_hoc_iqcomp_coherent.h5 --save-iq-comp --skip-hoc --skip-v-hoc
python scripts/10_precompute_symbol_constellation.py --raw-data data/processed/leo_7mods_snr_balanced_trainval_pilot.h5 --mu-feature-data data/features/exact_mixture/protocol_a_trainval_pilot.h5 --output data/features/exact_mixture/protocol_a_trainval_symbol_exact.h5 --candidate-score-mode exact_mixture --snr-source pilot_evm
python scripts/06_precompute_pilot_mu.py --raw-data data/processed/leo_7mods_snr_balanced_independent_test_pilot.h5 --output data/features/exact_mixture/protocol_a_test_pilot.h5 --estimator-type coherent_grid --pilot-weighting coherent
python scripts/09_precompute_external_mu_hoc.py --raw-data data/processed/leo_7mods_snr_balanced_independent_test_pilot.h5 --mu-feature-data data/features/exact_mixture/protocol_a_test_pilot.h5 --output data/features/leo_7mods_snr_balanced_independent_test_hoc_iqcomp_coherent.h5 --save-iq-comp --skip-hoc --skip-v-hoc
python scripts/10_precompute_symbol_constellation.py --raw-data data/processed/leo_7mods_snr_balanced_independent_test_pilot.h5 --mu-feature-data data/features/exact_mixture/protocol_a_test_pilot.h5 --output data/features/exact_mixture/protocol_a_test_symbol_exact.h5 --candidate-score-mode exact_mixture --snr-source pilot_evm
```

## Training and independent evaluation

Start with one proposed-model seed:

```powershell
python scripts/17_run_exact_mixture_seeds.py --protocols A --methods proposed --seeds 41 --device cuda
```

Run the five-seed proposed/MCNet comparison:

```powershell
python scripts/17_run_exact_mixture_seeds.py --protocols A --methods proposed mcnet --seeds 41 73 107 149 211 --device cuda
```

Use `--device cpu` for CPU training. The runner uses
`configs/train/train_long.yaml`, resumes incomplete training from `last.pt`,
and skips completed runs. Outputs go to `outputs/checkpoints/exact_mixture/protocol_a/`
and `outputs/results/exact_mixture/protocol_a/`. The latter contains per-seed test
predictions, `seed_runs.csv`, `seed_summary.csv`, and `summary.json`.
To evaluate existing checkpoints and rebuild summaries:

```powershell
python scripts/17_run_exact_mixture_seeds.py --protocols A --methods proposed mcnet --stages eval summary --device cuda
```

Protocol B in script 17 is the older path. Use [RUN_HYBRID_DFRFT.md](RUN_HYBRID_DFRFT.md)
for the promoted Protocol B front end. STARNet has a separate Protocol A runner,
`scripts/16_run_starnet_seeds.py`; the two-method command above does not run it.
