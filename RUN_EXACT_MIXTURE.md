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
python scripts/08_precompute_external_mu_hoc.py --raw-data data/processed/leo_7mods_snr_balanced_trainval_pilot.h5 --mu-feature-data data/features/exact_mixture/protocol_a_trainval_pilot.h5 --output data/features/leo_7mods_snr_balanced_trainval_hoc_iqcomp_coherent.h5 --save-iq-comp --skip-hoc --skip-v-hoc
python scripts/10_precompute_symbol_constellation.py --raw-data data/processed/leo_7mods_snr_balanced_trainval_pilot.h5 --mu-feature-data data/features/exact_mixture/protocol_a_trainval_pilot.h5 --output data/features/exact_mixture/protocol_a_trainval_symbol_exact.h5 --candidate-score-mode exact_mixture --snr-source pilot_evm
python scripts/06_precompute_pilot_mu.py --raw-data data/processed/leo_7mods_snr_balanced_independent_test_pilot.h5 --output data/features/exact_mixture/protocol_a_test_pilot.h5 --estimator-type coherent_grid --pilot-weighting coherent
python scripts/08_precompute_external_mu_hoc.py --raw-data data/processed/leo_7mods_snr_balanced_independent_test_pilot.h5 --mu-feature-data data/features/exact_mixture/protocol_a_test_pilot.h5 --output data/features/leo_7mods_snr_balanced_independent_test_hoc_iqcomp_coherent.h5 --save-iq-comp --skip-hoc --skip-v-hoc
python scripts/10_precompute_symbol_constellation.py --raw-data data/processed/leo_7mods_snr_balanced_independent_test_pilot.h5 --mu-feature-data data/features/exact_mixture/protocol_a_test_pilot.h5 --output data/features/exact_mixture/protocol_a_test_symbol_exact.h5 --candidate-score-mode exact_mixture --snr-source pilot_evm
```

## Training and independent evaluation

Start with one proposed-model seed:

```powershell
python scripts/14_run_exact_mixture_seeds.py --protocols A --methods proposed --seeds 41 --device cuda
```

Run the five-seed proposed/MCNet comparison:

```powershell
python scripts/14_run_exact_mixture_seeds.py --protocols A --methods proposed mcnet --seeds 41 73 107 149 211 --device cuda
```

Use `--device cpu` for CPU training. The runner uses
`configs/train/train_long.yaml`, resumes incomplete training from `last.pt`,
and skips completed runs. Outputs go to `outputs/checkpoints/exact_mixture/protocol_a/`
and `outputs/results/exact_mixture/protocol_a/`. The latter contains per-seed test
predictions, `seed_runs.csv`, `seed_summary.csv`, and `summary.json`.
To evaluate existing checkpoints and rebuild summaries:

```powershell
python scripts/14_run_exact_mixture_seeds.py --protocols A --methods proposed mcnet --stages eval summary --device cuda
```

Protocol B in script 15 is the older path. Use [RUN_HYBRID_DFRFT.md](RUN_HYBRID_DFRFT.md)
for the promoted Protocol B front end. STARNet has a separate Protocol A runner,
`scripts/15_run_starnet_seeds.py`; the two-method command above does not run it.

## Matched added-HOC ablation

Use the existing final A/B runner to add I/Q + full 48-D descriptor + HOC:

```powershell
python scripts/17_run_representation_ablation_seeds.py --protocols A B --methods fusion_hoc --device cuda
```

The default seeds are 41, 73, 107, 149, 211. The runner uses the unchanged
`train_long.yaml` settings (up to 100 epochs, validation early stopping),
selects `best.pt` on validation accuracy, and evaluates only the final test
split. Existing single-stream/fusion predictions are retained and included
when rebuilding summaries; archived fusion checkpoints are not required.

The `features` stage computes real 21-D HOC from each frozen `iq_comp` array:
C20, C21, C40, C41, C42, C60, C63, each represented by real, imaginary and
absolute parts after centering and unit-power normalization. HDF5 sidecars
ending in `_hoc_ablation.h5` link the original I/Q and metadata without
changing them. HOC uses signed-log1p and training-split standardization;
evaluation reuses the checkpoint statistics. Zero placeholder HOC is rejected.
The HOC MLP uses hidden dimensions (128, 128) and a 128-D output concatenated
with the original two streams. The default receiver remains two-stream.

The existing model configuration is reused with `--use-hoc-stream` at training;
evaluation restores the model configuration from the checkpoint. No separate
model YAML or code file is needed. Protocol B's historical `hybrid_dfrft`
directory denotes DCFT acquisition plus local coherent refinement.

Results are written alongside the original representation ablations under
`fusion_hoc/seed_<seed>_test.csv`. `seed_summary.csv` reports sample SD;
`paired_hoc_gain.csv` and `summary.json` report per-seed percentage-point gains
against the original fusion receiver. `formal_five_seed_complete` is true
only when all registered variants have the five specified seeds. Resume with the
same command; add `--dry-run` to inspect commands. The original three-method
plot remains available; add `--include-hoc` to script 49 after both protocols
have completed five-seed results.

## I/Q + HOC auxiliary-representation control

```powershell
python scripts/17_run_representation_ablation_seeds.py --protocols A B --methods iq_hoc --device cuda
```

This control consumes compensated I/Q and the same real 21-D HOC sidecars,
with the constellation descriptor branch disabled. Its HOC encoder has hidden
widths (64, 64) and a 128-D output, matching the descriptor encoder in the
original fusion model. Both two-stream models use the same I/Q encoder,
256-D concatenation, 64-unit fusion MLP, optimizer, splits, and five seeds.
The input dimensions differ (21 HOC versus 48 descriptor features), so total
parameter counts are not exactly equal. The earlier `fusion_hoc` experiment
retains its original (128, 128) HOC hidden widths and results.

The existing model YAML is reused through `--use-hoc-stream --hoc-hidden-dims
64 64`; evaluation restores these settings from the checkpoint. The symbol
file remains available for feature-dimension metadata and evaluation diagnostics,
but its descriptor values are neither loaded as model inputs nor encoded by
this control. HOC statistics are fitted on the training split only.

Predictions appear under `iq_hoc/seed_<seed>_test.csv` in each final protocol's
representation-ablation directory. Summary rebuilding retains all existing
methods. `paired_descriptor_vs_hoc.csv` reports, seed by seed,
`100 * (accuracy(fusion) - accuracy(iq_hoc))`: positive values favor the
48-D descriptor. `paired_iq_hoc_gain.csv` reports the gain of I/Q + HOC over
I/Q-only. Each aggregate includes mean, sample SD and positive-seed count.
`formal_five_seed_complete` now requires all five variants to have the five
specified seeds. Do not report new scores before training and testing finish.

## IQFormer comparison

Run IQFormer on the final Protocol-A data and matched five seeds:

```powershell
python scripts/16_run_hybrid_dfrft_seeds.py --protocols A --methods iqformer --seeds 41 73 107 149 211 --device cuda --report
```

See README for the official architecture, STFT verification, window policy,
common training settings, output tables, and computational-cost conventions.
