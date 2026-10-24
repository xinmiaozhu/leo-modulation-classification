# Protocol B: DCFT receiver (historical hybrid_dfrft paths)

Run from the repository root after the installation in [README](README.md).
This guide uses DCFT acquisition followed by local coherent refinement and script 17, whose inputs are
under `data/features/hybrid_dfrft/`.

## Data

Generate the joint-impairment dataset with the first command block in
[Practical protocol](experiments/PRACTICAL_PROTOCOL.md#joint-impairment-split).
It creates `data/processed/leo_7mods_joint_practical.h5` and
`data/splits/leo_7mods_joint_practical_splits.npz`: 200/100/100 samples per
modulation and SNR for train/validation/test. Do not use the independent
Protocol A test dataset here.

## Pilot estimation, compensation, and descriptor

These pilot search settings match the existing promoted-front-end summary:
coherent pilot weighting, timing phase 0, CFO bounds -550 to 550 Hz,
DCFT coarse rate step 160 Hz/s, FFT size 2048, and local refinement.

```powershell
python scripts/06_precompute_pilot_mu.py --raw-data data/processed/leo_7mods_joint_practical.h5 --output data/features/hybrid_dfrft/protocol_b_pilot_hybrid.h5 --estimator-type hybrid_dfrft --pilot-weighting coherent --timing-phases 0 --fd0-min -550 --fd0-max 550 --fd0-step 25 --fine-fd0-radius 30 --fine-fd0-step 1 --dfrft-coarse-mu-step 160 --dfrft-fft-size 2048 --dfrft-fine-mu-radius 200 --dfrft-fine-mu-step 5 --dfrft-fine-fd0-radius 30 --dfrft-fine-fd0-step 1
python scripts/08_precompute_external_mu_hoc.py --raw-data data/processed/leo_7mods_joint_practical.h5 --mu-feature-data data/features/hybrid_dfrft/protocol_b_pilot_hybrid.h5 --output data/features/hybrid_dfrft/protocol_b_iqcomp.h5 --fd0-key pilot_fd0_hat --compensate-fd0 --save-iq-comp --skip-hoc --skip-v-hoc
python scripts/10_precompute_symbol_constellation.py --raw-data data/processed/leo_7mods_joint_practical.h5 --mu-feature-data data/features/hybrid_dfrft/protocol_b_pilot_hybrid.h5 --output data/features/hybrid_dfrft/protocol_b_symbol_exact.h5 --fd0-key pilot_fd0_hat --candidate-score-mode exact_mixture --snr-source pilot_evm
```

The I/Q baselines, including IQFormer, consume the frozen compensated I/Q.
IQFormer constructs its STFT internally and does not use HOC placeholders or
the proposed receiver's constellation descriptor. Preserve preprocessing
metadata alongside the results.

## Training and test evaluation

Start with one seed:

```powershell
python scripts/16_run_hybrid_dfrft_seeds.py --methods proposed --seeds 41 --device cuda
```

Run the full comparison:

```powershell
python scripts/16_run_hybrid_dfrft_seeds.py --seeds 41 73 107 149 211 --device cuda
```

Use `--device cpu` if needed. Training uses `configs/train/train_long.yaml`.
The runner resumes incomplete training and skips completed runs. Checkpoints
are written under `outputs/checkpoints/hybrid_dfrft/protocol_b/`; test predictions,
`seed_runs.csv`, `seed_summary.csv`, and paired-difference `summary.json` are
written under `outputs/results/hybrid_dfrft/protocol_b/`.

To complete the four missing baselines for the seed-41 figure:

```powershell
python scripts/16_run_hybrid_dfrft_seeds.py --methods cnn2 cnn_lstm_dual satellite_cnn iqformer --seeds 41 --device cuda
python scripts/47_plot_protocol_b_accuracy_vs_snr.py --seed 41
```

CNN2 and IQCNet use z-score-normalized compensated I/Q; CNN-LSTM uses power
normalization, matching Protocol A. IQFormer uses z-score-normalized I/Q and
its official real-STFT representation. All four models select the best
checkpoint on validation data and evaluate only the Protocol B test split.
The plot requires all seven methods at the selected seed, checks that their
sample indices, labels and SNRs match, and writes PDF, PNG, SVG and source tables.
Summary rebuilding includes previously completed methods and seeds, so running
only the missing baselines preserves the existing comparison rows.

The main experiment runners are Protocol A Proposed/MCNet (15), STARNet (16),
and the final Protocol A/B external baselines (17). Protocol A/B comparison plots are 47/48.

To evaluate existing checkpoints and rebuild summaries:

```powershell
python scripts/16_run_hybrid_dfrft_seeds.py --stages eval summary --device cuda
```

After preparing the input files, append `--dry-run` to inspect commands without
training or evaluating. Hyperparameters must be selected on validation data;
the test split is reserved for final reporting. For representation ablations,
see `scripts/18_run_hybrid_dfrft_representation_ablation.py --help`.

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
