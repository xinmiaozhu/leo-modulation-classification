# Protocol B: hybrid DFRFT receiver

Run from the repository root after the installation in [README](README.md).
This guide uses the promoted hybrid front end and script 18, whose inputs are
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
DFRFT coarse rate step 160 Hz/s, FFT size 2048, and local refinement.

```powershell
python scripts/06_precompute_pilot_mu.py --raw-data data/processed/leo_7mods_joint_practical.h5 --output data/features/hybrid_dfrft/protocol_b_pilot_hybrid.h5 --estimator-type hybrid_dfrft --pilot-weighting coherent --timing-phases 0 --fd0-min -550 --fd0-max 550 --fd0-step 25 --fine-fd0-radius 30 --fine-fd0-step 1 --dfrft-coarse-mu-step 160 --dfrft-fft-size 2048 --dfrft-fine-mu-radius 200 --dfrft-fine-mu-step 5 --dfrft-fine-fd0-radius 30 --dfrft-fine-fd0-step 1
python scripts/09_precompute_external_mu_hoc.py --raw-data data/processed/leo_7mods_joint_practical.h5 --mu-feature-data data/features/hybrid_dfrft/protocol_b_pilot_hybrid.h5 --output data/features/hybrid_dfrft/protocol_b_iqcomp.h5 --fd0-key pilot_fd0_hat --compensate-fd0 --save-iq-comp --skip-hoc --skip-v-hoc
python scripts/10_precompute_symbol_constellation.py --raw-data data/processed/leo_7mods_joint_practical.h5 --mu-feature-data data/features/hybrid_dfrft/protocol_b_pilot_hybrid.h5 --output data/features/hybrid_dfrft/protocol_b_symbol_exact.h5 --fd0-key pilot_fd0_hat --candidate-score-mode exact_mixture --snr-source pilot_evm
```

The zero HOC placeholders are appropriate for the three models below, which
use compensated I/Q and, for the proposed method, the 48-dimensional descriptor.
They are not HOC-baseline inputs. Preserve preprocessing `.summary.json` files
and dataset metadata alongside your results. These commands were checked
against the current CLI; full data regeneration and retraining are not part
of the documentation update, and exact historical scores are not guaranteed.

## Training and test evaluation

Start with one seed:

```powershell
python scripts/18_run_hybrid_dfrft_seeds.py --methods proposed --seeds 41 --device cuda
```

Run the full comparison:

```powershell
python scripts/18_run_hybrid_dfrft_seeds.py --methods proposed mcnet starnet --seeds 41 73 107 149 211 --device cuda
```

Use `--device cpu` if needed. Training uses `configs/train/train_long.yaml`.
The runner resumes incomplete training and skips completed runs. Checkpoints
are written under `outputs/checkpoints/hybrid_dfrft/protocol_b/`; test predictions,
`seed_runs.csv`, `seed_summary.csv`, and paired-difference `summary.json` are
written under `outputs/results/hybrid_dfrft/protocol_b/`.

To evaluate existing checkpoints and rebuild summaries:

```powershell
python scripts/18_run_hybrid_dfrft_seeds.py --stages eval summary --device cuda
```

After preparing the input files, append `--dry-run` to inspect commands without
training or evaluating. Hyperparameters must be selected on validation data;
the test split is reserved for final reporting. For representation ablations,
see `scripts/20_run_hybrid_representation_ablation.py --help`.
