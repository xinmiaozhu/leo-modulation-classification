# LEO modulation classification

This project implements a compensation-first AMC receiver: pilot-aided carrier estimation, compensated I/Q,
a 48-dimensional exact-mixture constellation descriptor, and a two-stream classifier.

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

Run commands from the repository root in an activated Python environment.
Dependencies are declared in `setup.py`; no separate requirements file is needed.
The `dev` extra installs pytest. Training guides use CUDA; use `--device cpu`
when a CUDA-capable PyTorch installation is unavailable.

Dataset-generation commands are in [PRACTICAL_PROTOCOL.md](experiments/PRACTICAL_PROTOCOL.md).
The [script index](scripts/README.md) lists all entry points in task order and
explains the optional experiment branches.

The common training and evaluation entry points are `scripts/12_train_model.py`
and `scripts/13_evaluate_model.py`. Their default model is `drc_dualnet`, using
`configs/model/dualnet_iq_evm.yaml`. The `DRCDualNet` classifier uses the I/Q and
descriptor streams; existing two-stream state-dict checkpoints remain compatible.
The other two `dualnet_*.yaml` files implement the paper's single-stream ablations.
The existing representation runner (18) also evaluates the optional HOC branch
with `--methods fusion_hoc` and the I/Q + HOC control with `--methods iq_hoc`
under both final protocols. The latter disables the descriptor branch and
matches its encoder widths for the auxiliary-representation comparison.
Both controls use real 21-D HOC from frozen compensated I/Q and seeds
41, 73, 107, 149, 211. Only `fusion_hoc` also consumes the complete 48-D
descriptor. See the training guides for commands.
The six `paper_*.yaml` files configure the published comparison baselines.

`src/` contains shared implementations, `scripts/` contains experiment entry points,
`configs/` contains active settings, and `tests/` contains regression checks.
Generated datasets, features and splits live in `data/`; checkpoints, results and
paper figures live in `outputs/`. HOC extraction remains available for diagnostics
and for the optional matched-seed HOC comparisons.

The independent [four-length Protocol-B campaign](experiments/FRAME_LENGTH_PROTOCOL.md)
uses `scripts/50_run_frame_length_experiment.py` for N=2048/4096/8192/16384,
three models and five matched seeds. Script 52 publishes its manuscript table
and reference-styled figure only after all 60 results pass validation. Its
outputs are isolated under `outputs/results/frame_length_protocol_b/`.

Generate the two pilot-receiver characterization panels separately with:

```powershell
python scripts/43_plot_pilot_receiver_panels.py
```

This uses the existing coherent-pilot estimates and pilot-count summary. It writes
`pilot_receiver_characterization_coherent_a.pdf` and
`pilot_receiver_characterization_coherent_b.pdf` in `outputs/figures/paper/`,
plus `pilot_receiver_snr_error.csv` in `outputs/results/practical_receiver_checks_coherent/`.
It does not produce a combined figure. Use `--help` to override input and output paths.

Author: Yehui. To cite the software:
Yehui. *LEO modulation classification*, version 0.1.0 (computer software).
Third-party implementation provenance is documented in
[STARNet reproduction](docs/starnet_reproduction.md).

## Unified five-seed baseline comparison

Complete CNN2, CNN-LSTM and IQCNet on the final Protocol A/B inputs with:

```powershell
.\.venv-cuda\Scripts\python.exe scripts/16_run_hybrid_dfrft_seeds.py --protocols A B --methods cnn2 cnn_lstm_dual satellite_cnn --seeds 41 73 107 149 211 --device cuda --five-seed-report
```

The three Protocol-B seed-41 runs have verified training commands, checkpoint
configurations and aligned test predictions and are reused. Protocol-A seed-41
checkpoints are unavailable, so those three runs are repeated in the canonical
final-protocol directories. This campaign therefore runs 27 training/test pairs.
The optimizer, early stopping, input normalization and model architectures are
unchanged; CNN-LSTM retains its power normalization. Completed runs are skipped
on resume. All checkpoints are selected on validation accuracy.

`outputs/results/baseline_comparison/completeness.json` lists the missing cells
in the full 7-model x 2-protocol x 5-seed comparison. Final `seed_summary.csv`,
`paired_gains.csv` and `protocol_a_accuracy_complexity.csv` are published only
when all 70 test results pass sample-index, label, SNR and accuracy checks.
Accuracy columns in this directory use percent and sample SD (`ddof=1`).
The complexity table retains previous cost measurements, replaces accuracy with
five-seed mean/SD, and labels the component-sum E2E latency estimate explicitly.
It does not imply a new joint timing benchmark.

Formal Fig. 5, Fig. 6 and Table IV now default to five-seed accuracy:

```powershell
python scripts/46_plot_paper_baseline_comparison.py
python scripts/47_plot_protocol_b_accuracy_vs_snr.py
python scripts/40_benchmark_paper_models.py --table-iv
```

The first two commands generate five-seed mean curves without uncertainty bands in
`outputs/figures/paper/paper_baseline_protocol_{a,b}_accuracy_vs_snr.*`.
The explicit `*_five_seed_accuracy_vs_snr.*` names remain compatible aliases.
Table IV is `outputs/results/baseline_comparison/table_iv.csv`, with accuracy
mean, sample SD, all five seed IDs, and a formatted mean-plus-SD column.
`table_iv_metadata.json` separates training-seed uncertainty from hardware timing:
parameters/MACs describe fixed architectures, and existing CPU benchmark medians
are not five-checkpoint latency means or SDs. No single-seed accuracy override
is accepted by the formal table entry point.
All formal publication commands require the complete 70-result comparison.
Per-seed prediction records remain available in the comparison results, while
publication figures use the aggregate across all five matched seeds.
Protocol B continues to use DCFT acquisition plus coherent refinement.

## IQFormer external baseline

IQFormer (IEEE TCCN 2025, 11(3):1623-1634) uses the authors' RadioML architecture:
three stages [2,3,2], widths [64,64,64], a two-layer bidirectional LSTM, and
additive attention. The implementation is integrated into `src/models/paper_baselines.py`.
Upstream: https://github.com/WestdoorSad/IQFormer, commit
`7ee6ac949551b24d45f218762cab919e0cb6b4f9` (MIT notice preserved in the source).
The port matched the unchanged upstream network exactly after loading identical
weights; its STFT was checked against SciPy. The batch-one squeeze issue is fixed.

The two inputs are compensated I/Q and the real part of the first 32 frequency
bins from `scipy.signal.stft(I, 1., 'blackman', 31, 30, 128)`, matching the
upstream complex-to-float conversion. This is not a magnitude spectrogram.
An equivalent real DFT convolution computes the STFT on the model device.
It is included in measured inference and end-to-end latency.

For 8192-sample frames, training averages logits over four randomly selected
128-sample windows; validation/test average all 64 non-overlapping windows.
This is a long-frame adaptation of the published model. We retain the common
paper training setup (AdamW, 0.0005 initial LR, batch 128, 100-epoch maximum,
cosine schedule, validation early stopping patience 25), rather than claiming
an exact reproduction of the paper's original datasets/training recipe.
No test results are used for configuration selection.

```powershell
python scripts/16_run_hybrid_dfrft_seeds.py --protocols A B --methods iqformer --seeds 41 73 107 149 211 --device cuda --report
```

The runner resumes incomplete runs and skips completed runs. Final predictions
are under `outputs/results/{exact_mixture,hybrid_dfrft}/protocol_{a,b}/iqformer/`.
`--report` is now an alias for `--five-seed-report` and publishes formal figures
and Table IV only after all seven models have all five seeds under both protocols.
Combined formal tables are written to `outputs/results/baseline_comparison/`;
the older `iqformer_comparison/` summaries remain available for traceability.
Hardware profiling is a separate explicit invocation of script 41 with input paths.
HOC representation ablations remain separate and unchanged.
