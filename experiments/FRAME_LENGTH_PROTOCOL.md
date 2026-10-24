# Four-length Protocol-B experiment

The retired multi-length residual-gamma diagnostic has been removed from script 32.
Its fixed-model classification scans, theoretical diagnostics, and helper functions
used by scripts 33 and 36 remain available. This campaign has independent data,
checkpoints, logs, and paper artifacts.

## Frozen design

- Lengths: 2048, 4096, 8192, 16384 complex samples at 200 kHz.
- Seven paper modulations; SNR -10 to 20 dB in 3-dB steps.
- 200/100/100 training/validation/test frames per modulation and SNR:
  15,400/7,700/7,700 independent parent frames. Data seed: 20261022.
- Each parent is a 16384-sample received waveform. Shorter records are exact
  prefixes, including identical noise, phase noise, timing, and carrier dynamics.
  Parent split membership is assigned before slicing; all length variants stay
  in the same split. HDF5 virtual datasets reference the parent by relative path.
- Nominal SNR uses expected noise power relative to the complete parent waveform;
  finite-prefix signal/noise power fluctuations are not independently renormalized.
- Protocol B: CFO uniform [-500,500] Hz, rate uniform [-8160,-180] Hz/s,
  fractional timing uniform [-0.5,0.5] samples, flat Rician K=10 dB, Wiener
  phase-noise increments with standard deviation 0.0001 rad.
- RRC roll-off 0.35, span 8, 8 samples/symbol; common QPSK comb pilots every
  16 symbols, no guard offset. Pilot counts: 16,32,64,128.
- Descriptor: all 48 exact-mixture features, receiver-derived cross-fitted pilot
  residual, at most 512 available payload symbols, eight edge symbols excluded
  at either end. Short observations never duplicate symbols to reach 512.
- Models: proposed I/Q + CD, I/Q-only ablation, IQFormer. Each is independently
  trained for each length with seeds 41,73,107,149,211: 60 training/test cells.
- Training matches `train_long.yaml`: AdamW, LR 0.0005, batch 128, maximum 100
  epochs, cosine schedule, validation early stopping patience 25. No test-based
  checkpoint selection. Descriptor normalization is fitted on training only.
- IQFormer keeps its established long-frame implementation: four randomly
  selected 128-sample windows in training, all non-overlapping windows in
  validation/test. It does not consume the constellation descriptor.

## Front end and long-frame checks

All lengths use DCFT acquisition and local coherent refinement, coherent pilot
weighting, nominal timing phase zero, and a frame-start CFO search [-550,550] Hz.
At N<=8192 the paper's rate spacing 160/5 Hz/s and FFT length 2048 remain.
At N=16384 the coarse/fine rate spacing is 40/1.25 Hz/s, fine rate radius is
50 Hz/s, FFT length is 4096, and fine CFO spacing is 0.5 Hz (radius 30 Hz).
The rule holds or improves accumulated-phase grid resolution without selecting
settings on test data. Costs therefore include the necessary longer-frame search.

The centered frequency can cross the comb-pilot Nyquist interval at N=16384.
The optional `--unwrap-centered-cfo` lifts periodic FFT bins into the known
frame-start CFO interval for each rate candidate. The interval width must be
smaller than the comb-pilot sample rate. The option is enabled for every length
in this campaign; historical commands retain their original default behavior.
CPU/CUDA parity and an explicit wrapped-frequency recovery case are tested.

Before feature extraction, the existing circular-orbit implementation checks
quadratic-phase approximation error at all four durations, over maximum
elevations 20,30,35,50,65,80,90 degrees. These geometry checks describe that
circular-orbit model; the main signal experiment remains the declared residual
quadratic-phase stress model.

## Run and resume

Exact PDF page-size checks require the paper extra (`python -m pip install -e ".[paper]"`).

```powershell
.\.venv-cuda\Scripts\python.exe scripts/50_run_frame_length_experiment.py
```

For an isolated integration check (308 parent frames, one training epoch,
one seed, all four lengths and three models):

```powershell
.\.venv-cuda\Scripts\python.exe scripts/50_run_frame_length_experiment.py --smoke --stages data features train diagnostics
```

Smoke artifacts have a separate `_smoke` campaign name and cannot be published
as paper results. Formal stages can be selected with `--stages`; completed
commands are skipped only when their command hash matches. Interrupted training
uses `last.pt`. A campaign lock prevents two controllers writing the same results.
After a crash, inspect the PID recorded in `campaign.lock` before deleting a
stale lock; never start a second live controller. Input generation is streamed
and can resume from the last completed 100-frame batch.

The complete campaign is computationally long on the local RTX 4060. No reduced
training run is substituted for the five-seed formal experiment.

## Outputs and interpretation

- Data: `data/experiments/frame_length_protocol_b/`.
- Weights: `outputs/checkpoints/frame_length_protocol_b/`.
- Results: `outputs/results/frame_length_protocol_b/`.
- Current step and child log: `status.json`; completed cells: `completeness.json`.
- Formal artifacts: `outputs/results/frame_length_protocol_b/paper/`.

After the 60 validated test results, frozen diagnostics, and timing runs exist:

```powershell
.\.venv\Scripts\python.exe scripts/51_plot_frame_length_experiment.py
```

The final controller invokes the same publication function automatically. It
writes a CSV/LaTeX table, a PDF/PNG figure, figure LaTeX, and an English manuscript
paragraph. It checks test indices, labels, SNR, correctness flags, training seed,
input provenance, and validation checkpoint selection before publishing.

The figure shows the proposed receiver's four five-seed mean accuracy/SNR
curves. It reuses `paper_baseline_protocol_a_accuracy_vs_snr` typography, 3.5x3.15
inch working canvas, 10:9 axes aspect, inset geometry, markers, grid, legend,
and 600 dpi. Exported PDF page dimensions and PNG pixel dimensions are verified
against the reference artifact (246.723712x219.246589 pt; 2058x1827 pixels).

The table reports frame duration, pilot count, rate/CFO/phase-bound P95, each
model's accuracy mean and sample SD, and measured complete receiver CPU median
latency. The CPU timing uses one thread, seed-41 weights, 77 class/SNR-stratified
validation frames, three repeats and ten warmups. It includes estimation,
compensation, descriptor construction, normalization and inference; it excludes
disk I/O and observation waiting time. Timing variability is separate from
training-seed accuracy uncertainty.

No-compensation and oracle-carrier diagnostics reuse each proposed checkpoint
and its original normalization, rebuilding the descriptor from the changed
frame. Oracle processing receives only true carrier parameters, never payload
labels, true noise variance, or true channel gains. These are frozen-input
diagnostics, not separately retrained alternative receivers or accuracy bounds.
They do not reuse the 8192-point empirical calibration thresholds or claim
new calibrated guarantees. Such thresholds require a separate calibration set.
