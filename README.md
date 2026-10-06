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

The common training and evaluation entry points are `scripts/13_train_model.py`
and `scripts/14_evaluate_model.py`. Their default model is `drc_dualnet`, using
`configs/model/dualnet_iq_evm.yaml`. The `DRCDualNet` classifier uses the I/Q and
descriptor streams; existing two-stream state-dict checkpoints remain compatible.
The other two `dualnet_*.yaml` files implement the paper's single-stream ablations.
The six `paper_*.yaml` files configure the published comparison baselines.

`src/` contains shared implementations, `scripts/` contains experiment entry points,
`configs/` contains active settings, and `tests/` contains regression checks.
Generated datasets, features and splits live in `data/`; checkpoints, results and
paper figures live in `outputs/`. HOC extraction remains available for diagnostics
and the NASA HOC-NN baseline.

Generate the two pilot-receiver characterization panels separately with:

```powershell
python scripts/44_plot_pilot_receiver_panels.py
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
