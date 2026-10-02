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

The common training and evaluation entry points are `scripts/13_train_model.py`
and `scripts/14_evaluate_model.py`. Their default model is `drc_triplenet`, using
`configs/model/triplenet_iq_evm.yaml`. The historical class name is retained for
checkpoint compatibility; this configuration enables only the I/Q and descriptor streams.
The other two `triplenet_*.yaml` files implement the paper's single-stream ablations.
The six `paper_*.yaml` files configure the published comparison baselines.

`src/` contains shared implementations, `scripts/` contains experiment entry points,
`configs/` contains active settings, and `tests/` contains regression checks.
Generated datasets, features and splits live in `data/`; checkpoints, results and
paper figures live in `outputs/`. HOC extraction remains available for diagnostics
and the NASA HOC-NN baseline.

Author: Yehui. To cite the software:
Yehui. *LEO modulation classification*, version 0.1.0 (computer software).
Third-party implementation provenance is documented in
[STARNet reproduction](docs/starnet_reproduction.md).
