"""Full project smoke test with tiny data."""

from pathlib import Path
import h5py
import numpy as np

from src.datasets.build_dataset import DatasetBuildConfig, build_dataset_arrays
from src.drc_hoc.extractor import DRCHOCConfig, DRCHOCExtractor
from src.models.drc_dualnet import DRCDualNet


def test_tiny_end_to_end_components():
    cfg = DatasetBuildConfig(
        seed=0,
        modulations=("BPSK", "QPSK"),
        num_samples_per_class=1,
        num_symbols=32,
        samples_per_symbol=4,
        target_num_samples=128,
        snr_db_min=10,
        snr_db_max=20,
        mu_min_hz_per_s=-1000,
        mu_max_hz_per_s=1000,
    )
    arrays = build_dataset_arrays(cfg, show_progress=False)
    drc = DRCHOCExtractor(DRCHOCConfig(sample_rate_hz=1e6, estimator_type="paper_strict_dfrft", alpha_steps=41, mu_grid_min=-1000, mu_grid_max=1000, candidate_orders=(2,4), num_subwindows=4))
    out = drc.extract(arrays["iq"][0], mu_true_hz_per_s=float(arrays["mu"][0]))
    hoc_dim = out["h_drc"].shape[0]
    model = DRCDualNet(
        hoc_dim=hoc_dim, evm_dim=48, num_classes=2, feature_dim=32,
        use_hoc_stream=False, use_metadata=False, use_constellation_image=False,
    )
    import torch
    iq = torch.tensor(arrays["iq"], dtype=torch.float32)
    hoc = torch.stack([torch.tensor(out["h_drc"], dtype=torch.float32), torch.tensor(out["h_drc"], dtype=torch.float32)])
    meta = torch.randn(2, 3)
    logits = model(iq, hoc, meta, evm_features=torch.randn(2, 48))
    assert logits.shape == (2, 2)
