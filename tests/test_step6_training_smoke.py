"""Smoke tests for training modules."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.drc_triplenet import DRCTripleNet
from src.training.metrics import accuracy_from_logits
from src.training.evaluator import Evaluator


def test_accuracy_from_logits():
    logits = torch.tensor([[2.0, 1.0], [0.1, 0.9]])
    labels = torch.tensor([0, 1])
    acc = accuracy_from_logits(logits, labels)
    assert acc == 1.0


def test_evaluator_with_fake_loader():
    class FakeDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return {
                "iq": torch.randn(2, 64),
                "hoc": torch.randn(21),
                "meta": torch.randn(3),
                "evm_features": torch.randn(48),
                "label": torch.tensor(idx % 2, dtype=torch.long),
                "snr_db": torch.tensor(10.0),
                "gamma": torch.tensor(1.0),
                "domain_id": torch.tensor(0, dtype=torch.long),
            }

    model = DRCTripleNet(
        hoc_dim=21, evm_dim=48, num_classes=2, feature_dim=32,
        use_hoc_stream=False, use_metadata=False, use_constellation_image=False,
    )
    loader = DataLoader(FakeDataset(), batch_size=2)
    evaluator = Evaluator(model, device="cpu")
    df, summary = evaluator.evaluate_and_summarize(loader)
    assert len(df) == 4
    assert "accuracy" in summary
    assert "pred" in df.columns
