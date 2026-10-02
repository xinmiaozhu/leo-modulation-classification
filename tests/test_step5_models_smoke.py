"""Smoke tests for src/models."""

from pathlib import Path
import argparse
import importlib.util

import pytest
import torch

from src.models.drc_triplenet import DRCTripleNet
from src.models.losses import DRCTrainingLoss
from src.models.paper_baselines import PAPER_BASELINE_MODELS, build_paper_baseline


def test_paper_baselines_forward():
    B, N, hoc_dim, num_classes = 2, 1024, 21, 7
    iq = torch.randn(B, 2, N)
    hoc = torch.randn(B, hoc_dim)

    for model_name in PAPER_BASELINE_MODELS:
        model_hoc_dim = 10 if model_name == "paper_nasa_hoc_nn" else hoc_dim
        model = build_paper_baseline(
            model_name,
            num_classes=num_classes,
            hoc_dim=model_hoc_dim,
        )
        model_hoc = torch.randn(B, model_hoc_dim)
        logits, aux = model(iq, model_hoc, return_aux=True)
        assert logits.shape == (B, num_classes)
        assert aux["f_out"].shape[0] == B


@pytest.mark.parametrize(
    ("use_iq", "use_hoc", "use_evm"),
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_drc_triplenet_input_combinations(use_iq, use_hoc, use_evm):
    batch, num_samples, hoc_dim, evm_dim, num_classes = 2, 128, 21, 48, 7
    model = DRCTripleNet(
        hoc_dim=hoc_dim,
        evm_dim=evm_dim,
        num_classes=num_classes,
        feature_dim=16,
        raw_channels=(8, 12, 16),
        raw_kernel_sizes=(7, 5, 3),
        hoc_hidden_dims=(16,),
        evm_hidden_dims=(16,),
        fusion_hidden_dim=16,
        use_iq_stream=use_iq,
        use_hoc_stream=use_hoc,
        use_metadata=False,
        use_constellation_image=False,
        use_evm_features=use_evm,
        dropout=0.0,
    )
    iq = torch.randn(batch, 2, num_samples) if use_iq else None
    hoc = torch.randn(batch, hoc_dim) if use_hoc else None
    evm = torch.randn(batch, evm_dim) if use_evm else None

    logits, aux = model(iq, hoc, evm_features=evm, return_aux=True)

    assert logits.shape == (batch, num_classes)
    assert aux["f_out"].shape == (batch, 16)


@pytest.mark.parametrize("config_path", sorted((Path(__file__).resolve().parents[1] / "configs/model").glob("*.yaml")), ids=lambda p: p.stem)
def test_retained_model_configs_train_eval_compatible(config_path):
    root = Path(__file__).resolve().parents[1]
    modules = []
    for name in ("13_train_model", "14_evaluate_model"):
        spec = importlib.util.spec_from_file_location(name, root / "scripts" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    model_name = config_path.stem if config_path.stem.startswith("paper_") else "drc_triplenet"
    args = argparse.Namespace(model=model_name, model_config=str(config_path))
    hoc_dim = 10 if model_name == "paper_nasa_hoc_nn" else 21
    trained = modules[0].build_model(args, num_classes=7, hoc_dim=hoc_dim, evm_dim=48).eval()
    evaluated = modules[1].build_model(args, num_classes=7, hoc_dim=hoc_dim, evm_dim=48).eval()
    evaluated.load_state_dict(trained.state_dict(), strict=True)
    with torch.inference_mode():
        inputs = (torch.randn(2, 2, 1024), torch.randn(2, hoc_dim))
        kwargs = {"evm_features": torch.randn(2, 48)} if model_name == "drc_triplenet" else {}
        expected = trained(*inputs, **kwargs)
        actual = evaluated(*inputs, **kwargs)
    assert actual.shape == (2, 7)
    torch.testing.assert_close(actual, expected)


def test_paper_receiver_loss_backward():
    model = DRCTripleNet(
        hoc_dim=21, evm_dim=48, num_classes=7, feature_dim=16,
        use_hoc_stream=False, use_metadata=False, use_constellation_image=False,
    )
    logits, aux = model(torch.randn(2, 2, 128), None, evm_features=torch.randn(2, 48), return_aux=True)
    loss, logs = DRCTrainingLoss(lambda_gate=0.0)(logits, torch.tensor([0, 1]), aux=aux)
    loss.backward()
    assert torch.isfinite(loss)
    assert "loss_cls" in logs
    for parameter in model.classifier.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
