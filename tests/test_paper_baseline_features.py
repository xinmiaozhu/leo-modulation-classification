"""IQFormer preprocessing and native-frame aggregation regression checks."""
import numpy as np
import pytest
import torch
from scipy.signal import stft
from src.models.paper_baselines import IQFormerBaseline


@pytest.mark.parametrize("batch,length", [(1, 128), (3, 128), (2, 1024)])
def test_iqformer_stft_matches_official_scipy_real_input(batch, length):
    rng = np.random.default_rng(41)
    iq = rng.normal(size=(batch, 2, length)).astype(np.float32)
    model = IQFormerBaseline(num_classes=7).eval()
    expected = np.stack([stft(x[0], 1., 'blackman', 31, 30, 128)[2][:32].real[None] for x in iq])
    actual = model.stft_features(torch.from_numpy(iq)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=4e-7)


def test_iqformer_batch_one_aggregation_and_both_branches_receive_gradients():
    torch.manual_seed(41)
    model = IQFormerBaseline(7, train_windows=0).eval()
    iq = torch.randn(1, 2, 256)
    with torch.no_grad():
        expected = torch.stack([model.core(x, model.stft_features(x)) for x in iq.split(128, dim=-1)]).mean(0)
        actual, aux = model(iq, return_aux=True)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    assert aux['num_windows'].item() == 2
    model.train()
    model(torch.randn(2, 2, 256)).square().mean().backward()
    for branch in [model.core.patch_embedIQ, model.core.patch_embedSTFT]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
