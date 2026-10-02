import numpy as np

from src.drc_hoc.cumulants import cumulant_pq
from src.drc_hoc.paper_features import (
    NASA_ORDERS,
    mixed_cumulants,
    nasa_feature_vector,
)


def test_fast_nasa_cumulants_match_partition_definition():
    rng = np.random.default_rng(7)
    z = rng.standard_normal(96) + 1j * rng.standard_normal(96)
    z = z - np.mean(z)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2))

    actual = mixed_cumulants(z)
    for order in NASA_ORDERS:
        expected = cumulant_pq(z, *order, center=False)
        np.testing.assert_allclose(actual[order], expected, rtol=1e-10, atol=1e-10)


def test_nasa_feature_vector_uses_pilot_evm_not_true_snr():
    symbols = np.tile(np.asarray([1.0, 1.0j, -1.0, -1.0j]), 64)
    features = nasa_feature_vector(symbols, pilot_evm_mse=0.01)
    assert features.shape == (10,)
    np.testing.assert_allclose(features[-1], 20.0, atol=1e-5)
