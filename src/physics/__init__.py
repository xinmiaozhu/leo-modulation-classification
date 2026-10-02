"""Physics modules for LEO geometry, Doppler modeling and gamma metrics."""

from .doppler_model import DopplerModel, DopplerState
from .gamma_metric import (
    compute_gamma,
    compute_gamma_from_window,
    hoc_relative_error,
    find_gamma_boundary,
    categorize_gamma_region,
)

__all__ = [
    "DopplerModel",
    "DopplerState",
    "compute_gamma",
    "compute_gamma_from_window",
    "hoc_relative_error",
    "find_gamma_boundary",
    "categorize_gamma_region",
]
