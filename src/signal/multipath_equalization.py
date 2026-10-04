"""Finite-frame, time-varying channel inversion for an explicitly oracle control."""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr


def compensated_path_coefficients(taps, delays, doppler_hz, n, fs, mu_hat=0., fd0_hat=0.):
    """D_hat^H H D_hat; retain carrier estimation error in the latent signal."""
    t = np.arange(n, dtype=float) / fs
    coefficients = []
    for delay, nu in zip(delays, doppler_hz):
        td = t - delay / fs
        phase_difference = 2 * np.pi * fd0_hat * (td - t) + np.pi * mu_hat * (td**2 - t**2)
        coefficients.append(taps[delay] * np.exp(2j * np.pi * nu * t + 1j * phase_difference))
    return np.asarray(coefficients)


def channel_operator(coefficients, delays):
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    delays = np.asarray(delays, dtype=int)
    if coefficients.ndim != 2 or len(coefficients) != len(delays) or np.any(delays < 0):
        raise ValueError("Expected one coefficient series per non-negative path delay.")
    n = coefficients.shape[1]

    def forward(x):
        y = np.zeros(n, dtype=np.complex128)
        for c, d in zip(coefficients, delays):
            if d < n:
                y[d:] += c[d:] * x[:n-d]
        return y

    def adjoint(y):
        x = np.zeros(n, dtype=np.complex128)
        for c, d in zip(coefficients, delays):
            if d < n:
                x[:n-d] += c[d:].conj() * y[d:]
        return x

    return LinearOperator((n, n), matvec=forward, rmatvec=adjoint, dtype=np.complex128)


def oracle_equalize(y, coefficients, delays, regularization=1e-3, max_iterations=1000):
    if regularization < 0 or max_iterations <= 0:
        raise ValueError("Invalid regularization or iteration budget.")
    result = lsqr(channel_operator(coefficients, delays), y,
                  damp=np.sqrt(regularization), atol=1e-7, btol=1e-7,
                  iter_lim=max_iterations)
    return result[0], {"stop_code": int(result[1]), "iterations": int(result[2]),
                       "residual_norm": float(result[3])}
