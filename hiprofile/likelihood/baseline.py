"""Chebyshev baselines and baseline priors from signal-free neighbour spectra."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hiprofile.model.kernel import chebyshev_basis


def fit_chebyshev(flux: np.ndarray, good: np.ndarray, order: int) -> np.ndarray:
    """Least-squares Chebyshev coefficients c_0..c_order over the band."""
    t = chebyshev_basis(flux.size, order)
    coef, *_ = np.linalg.lstsq(t[:, good].T, flux[good], rcond=None)
    return coef


def baseline_prior_from_spectra(
    spectra: Sequence[np.ndarray],
    order: int,
    goods: Sequence[np.ndarray] | None = None,
    min_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical mean and covariance of Chebyshev coefficients of signal-free spectra.

    The spectra must share the channel count of the target spectrum. The
    covariance gets ``min_scale**2`` added on the diagonal so that it never
    becomes tighter than the per-spectrum fit uncertainty.
    """
    coefs = []
    for i, y in enumerate(spectra):
        y = np.asarray(y, float)
        good = np.isfinite(y) if goods is None else (np.asarray(goods[i], bool) & np.isfinite(y))
        coefs.append(fit_chebyshev(np.where(good, y, 0.0), good, order))
    c = np.array(coefs)
    if c.shape[0] < order + 3:
        raise ValueError("need more neighbour spectra than baseline coefficients + 2")
    mean = c.mean(axis=0)
    cov = np.cov(c.T).reshape(order + 1, order + 1) + np.eye(order + 1) * min_scale**2
    return mean, cov
