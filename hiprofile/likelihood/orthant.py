"""Positivity of the amplitudes: orthant probabilities of the conditional Gaussian.

The amplitudes a_ring, a_disk (and RFI amplitudes) have half-normal priors,
i.e. zero-mean Gaussians truncated to a >= 0. Marginalising them gives

    p(y | theta) = p_gauss(y | theta) * P(a >= 0 | theta, y) / P_prior(a >= 0)

with P_prior = 2^-n_amp. The sampling target uses, per component, the exact
orthant probability of that component's amplitudes (bivariate normal CDF via
Owen's T function for a profile, the normal CDF for an RFI component); for
several components the cross-component correlations are left out and
corrected exactly afterwards by Monte Carlo reweighting
(:mod:`hiprofile.inference.amplitudes`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import special

LOG2 = float(np.log(2.0))
_TINY = 1e-300


def log_bvn_orthant(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """ln P(X1 > 0, X2 > 0) for bivariate normals, mean (..., 2), cov (..., 2, 2).

    Phi2(a, b; rho) = (Phi(a) + Phi(b)) / 2 - T(a, (b - rho a) / (a s)) - T(b, (a - rho b) / (b s)) - beta
    with a = mu1 / sigma1, b = mu2 / sigma2, s = sqrt(1 - rho^2) and beta = 1/2
    if a b < 0 (or a b = 0 and a + b < 0), else 0 (Owen 1956).
    """
    s1 = np.sqrt(np.maximum(cov[..., 0, 0], _TINY))
    s2 = np.sqrt(np.maximum(cov[..., 1, 1], _TINY))
    rho = np.clip(cov[..., 0, 1] / (s1 * s2), -1 + 1e-12, 1 - 1e-12)
    a = mean[..., 0] / s1
    b = mean[..., 1] / s2
    eps = 1e-12
    a_ = np.where(np.abs(a) < eps, np.where(a < 0, -eps, eps), a)
    b_ = np.where(np.abs(b) < eps, np.where(b < 0, -eps, eps), b)
    sq = np.sqrt(1.0 - rho**2)
    p = (
        0.5 * (special.ndtr(a) + special.ndtr(b))
        - special.owens_t(a_, (b - rho * a_) / (a_ * sq))
        - special.owens_t(b_, (a - rho * b_) / (b_ * sq))
    )
    beta = np.where((a_ * b_ < 0), 0.5, 0.0)
    p = p - beta
    # for very small probabilities the formula cancels; bound by the exact
    # independent-case product scaled by the most favourable correlation
    upper = np.minimum(special.ndtr(a), special.ndtr(b))
    p = np.clip(p, 0.0, upper)
    return np.log(np.maximum(p, _TINY))


def log_orthant(components: Sequence[str], mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Sum over components of ln P(component amplitudes >= 0 | theta, y).

    mean (B, q), cov (B, q, q) of the amplitude columns in component order.
    """
    out = np.zeros(mean.shape[0])
    col = 0
    for c in components:
        if c == "profile":
            sl = slice(col, col + 2)
            out += log_bvn_orthant(mean[:, sl], cov[:, sl, sl])
            col += 2
        else:
            sd = np.sqrt(np.maximum(cov[:, col, col], _TINY))
            out += special.log_ndtr(mean[:, col] / sd)
            col += 1
    return out


def orthant_is_exact(components: Sequence[str]) -> bool:
    """True if the per-component orthant term is the exact joint probability."""
    return len(components) <= 1
