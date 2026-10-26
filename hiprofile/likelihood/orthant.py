"""Positivity of the amplitudes: orthant probabilities of the conditional Gaussian.

The component amplitudes (the integrated flux S of each profile, the flux of
each RFI Gaussian) have half-normal priors, i.e. zero-mean Gaussians
truncated to a >= 0. Marginalising them gives

    p(y | theta, K) = p_gauss(y | theta, K) * P(a >= 0 | theta, y, K) / P_prior(a >= 0)

with P_prior = 2^-q for q amplitudes. The orthant probability is exact for one
amplitude (normal CDF) and two (bivariate normal CDF via Owen's T function).
For three or more the product of the marginal probabilities is used in the
sampling target and corrected afterwards by Monte Carlo reweighting
(:mod:`hiprofile.inference.amplitudes`).
"""

from __future__ import annotations

import numpy as np
from scipy import special

LOG2 = float(np.log(2.0))
_TINY = 1e-300
_TAIL = 1e-200  # below this the closed form is replaced by the log-space tail approximation


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
    # Far in the tail (a component many sigma negative, as when a second profile
    # takes up a structure as negative flux) the formula underflows; flooring it
    # would cap the positivity penalty, so use the log-space approximation there.
    tail = p < _TAIL
    if np.any(tail):
        p = np.where(tail, 1.0, p)
        return np.where(tail, _log_bvn_orthant_tail(a, b, rho), np.log(p))
    return np.log(p)


def _log_bvn_orthant_tail(a: np.ndarray, b: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """ln P(Z1 > -a, Z2 > -b) for small probabilities (Mendell & Elston 1974).

    P(Z2 > h2) times a normal approximation of P(Z1 > h1 | Z2 > h2), with the
    less probable variable conditioned on first; all in log space.
    """
    swap = a > b  # condition on the more negative standardised mean first
    h1, h2 = -np.where(swap, a, b), -np.where(swap, b, a)
    log_q2 = special.log_ndtr(-h2)
    lam = np.exp(-0.5 * h2**2 - 0.5 * np.log(2 * np.pi) - log_q2)  # E[Z2 | Z2 > h2]
    var2 = np.maximum(1.0 + h2 * lam - lam**2, 0.0)  # Var[Z2 | Z2 > h2]
    sd1 = np.sqrt(np.maximum(1.0 - rho**2 * (1.0 - var2), 1e-12))
    return log_q2 + special.log_ndtr(-(h1 - rho * lam) / sd1)


def log_orthant(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """ln P(a >= 0 | theta, y) for amplitude means (B, q) and covariances (B, q, q).

    Exact for q <= 2; the product of the marginals for q >= 3.
    """
    q = mean.shape[-1]
    if q == 0:
        return np.zeros(mean.shape[0])
    if q == 2:
        return log_bvn_orthant(mean, cov)
    sd = np.sqrt(np.maximum(np.diagonal(cov, axis1=-2, axis2=-1), _TINY))
    return np.sum(special.log_ndtr(mean / sd), axis=-1)


def orthant_is_exact(n_amp: int) -> bool:
    """True if :func:`log_orthant` is the exact joint probability for ``n_amp`` amplitudes."""
    return n_amp <= 2
