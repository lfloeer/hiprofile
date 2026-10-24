"""Linear parameters: draws from beta | theta, y and the positivity of the amplitudes.

The physical model uses half-normal priors on the component amplitudes (zero-mean
Gaussians truncated to a >= 0). Marginalising them gives

    p(y | theta) = p_gauss(y | theta) * P(a >= 0 | theta, y) / 2^-n_amp.

The sampling target (:meth:`MarginalLikelihood.loglike`) contains the
orthant probability per component, P_approx(theta), which is exact for one
component. For several components the remaining factor
P(a >= 0 | theta, y) / P_approx(theta) is estimated here by Monte Carlo and
applied as importance weight; the evidence changes by the log of its
posterior mean. One accepted draw per sample gives the amplitudes.

Without ``positivity`` the sampler targets p_gauss and the full orthant
probability is applied here instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hiprofile.likelihood.marginal import MarginalLikelihood
from hiprofile.likelihood.orthant import orthant_is_exact


@dataclass
class AmplitudeDraws:
    beta: np.ndarray  # (n, p) one draw per theta sample (NaN if none accepted)
    accept: np.ndarray  # (n,) Monte Carlo estimate of P(amp >= 0 | theta, y)
    beta_mean: np.ndarray  # (n, p) conditional mean
    factor: np.ndarray  # (n,) importance factor from the sampling target to the truncated model
    log_offset: float  # constant ln-evidence offset (prior normalisation when not in the target)

    def log_z_correction(self, weights: np.ndarray) -> float:
        """ln Z_trunc - ln Z_target for normalised posterior weights of the target."""
        mean_factor = float(np.sum(weights * self.factor))
        return float(np.log(max(mean_factor, 1e-300)) + self.log_offset)

    def negative_fraction(self, weights: np.ndarray) -> float:
        """Posterior fraction of Gaussian amplitude draws that are negative."""
        return float(1.0 - np.sum(weights * self.accept))


def draw_amplitudes(
    like: MarginalLikelihood,
    theta: np.ndarray,
    rng: np.random.Generator,
    n_draw: int = 64,
    chunk: int = 512,
) -> AmplitudeDraws:
    n = theta.shape[0]
    p = like.n_linear
    q = like.n_amp
    beta = np.full((n, p), np.nan)
    accept = np.ones(n)
    approx = np.ones(n)
    means = np.zeros((n, p))
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        mean, cov = like.conditional(theta[sl])
        means[sl] = mean
        if p == 0:
            continue
        chol = np.linalg.cholesky(cov + 1e-300 * np.eye(p))
        z = rng.standard_normal((mean.shape[0], n_draw, p))
        draws = mean[:, None, :] + np.einsum("bpq,bdq->bdp", chol, z)
        if q == 0:
            beta[sl] = draws[:, 0]
            continue
        ok = np.all(draws[:, :, :q] >= 0, axis=2)
        accept[sl] = ok.mean(axis=1)
        if like.positivity:
            approx[sl] = np.exp(like.log_orthant(mean[:, :q], cov[:, :q, :q]))
        first = np.argmax(ok, axis=1)
        chosen = draws[np.arange(draws.shape[0]), first]
        chosen[~ok.any(axis=1)] = np.nan
        beta[sl] = chosen
    if q == 0:
        factor, offset = np.ones(n), 0.0
    elif like.positivity:
        if orthant_is_exact(like.components):
            factor = np.ones(n)
        else:
            factor = np.where(approx > 1e-300, accept / np.maximum(approx, 1e-300), 0.0)
        offset = 0.0
    else:
        factor, offset = accept, q * float(np.log(2.0))
    return AmplitudeDraws(beta=beta, accept=accept, beta_mean=means, factor=factor, log_offset=offset)
