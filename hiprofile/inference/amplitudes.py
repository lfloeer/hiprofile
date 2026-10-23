"""Linear parameters: draws from beta | theta, y and the positivity truncation.

The sampler works with the Gaussian-marginalised likelihood, i.e. a zero-mean
Gaussian prior on the component amplitudes. The physical model uses that prior
truncated to amplitudes >= 0 (a half-normal for each amplitude). The truncated
model is obtained by importance reweighting:

    Z_trunc = Z_gauss * E_post[ P(amp >= 0 | theta, y) ] / P_prior(amp >= 0)

with P_prior = 2^-n_amp. P(amp >= 0 | theta, y) is estimated per posterior
sample by Monte Carlo; one accepted draw per sample gives the amplitudes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hiprofile.likelihood.marginal import MarginalLikelihood


@dataclass
class AmplitudeDraws:
    beta: np.ndarray  # (n, p) one draw per theta sample (NaN if none accepted)
    accept: np.ndarray  # (n,) estimated P(amp >= 0 | theta, y)
    beta_mean: np.ndarray  # (n, p) conditional mean

    def log_z_correction(self, weights: np.ndarray, n_amp: int) -> float:
        """ln Z_trunc - ln Z_gauss for normalised posterior weights."""
        mean_acc = float(np.sum(weights * self.accept))
        return float(np.log(max(mean_acc, 1e-300)) + n_amp * np.log(2.0))

    def negative_fraction(self, weights: np.ndarray) -> float:
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
        first = np.argmax(ok, axis=1)
        chosen = draws[np.arange(draws.shape[0]), first]
        chosen[~ok.any(axis=1)] = np.nan
        beta[sl] = chosen
    return AmplitudeDraws(beta=beta, accept=accept, beta_mean=means)
