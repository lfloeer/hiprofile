"""Linear parameters: draws from beta | theta, y, K and the positivity of the amplitudes.

The physical model uses half-normal priors on the component amplitudes (zero-mean
Gaussians truncated to a >= 0). Marginalising them gives

    p(y | theta) = sum_K P(K) p_gauss(y | theta, K) * P(a >= 0 | theta, y, K) / 2^-q.

The sampling target (:meth:`MarginalLikelihood.loglike`) contains the orthant
probability per K, P_approx(theta, K), which is exact for up to two amplitudes.
For three or more the remaining factor P_true / P_approx is estimated here by
Monte Carlo and applied as importance weight; the evidence changes by the log
of its posterior mean. For each sample the baseline order is drawn from
P(K | theta, y) and the amplitudes from the K-conditional Gaussian until a
draw with non-negative amplitudes is found.

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
    beta: np.ndarray  # (n, p) one draw per theta sample, zero-padded beyond K (NaN if none accepted)
    accept: np.ndarray  # (n,) Monte Carlo estimate of P(amp >= 0 | theta, y, K)
    beta_mean: np.ndarray  # (n, p) conditional mean
    factor: np.ndarray  # (n,) importance factor from the sampling target to the truncated model
    log_offset: float  # constant ln-evidence offset (prior normalisation when not in the target)
    k: np.ndarray  # (n,) baseline order drawn for each sample

    def log_z_correction(self, weights: np.ndarray) -> float:
        """ln Z_trunc - ln Z_target for normalised posterior weights of the target."""
        mean_factor = float(np.sum(weights * self.factor))
        return float(np.log(max(mean_factor, 1e-300)) + self.log_offset)

    def negative_fraction(self, weights: np.ndarray) -> float:
        """Posterior fraction of Gaussian amplitude draws that are negative."""
        return float(1.0 - np.sum(weights * self.accept))


def _sample_k(pk: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One index per row drawn from the row-wise probabilities ``pk`` (b, nK)."""
    cdf = np.cumsum(pk, axis=1)
    cdf /= cdf[:, -1:]
    u = rng.uniform(size=(pk.shape[0], 1))
    return np.minimum(np.sum(cdf < u, axis=1), pk.shape[1] - 1)


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
    k_draw = np.full(n, like.baseline_order)
    k_values = np.asarray(like.k_values)
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        th = theta[sl]
        if like.n_k > 1:
            ks = k_values[_sample_k(like.k_posterior(th), rng)]
        else:
            ks = np.full(th.shape[0], like.baseline_order)
        k_draw[sl] = ks
        mean, cov = like.conditional(th, k=ks)
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
            approx[sl] = np.exp(like.log_orthant_amp(mean[:, :q], cov[:, :q, :q]))
        first = np.argmax(ok, axis=1)
        chosen = draws[np.arange(draws.shape[0]), first]
        chosen[~ok.any(axis=1)] = np.nan
        beta[sl] = chosen
    if q == 0:
        factor, offset = np.ones(n), 0.0
    elif like.positivity:
        if orthant_is_exact(q):
            factor = np.ones(n)
        else:
            factor = np.where(approx > 1e-300, accept / np.maximum(approx, 1e-300), 0.0)
        offset = 0.0
    else:
        factor, offset = accept, q * float(np.log(2.0))
    return AmplitudeDraws(
        beta=beta, accept=accept, beta_mean=means, factor=factor, log_offset=offset, k=k_draw
    )
