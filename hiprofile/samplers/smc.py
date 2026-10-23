"""Adaptive tempered sequential Monte Carlo with random-walk Metropolis moves.

All particles are evaluated in one batched likelihood call per step, so with
the JAX likelihood backend the expensive part runs as a single XLA call (CPU or
GPU). The evidence is the product of the mean incremental weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit, logsumexp

from hiprofile.samplers.base import (
    CountingLikelihood,
    LogLike,
    SamplerResult,
    Timer,
    Transform,
    systematic_resample,
)


def _log_prior_z(z: np.ndarray) -> np.ndarray:
    return -np.sum(np.logaddexp(0.0, z) + np.logaddexp(0.0, -z), axis=1)


def _next_beta(log_l: np.ndarray, beta: float, target_ess: float) -> float:
    lo, hi = beta, 1.0

    def ess(b: float) -> float:
        lw = (b - beta) * log_l
        lw -= np.max(lw)
        w = np.exp(lw)
        return w.sum() ** 2 / np.sum(w**2)

    if ess(1.0) >= target_ess:
        return 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ess(mid) < target_ess:
            hi = mid
        else:
            lo = mid
    return max(lo, beta + 1e-8)


def run_smc(
    loglike: LogLike,
    prior_transform: Transform,
    ndim: int,
    *,
    seed: int,
    n_live: int = 400,
    n_particles: int | None = None,
    ess_fraction: float = 0.5,
    n_moves: int = 10,
    max_steps: int = 500,
    **options: Any,
) -> SamplerResult:
    rng = np.random.default_rng(seed)
    like = CountingLikelihood(loglike)
    n = n_particles or max(1000, 2 * n_live)

    def eval_ll(z: np.ndarray) -> np.ndarray:
        u = np.clip(expit(z), 1e-15, 1 - 1e-15)
        return like(prior_transform(u))

    with Timer() as t:
        u = rng.uniform(size=(n, ndim))
        z = np.log(u) - np.log1p(-u)
        log_l = eval_ll(z)
        log_pz = _log_prior_z(z)
        beta = 0.0
        log_z = 0.0
        var_terms = 0.0
        betas = [0.0]
        acc_hist = []
        for _ in range(max_steps):
            new = _next_beta(log_l, beta, ess_fraction * n)
            lw = (new - beta) * log_l
            log_z += float(logsumexp(lw) - np.log(n))
            w = np.exp(lw - np.max(lw))
            var_terms += float(np.var(w) / (np.mean(w) ** 2 * n))
            idx = systematic_resample(w / w.sum(), n, rng)
            z, log_l, log_pz = z[idx], log_l[idx], log_pz[idx]
            beta = new
            betas.append(beta)
            cov = np.cov(z.T).reshape(ndim, ndim) + 1e-9 * np.eye(ndim)
            chol = np.linalg.cholesky(cov)
            scale = 2.38 / np.sqrt(ndim)
            acc_total = 0.0
            for _ in range(n_moves):
                prop = z + scale * rng.standard_normal((n, ndim)) @ chol.T
                ll_p = eval_ll(prop)
                lp_p = _log_prior_z(prop)
                log_a = beta * (ll_p - log_l) + lp_p - log_pz
                acc = np.log(rng.uniform(size=n)) < log_a
                z[acc], log_l[acc], log_pz[acc] = prop[acc], ll_p[acc], lp_p[acc]
                rate = acc.mean()
                acc_total += rate
                scale *= np.exp(rate - 0.3)
            acc_hist.append(acc_total / n_moves)
            if beta >= 1.0:
                break
        u = np.clip(expit(z), 1e-15, 1 - 1e-15)
        theta = prior_transform(u)
    return SamplerResult(
        samples=theta,
        log_weights=np.zeros(n),
        log_l=log_l,
        log_z=log_z,
        log_z_err=float(np.sqrt(var_terms)),
        backend="smc",
        runtime=t.elapsed,
        n_like=like.n_calls,
        stats={"n_steps": len(betas) - 1, "n_particles": n, "mean_acceptance": float(np.mean(acc_hist))},
        trace={"beta": np.array(betas), "acceptance": np.array(acc_hist)},
    )
