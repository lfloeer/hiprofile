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


def _draw_mixture(dists: list[Any], mix: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    counts = rng.multinomial(n, mix)
    parts = [np.atleast_2d(d.rvs(size=c)) for d, c in zip(dists, counts, strict=True) if c > 0]
    return np.vstack(parts) if parts else np.zeros((0, dists[0].dim))


def _log_mixture(dists: list[Any], mix: np.ndarray, z: np.ndarray) -> np.ndarray:
    return logsumexp([np.log(m) + d.logpdf(z) for d, m in zip(dists, mix, strict=True)], axis=0)


def _next_beta(log_l: np.ndarray, beta: float, target_ess: float, base: np.ndarray | float = 0.0) -> float:
    lo, hi = beta, 1.0

    def ess(b: float) -> float:
        lw = base + (b - beta) * log_l
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
    n_runs: int = 2,
    **options: Any,
) -> SamplerResult:
    """Tempered SMC from the prior to the posterior, as ``n_runs`` independent runs.

    The particles are split over the runs (default two runs of 500), whose
    samples are pooled and whose evidences are averaged. The spread between
    the runs gives the evidence error: the incremental-weight variance of a
    single run misses the correlation the MCMC moves leave between particles
    and understated the run-to-run scatter by a factor 6-8 on profile fits
    with the baseline order marginalised.
    """
    n = n_particles or max(1000, 2 * n_live)
    runs = [
        _smc_once(loglike, prior_transform, ndim, seed=seed + 7 * i, n=max(n // n_runs, 50), **options)
        for i in range(max(n_runs, 1))
    ]
    lz = np.array([r.log_z for r in runs])
    log_z = float(logsumexp(lz) - np.log(lz.size))
    err_runs = float(np.std(lz, ddof=1) / np.sqrt(lz.size)) if lz.size > 1 else 0.0
    err_floor = float(np.sqrt(np.mean([r.stats["var_terms"] for r in runs]) / lz.size))
    return SamplerResult(
        samples=np.vstack([r.samples for r in runs]),
        log_weights=np.zeros(sum(r.samples.shape[0] for r in runs)),
        log_l=np.concatenate([r.log_l for r in runs]),
        log_z=log_z,
        log_z_err=max(err_runs, err_floor),
        backend="smc",
        runtime=sum(r.runtime for r in runs),
        n_like=sum(r.n_like for r in runs),
        stats={
            "n_runs": lz.size,
            "log_z_runs": lz.tolist(),
            "n_steps": int(np.mean([r.stats["n_steps"] for r in runs])),
            "n_particles": n,
            "mean_acceptance": float(np.mean([r.stats["mean_acceptance"] for r in runs])),
        },
        trace=runs[0].trace,
    )


def _smc_once(
    loglike: LogLike,
    prior_transform: Transform,
    ndim: int,
    *,
    seed: int,
    n: int,
    ess_fraction: float = 0.5,
    n_moves: int = 10,
    max_steps: int = 500,
    init_proposal: tuple[tuple[list[Any], np.ndarray], float] | None = None,
    **options: Any,
) -> SamplerResult:
    """One tempered SMC run with ``n`` particles.

    ``init_proposal = ((dists, mix), fraction)`` starts that fraction of the
    particles from a mixture of z-space distributions (the quick backend's
    importance proposal, centred on the modes it found) and the rest from the
    prior. The initial importance weights prior / q0 make the evidence exact
    for this defensive mixture, and q0 >= (1 - fraction) prior bounds them.
    """
    rng = np.random.default_rng(seed)
    like = CountingLikelihood(loglike)

    def eval_ll(z: np.ndarray) -> np.ndarray:
        u = np.clip(expit(z), 1e-15, 1 - 1e-15)
        return like(prior_transform(u))

    with Timer() as t:
        u = rng.uniform(size=(n, ndim))
        z = np.log(u) - np.log1p(-u)
        log_pz = _log_prior_z(z)
        base = np.zeros(n)
        proposal: tuple[list[Any], np.ndarray] | None = None
        if init_proposal is not None and init_proposal[1] > 0:
            (dists, mix), frac = init_proposal
            mix = np.asarray(mix, float) / np.sum(mix)
            proposal = (dists, mix)
            zs = _draw_mixture(dists, mix, round(frac * n), rng)
            if zs.shape[0]:
                z[: zs.shape[0]] = zs
                log_pz = _log_prior_z(z)
                log_q0 = np.logaddexp(np.log1p(-frac) + log_pz, np.log(frac) + _log_mixture(dists, mix, z))
                base = log_pz - log_q0
        log_l = eval_ll(z)
        beta = 0.0
        log_z = 0.0
        var_terms = 0.0
        betas = [0.0]
        acc_hist = []
        for _ in range(max_steps):
            new = _next_beta(log_l, beta, ess_fraction * n, base)
            lw = base + (new - beta) * log_l
            log_z += float(logsumexp(lw) - np.log(n))
            w = np.exp(lw - np.max(lw))
            var_terms += float(np.var(w) / (np.mean(w) ** 2 * n))
            idx = systematic_resample(w / w.sum(), n, rng)
            z, log_l, log_pz = z[idx], log_l[idx], log_pz[idx]
            base = np.zeros(n)
            beta = new
            betas.append(beta)
            cov = np.cov(z.T).reshape(ndim, ndim) + 1e-9 * np.eye(ndim)
            chol = np.linalg.cholesky(cov)
            scale = 2.38 / np.sqrt(ndim)
            acc_total = 0.0
            for m in range(n_moves):
                if proposal is not None and m % 2 == 1:
                    # independence move from the quick-stage mixture: lets particles jump
                    # into a narrow mode (a faint second line) that random walks cannot reach
                    prop = _draw_mixture(*proposal, n, rng)[rng.permutation(n)]
                    lq_cur = _log_mixture(*proposal, z)
                    lq_p = _log_mixture(*proposal, prop)
                    ll_p = eval_ll(prop)
                    lp_p = _log_prior_z(prop)
                    log_a = beta * (ll_p - log_l) + lp_p - log_pz + lq_cur - lq_p
                    acc = np.log(rng.uniform(size=n)) < log_a
                    z[acc], log_l[acc], log_pz[acc] = prop[acc], ll_p[acc], lp_p[acc]
                    continue
                prop = z + scale * rng.standard_normal((n, ndim)) @ chol.T
                ll_p = eval_ll(prop)
                lp_p = _log_prior_z(prop)
                log_a = beta * (ll_p - log_l) + lp_p - log_pz
                acc = np.log(rng.uniform(size=n)) < log_a
                z[acc], log_l[acc], log_pz[acc] = prop[acc], ll_p[acc], lp_p[acc]
                rate = acc.mean()
                acc_total += rate
                scale *= np.exp(rate - 0.3)
            acc_hist.append(acc_total / max(n_moves - (n_moves // 2 if proposal is not None else 0), 1))
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
        stats={
            "n_steps": len(betas) - 1,
            "n_particles": n,
            "mean_acceptance": float(np.mean(acc_hist)),
            "var_terms": var_terms,
        },
        trace={"beta": np.array(betas), "acceptance": np.array(acc_hist)},
    )
