"""Quick-look backend: multi-start MAP, Laplace approximation and importance sampling.

The optimisation runs in z = logit(u), where u is the unit-cube coordinate, so
that the prior density is p(z) = prod u (1 - u) and the problem is unbounded.
Around each distinct optimum the Laplace approximation gives a Gaussian; a
mixture of multivariate-t proposals built from them is used for importance
sampling, which corrects the Laplace evidence and yields weighted posterior
samples. Multimodality is only captured if the starts find every mode.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize, stats
from scipy.special import expit, logsumexp

from hiprofile.samplers.base import CountingLikelihood, LogLike, SamplerResult, Timer, Transform

LOG_2PI = np.log(2.0 * np.pi)


class _Target:
    def __init__(self, like: CountingLikelihood, transform: Transform) -> None:
        self.like = like
        self.transform = transform

    def __call__(self, z: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(z)
        u = expit(z)
        u = np.clip(u, 1e-15, 1 - 1e-15)
        lp = np.sum(np.log(u) + np.log1p(-u), axis=1)
        return self.like(self.transform(u)) + lp


def _grad(target: _Target, z: np.ndarray, h: float = 1e-5) -> tuple[float, np.ndarray]:
    d = z.size
    pts = np.vstack([z, z + h * np.eye(d), z - h * np.eye(d)])
    f = target(pts)
    return float(f[0]), (f[1 : d + 1] - f[d + 1 :]) / (2 * h)


def hessian(target: _Target, z: np.ndarray, h: float = 1e-3) -> np.ndarray:
    """Central finite-difference Hessian of log p at z (one batched call)."""
    d = z.size
    e = np.eye(d) * h
    pts = [z]
    for i in range(d):
        pts += [z + e[i], z - e[i]]
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    for i, j in pairs:
        pts += [z + e[i] + e[j], z + e[i] - e[j], z - e[i] + e[j], z - e[i] - e[j]]
    f = target(np.array(pts))
    f0 = f[0]
    hmat = np.empty((d, d))
    for i in range(d):
        hmat[i, i] = (f[1 + 2 * i] - 2 * f0 + f[2 + 2 * i]) / h**2
    off = 1 + 2 * d
    for k, (i, j) in enumerate(pairs):
        a, b, c, dd = f[off + 4 * k : off + 4 * k + 4]
        hmat[i, j] = hmat[j, i] = (a - b - c + dd) / (4 * h**2)
    return hmat


# The prior in z = logit(u) is logistic with variance pi^2 / 3; no posterior
# direction can be wider, so precision eigenvalues are clipped at its inverse.
PRECISION_FLOOR = 3.0 / np.pi**2


def _precision(hmat: np.ndarray, floor: float = PRECISION_FLOOR) -> tuple[np.ndarray, np.ndarray]:
    """(precision, covariance) of -H made positive definite by clipping eigenvalues."""
    w, v = np.linalg.eigh(-0.5 * (hmat + hmat.T))
    w = np.maximum(w, floor)
    prec = (v * w) @ v.T
    cov = (v / w) @ v.T
    return 0.5 * (prec + prec.T), 0.5 * (cov + cov.T)


def find_modes(
    target: _Target,
    ndim: int,
    rng: np.random.Generator,
    n_init: int,
    n_starts: int,
    maxiter: int = 200,
    init_u: np.ndarray | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Distinct local maxima of log p(z), best first.

    Starts are the best of ``n_init`` random points, or the given ``init_u``
    (warm start, e.g. the modes of a neighbouring model).
    """
    if init_u is not None:
        u0 = np.clip(np.atleast_2d(init_u), 1e-6, 1 - 1e-6)
        n_starts = u0.shape[0]
    else:
        u0 = rng.uniform(0.02, 0.98, size=(n_init, ndim))
    z0 = np.log(u0) - np.log1p(-u0)
    f0 = target(z0)
    order = np.argsort(f0)[::-1]
    modes: list[tuple[np.ndarray, float]] = []
    for idx in order[: n_starts * 3]:
        if len(modes) >= n_starts:
            break
        start = z0[idx]
        if any(np.max(np.abs(expit(start) - expit(m))) < 0.02 for m, _ in modes):
            continue

        def fun(z: np.ndarray) -> tuple[float, np.ndarray]:
            f, g = _grad(target, z)
            return -f, -g

        res = optimize.minimize(fun, start, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
        z = np.clip(res.x, -30, 30)
        val = float(target(z)[0])
        dup = False
        for k, (m, v) in enumerate(modes):
            if np.max(np.abs(expit(z) - expit(m))) < 0.01:
                dup = True
                if val > v:
                    modes[k] = (z, val)
                break
        if not dup:
            modes.append((z, val))
    modes.sort(key=lambda t: -t[1])
    return modes


def _importance(target, dists, mix, n, rng, ndim):
    counts = rng.multinomial(n, mix)
    zs = np.vstack(
        [
            np.atleast_2d(d.rvs(size=max(c, 1))).reshape(-1, ndim)[:c]
            for d, c in zip(dists, counts, strict=True)
        ]
    )
    log_q = logsumexp(
        np.array([np.log(m) + d.logpdf(zs) for m, d in zip(mix, dists, strict=True)]).reshape(len(dists), -1),
        axis=0,
    )
    log_p = target(zs)
    return zs, log_p, log_p - log_q


def run_quick(
    loglike: LogLike,
    prior_transform: Transform,
    ndim: int,
    *,
    seed: int,
    n_live: int = 400,
    n_samples: int = 4000,
    n_starts: int = 4,
    n_init: int | None = None,
    df: float = 5.0,
    inflate: float = 1.3,
    mode_window: float = 20.0,
    n_adapt: int = 2,
    init_u: np.ndarray | None = None,
    **options: Any,
) -> SamplerResult:
    rng = np.random.default_rng(seed)
    like = CountingLikelihood(loglike)
    target = _Target(like, prior_transform)
    with Timer() as t:
        modes = find_modes(target, ndim, rng, n_init or max(256, 64 * ndim), n_starts, init_u=init_u)
        best = modes[0][1]
        comps = []
        for z, val in modes:
            if val < best - mode_window:
                continue
            prec, cov = _precision(hessian(target, z))
            logdet_prec = np.linalg.slogdet(prec)[1]
            log_laplace = val + 0.5 * ndim * LOG_2PI - 0.5 * logdet_prec
            comps.append((z, cov, log_laplace))
        log_lap = np.array([c[2] for c in comps])
        log_z_laplace = float(logsumexp(log_lap))
        mix = np.exp(log_lap - log_z_laplace)
        # Defensive mixture: always keep 10 % on the best mode
        mix = 0.9 * mix + 0.1 * (np.arange(len(comps)) == 0)
        dists = [stats.multivariate_t(loc=z, shape=cov * inflate**2, df=df, seed=rng) for z, cov, _ in comps]
        n_round = n_samples // 2 if n_adapt else n_samples
        zs, log_p, log_w = _importance(target, dists, mix, n_round, rng, ndim)
        # Adaptive importance sampling: add a moment-matched t component fitted
        # to the weighted samples, keep the Laplace components for robustness.
        for _ in range(n_adapt):
            w = np.exp(log_w - log_w.max())
            w /= w.sum()
            mean = w @ zs
            cov = np.cov(zs.T, aweights=w).reshape(ndim, ndim) + 1e-9 * np.eye(ndim)
            try:
                extra = stats.multivariate_t(loc=mean, shape=cov, df=df, seed=rng)
            except (ValueError, np.linalg.LinAlgError):
                break
            dists = [*dists[: len(comps)], extra]
            mix = np.r_[0.4 * mix[: len(comps)] / mix[: len(comps)].sum(), 0.6]
            zs, log_p, log_w = _importance(target, dists, mix, n_samples, rng, ndim)
        log_z = float(logsumexp(log_w) - np.log(zs.shape[0]))
        w = np.exp(log_w - log_w.max())
        ess = float(w.sum() ** 2 / np.sum(w**2))
        rel_var = np.var(w) / max(np.mean(w) ** 2, 1e-300)
        log_z_err = float(np.sqrt(rel_var / zs.shape[0]))
        u = np.clip(expit(zs), 1e-15, 1 - 1e-15)
        theta = prior_transform(u)
        log_l = log_p - np.sum(np.log(u) + np.log1p(-u), axis=1)
    return SamplerResult(
        samples=theta,
        log_weights=log_w,
        log_l=log_l,
        log_z=log_z,
        log_z_err=log_z_err,
        backend="quick",
        runtime=t.elapsed,
        n_like=like.n_calls,
        stats={
            "log_z_laplace": log_z_laplace,
            "n_modes": len(comps),
            "map_u": expit(modes[0][0]).tolist(),
            "map_log_post": modes[0][1],
            "is_ess": ess,
            "modes_u": [expit(z).tolist() for z, _, _ in comps],
        },
    )
