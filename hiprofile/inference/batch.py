"""GPU path: many spectra of equal length at once.

``BatchedLikelihood`` stacks S spectra with the same channel count and width
(white noise, masks as zero weights) into one jitted, ``vmap``-ed JAX function
``theta (S, P, d) -> ln L (S, P)``. ``run_smc_batch`` runs the adaptive tempered
SMC of :mod:`hiprofile.samplers.smc` for all spectra simultaneously, so every
step is one XLA call over S x P parameter vectors (CPU or GPU).

``screen_batch`` uses both for a fast survey screening: evidences of N = 0
(analytic) and N = 1 (SMC) at a fixed baseline order K and the resulting
detection probability. Candidates are then fitted in full with
``hiprofile fit``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expit, logsumexp

from hiprofile._backend import get_xp
from hiprofile.config import Config
from hiprofile.inference.fit import prepare
from hiprofile.inference.priors import ModelPrior, build_prior
from hiprofile.model.kernel import SpectralGrid, chebyshev_basis, profile_basis
from hiprofile.spectrum import Spectrum

LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class BatchedLikelihood:
    grid: SpectralGrid  # reference grid; spectra differ only in v0
    v_shift: np.ndarray  # (S,) v0 of each spectrum minus grid.v0
    flux: np.ndarray  # (S, n)
    weight: np.ndarray  # (S, n) 1 / sigma, 0 for masked channels
    baseline_order: int
    amp_scale: np.ndarray  # (S,) prior std of a_ring
    baseline_scale: np.ndarray  # (S,)
    disk_scale_ratio: float = 1.0

    def __post_init__(self) -> None:
        import jax

        xp = get_xp("jax")
        _s, n = self.flux.shape
        t = chebyshev_basis(n, self.baseline_order)  # (nb, n)
        self.nb = t.shape[0]
        tw = t[None] * self.weight[:, None, :]  # (S, nb, n)
        rw = self.flux * self.weight
        self.n_good = (self.weight > 0).sum(axis=1)
        logdet_c = -2.0 * np.sum(np.log(np.where(self.weight > 0, self.weight, 1.0)), axis=1)
        amp = np.stack([self.amp_scale, self.amp_scale * self.disk_scale_ratio], axis=1)  # (S, 2)
        logdet_lam = 2.0 * np.sum(np.log(amp), axis=1) + 2.0 * self.nb * np.log(self.baseline_scale)
        g_tt = (
            np.einsum("skn,sjn->skj", tw, tw)
            + np.eye(self.nb)[None] / self.baseline_scale[:, None, None] ** 2
        )
        consts = dict(
            tw=xp.asarray(tw),
            rw=xp.asarray(rw),
            w=xp.asarray(self.weight),
            g_tt=xp.asarray(g_tt),
            b_t=xp.asarray(np.einsum("skn,sn->sk", tw, rw)),
            rr=xp.asarray(np.sum(rw**2, axis=1)),
            const=xp.asarray(logdet_c + logdet_lam + self.n_good * LOG_2PI),
            amp_prec=xp.asarray(1.0 / amp**2),
            shift=xp.asarray(self.v_shift),
        )
        grid = self.grid

        def one(theta, tw, rw, w, g_tt, b_t, rr, const, amp_prec, shift):
            th = theta.at[:, 0].add(-shift)
            bw = profile_basis(grid, th, "jax") * w  # (P, 2, n)
            g_bb = xp.einsum("pqn,prn->pqr", bw, bw) + xp.diag(amp_prec)
            g_bt = xp.einsum("pqn,kn->pqk", bw, tw)
            p = theta.shape[0]
            top = xp.concatenate([g_bb, g_bt], axis=2)
            bot = xp.concatenate([xp.swapaxes(g_bt, 1, 2), xp.broadcast_to(g_tt, (p, *g_tt.shape))], axis=2)
            a = xp.concatenate([top, bot], axis=1)
            bvec = xp.concatenate(
                [xp.einsum("pqn,n->pq", bw, rw), xp.broadcast_to(b_t, (p, b_t.shape[0]))], axis=1
            )
            chol = xp.linalg.cholesky(a)
            logdet_a = 2.0 * xp.sum(xp.log(xp.diagonal(chol, axis1=1, axis2=2)), axis=1)
            sol = xp.linalg.solve(a, bvec[..., None])[..., 0]
            ll = -0.5 * (rr - xp.sum(bvec * sol, axis=1) + const + logdet_a)
            return xp.where(xp.isfinite(ll), ll, -1e300)

        self._fn = jax.jit(jax.vmap(one))
        self._consts = consts
        # Null model (N = 0) evidence, analytic
        sol0 = np.linalg.solve(g_tt, np.asarray(consts["b_t"])[..., None])[..., 0]
        logdet0 = np.linalg.slogdet(g_tt)[1]
        logdet_lam0 = 2.0 * self.nb * np.log(self.baseline_scale)
        self.log_z_null = -0.5 * (
            np.sum(rw**2, axis=1)
            - np.sum(np.asarray(consts["b_t"]) * sol0, axis=1)
            + logdet_c
            + logdet_lam0
            + logdet0
            + self.n_good * LOG_2PI
        )

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        c = self._consts
        out = self._fn(
            get_xp("jax").asarray(theta),
            c["tw"],
            c["rw"],
            c["w"],
            c["g_tt"],
            c["b_t"],
            c["rr"],
            c["const"],
            c["amp_prec"],
            c["shift"],
        )
        return np.array(out)


def _log_prior_z(z: np.ndarray) -> np.ndarray:
    return -np.sum(np.logaddexp(0.0, z) + np.logaddexp(0.0, -z), axis=-1)


def _ess(lw: np.ndarray) -> np.ndarray:
    lw = lw - lw.max(axis=-1, keepdims=True)
    w = np.exp(lw)
    return w.sum(axis=-1) ** 2 / np.sum(w**2, axis=-1)


def run_smc_batch(
    like: BatchedLikelihood,
    priors: Sequence[ModelPrior],
    seed: int,
    n_particles: int = 1000,
    ess_fraction: float = 0.5,
    n_moves: int = 10,
    max_steps: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tempered SMC for S spectra at once. Returns (log_z (S,), theta (S, P, d), log_l (S, P))."""
    rng = np.random.default_rng(seed)
    s = len(priors)
    d = priors[0].ndim
    n = n_particles

    def transform(z: np.ndarray) -> np.ndarray:
        u = np.clip(expit(z), 1e-15, 1 - 1e-15)
        return np.stack([priors[i].transform(u[i]) for i in range(s)])

    u = rng.uniform(size=(s, n, d))
    z = np.log(u) - np.log1p(-u)
    log_l = like(transform(z))
    log_pz = _log_prior_z(z)
    beta = np.zeros(s)
    log_z = np.zeros(s)
    for _ in range(max_steps):
        active = beta < 1.0
        if not active.any():
            break
        lo, hi = beta.copy(), np.ones(s)
        target = ess_fraction * n
        done = _ess((1.0 - beta)[:, None] * log_l) >= target
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            ok = _ess((mid - beta)[:, None] * log_l) >= target
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        new = np.where(done, 1.0, np.maximum(lo, beta + 1e-8))
        new = np.where(active, new, 1.0)
        lw = (new - beta)[:, None] * log_l
        log_z += np.where(active, logsumexp(lw, axis=1) - np.log(n), 0.0)
        w = np.exp(lw - lw.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        for i in np.flatnonzero(active):
            pos = (rng.uniform() + np.arange(n)) / n
            idx = np.minimum(np.searchsorted(np.cumsum(w[i]), pos), n - 1)
            z[i], log_l[i], log_pz[i] = z[i, idx], log_l[i, idx], log_pz[i, idx]
        beta = new
        scale = np.full(s, 2.38 / math.sqrt(d))
        chol = np.stack(
            [np.linalg.cholesky(np.cov(z[i].T).reshape(d, d) + 1e-9 * np.eye(d)) for i in range(s)]
        )
        for _ in range(n_moves):
            prop = z + scale[:, None, None] * np.einsum("snd,sed->sne", rng.standard_normal((s, n, d)), chol)
            ll_p = like(transform(prop))
            lp_p = _log_prior_z(prop)
            log_a = beta[:, None] * (ll_p - log_l) + lp_p - log_pz
            acc = (np.log(rng.uniform(size=(s, n))) < log_a) & active[:, None]
            z = np.where(acc[..., None], prop, z)
            log_l = np.where(acc, ll_p, log_l)
            log_pz = np.where(acc, lp_p, log_pz)
            scale *= np.exp(acc.mean(axis=1) - 0.3)
    return log_z, transform(z), log_l


def screen_batch(
    spectra: Sequence[Spectrum], cfg: Config | None = None, k: int = 1, n_particles: int = 1000, seed: int = 0
) -> list[dict[str, Any]]:
    """Evidences for N = 0 and N = 1 at baseline order ``k`` for equal-length spectra.

    The N = 1 evidence is the Gaussian-marginal one (no positivity
    truncation); the detection probability uses ``cfg.model.poisson_rate``.
    """
    cfg = cfg or Config()
    ctxs = [prepare(sp, cfg, seed + i) for i, sp in enumerate(spectra)]
    g0 = ctxs[0].grid
    for c in ctxs:
        if c.grid.n_chan != g0.n_chan or abs(c.grid.dv - g0.dv) > 1e-9 * g0.dv or not c.noise.white:
            raise ValueError("screen_batch needs equal channel grids and white noise")
    weight = np.zeros((len(ctxs), g0.n_chan))
    for i, c in enumerate(ctxs):
        weight[i, c.noise.good_index] = 1.0 / c.noise.sigma
    like = BatchedLikelihood(
        grid=g0,
        v_shift=np.array([c.grid.v0 - g0.v0 for c in ctxs]),
        flux=np.array([np.where(c.good, c.spectrum.flux, 0.0) for c in ctxs]),
        weight=weight,
        baseline_order=k,
        amp_scale=np.array([c.amp_scale for c in ctxs]),
        disk_scale_ratio=cfg.priors.disk_scale_ratio,
        baseline_scale=np.array([c.baseline_scale for c in ctxs]),
    )
    priors = [build_prior(("profile",), cfg.priors, c.spectrum.velocity) for c in ctxs]
    log_z1, theta, _log_l = run_smc_batch(like, priors, seed, n_particles=n_particles)
    rate = cfg.model.poisson_rate
    out = []
    for i, c in enumerate(ctxs):
        l0 = float(like.log_z_null[i]) - rate
        l1 = float(log_z1[i]) + math.log(rate) - rate
        p1 = float(np.exp(l1 - np.logaddexp(l0, l1)))
        med = np.median(theta[i], axis=0)
        out.append(
            {
                "id": c.spectrum.id,
                "log_z_n0": float(like.log_z_null[i]),
                "log_z_n1": float(log_z1[i]),
                "p_detect": p1,
                **{f"{nm}_median": float(v) for nm, v in zip(priors[i].names, med, strict=True)},
            }
        )
    return out
