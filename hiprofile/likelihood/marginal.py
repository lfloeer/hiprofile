"""Likelihood with the linear parameters marginalised analytically.

For fixed nonlinear parameters theta the model is linear, m = M beta, with the
design matrix M = [component basis columns, Chebyshev T_0..T_K]. With the prior
beta ~ N(mu, Lambda) and noise covariance C,

    y | theta ~ N(M mu, C + M Lambda M^T).

Using whitened quantities (L^-1 with C = L L^T) and the Woodbury identity only a
p x p system with p = n_amplitudes + K + 1 has to be solved:

    A    = Lambda^-1 + M~^T M~
    b    = M~^T r~,          r~ = L^-1 (y - M mu)
    ln L = -1/2 [ r~^T r~ - b^T A^-1 b + ln|C| + ln|Lambda| + ln|A| + n ln 2 pi ]

and beta | theta, y ~ N(mu + A^-1 b, A^-1).

The component amplitudes have a zero-mean Gaussian prior that is truncated to
beta >= 0 afterwards (see :mod:`hiprofile.inference.amplitudes`); the
evidence of the truncated model differs by the posterior orthant probability.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hiprofile._backend import get_xp
from hiprofile.likelihood.noise import NoiseModel
from hiprofile.model.kernel import (
    GAUSS_NPARAM,
    PROFILE_NPARAM,
    SpectralGrid,
    chebyshev_basis,
    gaussian_basis,
    profile_basis,
)

NPARAM = {"profile": PROFILE_NPARAM, "gauss": GAUSS_NPARAM}
NAMP = {"profile": 2, "gauss": 1}
LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class MarginalLikelihood:
    """Marginal likelihood p(y | theta) for one spectrum and one model (components, K).

    Parameters
    ----------
    flux : (n_chan,) data on the channel grid (masked channels ignored)
    grid : channel grid incl. spectrometer response
    noise : noise on the good channels
    components : component kinds, e.g. ("profile", "profile", "gauss")
    baseline_order : Chebyshev order K (-1: no baseline)
    amp_scale : prior std of each component amplitude (per amplitude column)
    baseline_mean, baseline_cov : Gaussian prior of the Chebyshev coefficients
    noise_scale : if True, theta carries a trailing log10 noise scale factor
    """

    flux: np.ndarray
    grid: SpectralGrid
    noise: NoiseModel
    components: tuple[str, ...]
    baseline_order: int
    amp_scale: np.ndarray
    baseline_mean: np.ndarray
    baseline_cov: np.ndarray
    noise_scale: bool = False
    backend: str = "numpy"
    _pre: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.components = tuple(self.components)
        for c in self.components:
            if c not in NPARAM:
                raise ValueError(f"unknown component kind {c!r}")
        self.amp_scale = np.broadcast_to(np.asarray(self.amp_scale, dtype=float), (self.n_amp,)).copy()
        nb = self.baseline_order + 1
        self.baseline_mean = np.broadcast_to(np.asarray(self.baseline_mean, float), (nb,)).copy()
        self.baseline_cov = np.asarray(self.baseline_cov, float).reshape(nb, nb)
        self._precompute()

    # ------------------------------------------------------------------ layout
    @property
    def n_amp(self) -> int:
        return sum(NAMP[c] for c in self.components)

    @property
    def n_base(self) -> int:
        return self.baseline_order + 1

    @property
    def n_linear(self) -> int:
        return self.n_amp + self.n_base

    @property
    def n_theta(self) -> int:
        return sum(NPARAM[c] for c in self.components) + (1 if self.noise_scale else 0)

    @property
    def theta_slices(self) -> list[slice]:
        out, i = [], 0
        for c in self.components:
            out.append(slice(i, i + NPARAM[c]))
            i += NPARAM[c]
        return out

    @property
    def good_index(self) -> np.ndarray:
        return self.noise.good_index

    # ------------------------------------------------------------ precompute
    def _precompute(self) -> None:
        xp = get_xp(self.backend)
        idx = self.good_index
        y = np.asarray(self.flux, float)[idx]
        tb = chebyshev_basis(self.grid.n_chan, self.baseline_order)[:, idx]  # (nb, n)
        r0 = y - self.baseline_mean @ tb if self.n_base else y
        tw = self.noise.whiten(tb, "numpy") if self.n_base else tb
        rw = self.noise.whiten(r0, "numpy")
        prior_prec = np.zeros(self.n_linear)
        lam_b_inv = np.linalg.inv(self.baseline_cov) if self.n_base else np.zeros((0, 0))
        logdet_lam = 2.0 * np.sum(np.log(self.amp_scale))
        if self.n_base:
            logdet_lam += np.linalg.slogdet(self.baseline_cov)[1]
        prior_prec[: self.n_amp] = 1.0 / self.amp_scale**2
        self._pre.update(
            y=y,
            tb=tb,
            tw=xp.asarray(tw),
            rw=xp.asarray(rw),
            rr=float(rw @ rw),
            g_tt=xp.asarray(tw @ tw.T + lam_b_inv),
            b_t=xp.asarray(tw @ rw),
            amp_prec=xp.asarray(prior_prec[: self.n_amp]),
            logdet_lam=float(logdet_lam),
            logdet_c=self.noise.logdet,
            n=int(idx.size),
            idx=xp.asarray(idx),
        )
        if self.backend == "jax":
            import jax

            self._jit_terms = jax.jit(self._terms)

    # ---------------------------------------------------------------- model
    def basis(self, theta):
        """Component basis on all channels, shape (batch, n_amp, n_chan)."""
        xp = get_xp(self.backend)
        theta = xp.atleast_2d(xp.asarray(theta, dtype=float))
        cols = []
        for kind, sl in zip(self.components, self.theta_slices, strict=True):
            if kind == "profile":
                cols.append(profile_basis(self.grid, theta[:, sl], self.backend))
            else:
                cols.append(gaussian_basis(self.grid, theta[:, sl], self.backend))
        if not cols:
            return xp.zeros((theta.shape[0], 0, self.grid.n_chan))
        return xp.concatenate(cols, axis=1)

    def design(self, theta) -> np.ndarray:
        """Full design matrix (batch, n_linear, n_chan) as a NumPy array."""
        b = np.asarray(self.basis(theta))
        t = chebyshev_basis(self.grid.n_chan, self.baseline_order)
        t = np.broadcast_to(t, (b.shape[0], *t.shape))
        return np.concatenate([b, t], axis=1)

    def model(self, theta, beta) -> np.ndarray:
        """Model spectra (batch, n_chan) for theta and linear parameters beta."""
        m = self.design(theta)
        return np.einsum("bpn,bp->bn", m, np.atleast_2d(beta))

    # ------------------------------------------------------------ algebra
    def _terms(self, theta):
        xp = get_xp(self.backend)
        p = self._pre
        theta = xp.atleast_2d(theta)
        nb = theta.shape[0]
        if self.n_amp:
            basis = self.basis(theta)[:, :, p["idx"]]
            bw = self.noise.whiten(basis, self.backend)  # (b, q, n)
        else:
            bw = xp.zeros((nb, 0, p["n"]))
        tw, rw = p["tw"], p["rw"]
        rr = p["rr"]
        logdet_c = p["logdet_c"]
        if self.noise_scale:
            s = 10.0 ** theta[:, -1]
            inv = 1.0 / s
            bw = bw * inv[:, None, None]
            scale2 = inv**2
            logdet_c = logdet_c + 2.0 * p["n"] * xp.log(s)
        else:
            scale2 = xp.ones(nb)
        g_bb = xp.einsum("bqn,brn->bqr", bw, bw) + xp.eye(self.n_amp) * p["amp_prec"]
        b_b = xp.einsum("bqn,n->bq", bw, rw)
        if self.noise_scale:
            b_b = b_b * (1.0 / 10.0 ** theta[:, -1])[:, None]
        if self.n_base:
            g_bt = xp.einsum("bqn,kn->bqk", bw, tw)
            if self.noise_scale:
                g_bt = g_bt * (1.0 / 10.0 ** theta[:, -1])[:, None, None]
                lam_b_inv = p["g_tt"] - tw @ tw.T
                g_tt = (tw @ tw.T)[None] * scale2[:, None, None] + lam_b_inv[None]
                b_t = p["b_t"][None] * scale2[:, None]
            else:
                g_tt = xp.broadcast_to(p["g_tt"], (nb, *p["g_tt"].shape))
                b_t = xp.broadcast_to(p["b_t"], (nb, self.n_base))
            top = xp.concatenate([g_bb, g_bt], axis=2)
            bot = xp.concatenate([xp.swapaxes(g_bt, 1, 2), g_tt], axis=2)
            a = xp.concatenate([top, bot], axis=1)
            bvec = xp.concatenate([b_b, b_t], axis=1)
        else:
            a, bvec = g_bb, b_b
        return a, bvec, rr * scale2, logdet_c

    def loglike(self, theta):
        """ln p(y | theta), vectorised over the leading axis of ``theta``."""
        xp = get_xp(self.backend)
        p = self._pre
        if self.n_linear == 0:
            n = p["n"]
            val = -0.5 * (p["rr"] + p["logdet_c"] + n * LOG_2PI)
            return np.full(np.atleast_2d(theta).shape[0], val)
        terms = self._jit_terms if self.backend == "jax" else self._terms
        a, bvec, rr, logdet_c = terms(xp.asarray(theta, dtype=float))
        chol = xp.linalg.cholesky(a)
        logdet_a = 2.0 * xp.sum(xp.log(xp.diagonal(chol, axis1=1, axis2=2)), axis=1)
        sol = xp.linalg.solve(a, bvec[..., None])[..., 0]
        chi2 = rr - xp.sum(bvec * sol, axis=1)
        ll = -0.5 * (chi2 + logdet_c + p["logdet_lam"] + logdet_a + p["n"] * LOG_2PI)
        ll = xp.where(xp.isfinite(ll), ll, -1e300)
        return np.asarray(ll) if self.backend == "numpy" else ll

    def loglike_null(self) -> float:
        """ln p(y) of the baseline-only model (no components, no theta)."""
        if self.n_amp:
            raise ValueError("loglike_null is only defined without components")
        return float(np.asarray(self.loglike(np.zeros((1, max(self.n_theta, 1)))))[0])

    def conditional(self, theta) -> tuple[np.ndarray, np.ndarray]:
        """Mean (batch, p) and covariance (batch, p, p) of beta | theta, y."""
        xp = get_xp(self.backend)
        if self.n_linear == 0:
            nb = np.atleast_2d(theta).shape[0]
            return np.zeros((nb, 0)), np.zeros((nb, 0, 0))
        terms = self._jit_terms if self.backend == "jax" else self._terms
        a, bvec, _, _ = terms(xp.asarray(theta, dtype=float))
        a = np.asarray(a)
        bvec = np.asarray(bvec)
        cov = np.linalg.inv(a)
        cov = 0.5 * (cov + np.swapaxes(cov, 1, 2))
        mean = np.einsum("bpq,bq->bp", cov, bvec)
        mu = np.r_[np.zeros(self.n_amp), self.baseline_mean]
        return mean + mu, cov

    def __call__(self, theta):
        return self.loglike(theta)


def default_amp_scale(
    flux: np.ndarray, good: np.ndarray, sigma_rms: float, dv: float, factor: float = 10.0
) -> float:
    """Empirical scale of the half-normal amplitude prior (integrated flux units).

    The larger of ``factor`` times the noise of the integrated flux over the
    whole band and twice the integrated positive flux in the band, so that the
    prior neither dominates faint detections nor biases bright sources.
    """
    n = int(np.sum(good))
    band_noise = sigma_rms * dv * math.sqrt(n)
    total = float(np.sum(np.where(good, flux, 0.0)) * dv)
    return max(factor * band_noise, 2.0 * total)


def build_likelihood(
    flux: np.ndarray,
    grid: SpectralGrid,
    noise: NoiseModel,
    components: Sequence[str],
    baseline_order: int,
    amp_scale: float | np.ndarray,
    baseline_scale: float,
    baseline_mean: np.ndarray | None = None,
    baseline_cov: np.ndarray | None = None,
    noise_scale: bool = False,
    backend: str = "numpy",
) -> MarginalLikelihood:
    nb = baseline_order + 1
    mean = np.zeros(nb) if baseline_mean is None else np.asarray(baseline_mean)[:nb]
    cov = np.eye(nb) * baseline_scale**2 if baseline_cov is None else np.asarray(baseline_cov)[:nb, :nb]
    return MarginalLikelihood(
        flux=flux,
        grid=grid,
        noise=noise,
        components=tuple(components),
        baseline_order=baseline_order,
        amp_scale=np.asarray(amp_scale, dtype=float),
        baseline_mean=mean,
        baseline_cov=cov,
        noise_scale=noise_scale,
        backend=backend,
    )
