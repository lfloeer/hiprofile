"""Likelihood with the linear parameters and the baseline order marginalised analytically.

For fixed nonlinear parameters theta the model is linear, m = M beta, with the
design matrix M = [component columns, Chebyshev T_0..T_Kmax]. A profile
component contributes one column (1 - f) R + f D with the integrated flux S as
its amplitude; an RFI component one Gaussian column. With the prior
beta ~ N(mu, Lambda) and noise covariance C,

    y | theta, K ~ N(M_K mu_K, C + M_K Lambda_K M_K^T),

where M_K keeps the component columns and T_0..T_K. Using whitened quantities
(L^-1 with C = L L^T) and the Woodbury identity only a p x p system with
p = n_amplitudes + K + 1 has to be solved:

    A    = Lambda^-1 + M~^T M~
    b    = M~^T r~,          r~ = L^-1 (y - M mu)
    ln L = -1/2 [ r~^T r~ - b^T A^-1 b + ln|C| + ln|Lambda| + ln|A| + n ln 2 pi ]

and beta | theta, y, K ~ N(mu + A^-1 b, A^-1).

The baseline order is marginalised inside the likelihood,

    p(y | theta) = sum_K P(K) p(y | theta, K),     K = k_min..k_max, P(K) flat,

with one Cholesky factorisation: A_K is the leading principal block of A, so
ln|A_K| and b_K^T A_K^-1 b_K are partial sums of the full factorisation. This
needs independent zero-mean coefficient priors (the default); a baseline
prior from neighbour spectra (mean or full covariance) fixes K = k_max.

The amplitudes have half-normal priors (zero-mean Gaussians truncated to
beta >= 0). The orthant probability P(a >= 0 | theta, y, K) enters the
likelihood per K (exact for up to two amplitudes, see
:mod:`hiprofile.likelihood.orthant`); a remaining cross-amplitude correction
for three or more components is applied by :mod:`hiprofile.inference.amplitudes`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import logsumexp

from hiprofile._backend import get_xp
from hiprofile.likelihood import fast as _fast_ll
from hiprofile.likelihood.noise import NoiseModel
from hiprofile.likelihood.orthant import LOG2, log_orthant
from hiprofile.model.kernel import (
    GAUSS_NPARAM,
    PROFILE_NPARAM,
    SpectralGrid,
    chebyshev_basis,
    gaussian_basis,
    profile_basis,
)

# nonlinear parameters per component: profile (v_ctr, delta_v, delta_v_rand, alpha, f)
NPARAM = {"profile": PROFILE_NPARAM + 1, "gauss": GAUSS_NPARAM}
# linear amplitudes per component: S for a profile, the flux of an RFI Gaussian
NAMP = {"profile": 1, "gauss": 1}
PROFILE_THETA = ("v_ctr", "delta_v", "delta_v_rand", "alpha", "f")
LOG_2PI = math.log(2.0 * math.pi)
COMPILED_CHUNK = 64
NEG_INF = -1e300


@dataclass
class MarginalLikelihood:
    """Marginal likelihood p(y | theta) for one spectrum and one set of components.

    Parameters
    ----------
    flux : (n_chan,) data on the channel grid (masked channels ignored)
    grid : channel grid incl. spectrometer response
    noise : noise on the good channels
    components : component kinds, e.g. ("profile", "profile", "gauss")
    baseline_order : largest Chebyshev order K_max (-1: no baseline)
    k_min : smallest order of the K mixture (default: K_max, i.e. a single order)
    amp_scale : prior std of each component amplitude
    baseline_mean, baseline_cov : Gaussian prior of the Chebyshev coefficients
    noise_scale : if True, theta carries a trailing log10 noise scale factor
    positivity : include P(amplitudes >= 0 | theta, y, K) / P_prior in the likelihood
    """

    flux: np.ndarray
    grid: SpectralGrid
    noise: NoiseModel
    components: tuple[str, ...]
    baseline_order: int
    amp_scale: np.ndarray
    baseline_mean: np.ndarray
    baseline_cov: np.ndarray
    k_min: int | None = None
    noise_scale: bool = False
    backend: str = "numpy"
    positivity: bool = True
    _pre: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.components = tuple(self.components)
        for c in self.components:
            if c not in NPARAM:
                raise ValueError(f"unknown component kind {c!r}")
        if self.baseline_order < -1:
            raise ValueError("baseline_order must be >= -1")
        if self.k_min is None:
            self.k_min = self.baseline_order
        if not -1 <= self.k_min <= self.baseline_order:
            raise ValueError("k_min must satisfy -1 <= k_min <= baseline_order")
        self.amp_scale = np.broadcast_to(np.asarray(self.amp_scale, dtype=float), (self.n_amp,)).copy()
        nb = self.n_base
        self.baseline_mean = np.broadcast_to(np.asarray(self.baseline_mean, float), (nb,)).copy()
        self.baseline_cov = np.asarray(self.baseline_cov, float).reshape(nb, nb)
        if self.n_k > 1:
            diag = np.diag(np.diag(self.baseline_cov))
            if np.any(self.baseline_mean != 0.0) or not np.allclose(self.baseline_cov, diag):
                raise ValueError(
                    "marginalising over the baseline order needs independent zero-mean coefficient "
                    "priors; use a single order (k_min = baseline_order) with an informative prior"
                )
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
    def k_values(self) -> list[int]:
        assert self.k_min is not None
        return list(range(self.k_min, self.baseline_order + 1))

    @property
    def n_k(self) -> int:
        assert self.k_min is not None
        return self.baseline_order - self.k_min + 1

    @property
    def log_prior_k(self) -> float:
        """ln P(K) of the flat prior over the mixed orders."""
        return -math.log(self.n_k)

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
        lam_b_inv = np.linalg.inv(self.baseline_cov) if self.n_base else np.zeros((0, 0))
        logdet_lam_amp = 2.0 * float(np.sum(np.log(self.amp_scale)))
        # ln|Lambda_K| - ln|Lambda_amp| for each K as cumulative sums over the coefficients
        if self.n_base:
            if self.n_k > 1:
                log_lam_b = np.log(np.diag(self.baseline_cov))
            else:  # a single order may have a full covariance
                log_lam_b = np.full(self.n_base, np.linalg.slogdet(self.baseline_cov)[1] / self.n_base)
        else:
            log_lam_b = np.zeros(0)
        lam_cum = np.concatenate([[0.0], np.cumsum(log_lam_b)])
        self._pre.update(
            y=y,
            tb=tb,
            tw=xp.asarray(tw),
            rw=xp.asarray(rw),
            rr=float(rw @ rw),
            g_tt=xp.asarray(tw @ tw.T + lam_b_inv),
            b_t=xp.asarray(tw @ rw),
            amp_prec=xp.asarray(1.0 / self.amp_scale**2),
            logdet_lam_amp=logdet_lam_amp,
            log_lam_b=log_lam_b,
            logdet_lam_k=xp.asarray([logdet_lam_amp + lam_cum[k + 1] for k in self.k_values]),
            logdet_c=self.noise.logdet,
            n=int(idx.size),
            idx=xp.asarray(idx),
        )
        if self.backend == "jax":
            import jax

            self._jit_moments = jax.jit(self._moments_generic)

    # ---------------------------------------------------------------- model
    def basis(self, theta):
        """Component columns on all channels, shape (batch, n_amp, n_chan).

        A profile column is the unit-flux combination (1 - f) R + f D of the
        ring and disk kernels; an RFI column a unit-flux Gaussian.
        """
        xp = get_xp(self.backend)
        theta = xp.atleast_2d(xp.asarray(theta, dtype=float))
        cols = []
        for kind, sl in zip(self.components, self.theta_slices, strict=True):
            th = theta[:, sl]
            if kind == "profile":
                rd = profile_basis(self.grid, th[:, :PROFILE_NPARAM], self.backend)  # (b, 2, n)
                f = th[:, PROFILE_NPARAM][:, None]
                cols.append(((1.0 - f) * rd[:, 0] + f * rd[:, 1])[:, None, :])
            else:
                cols.append(gaussian_basis(self.grid, th, self.backend))
        if not cols:
            return xp.zeros((theta.shape[0], 0, self.grid.n_chan))
        return xp.concatenate(cols, axis=1)

    def design(self, theta) -> np.ndarray:
        """Full design matrix (batch, n_linear, n_chan) as a NumPy array (all orders up to K_max)."""
        b = np.asarray(self.basis(theta))
        t = chebyshev_basis(self.grid.n_chan, self.baseline_order)
        t = np.broadcast_to(t, (b.shape[0], *t.shape))
        return np.concatenate([b, t], axis=1)

    def model(self, theta, beta) -> np.ndarray:
        """Model spectra (batch, n_chan) for theta and linear parameters beta (zero-padded)."""
        m = self.design(theta)
        return np.einsum("bpn,bp->bn", m, np.atleast_2d(beta))

    # ------------------------------------------------------------ algebra
    def _terms(self, theta):
        """A (b, p, p), b (b, p), r~^T r~ (b,), ln|C| (b,) for the full column set."""
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
        return a, bvec, rr * scale2, logdet_c + xp.zeros(nb)

    def _moments_generic(self, theta):
        """Per order K: ln L_K (b, nK), amplitude mean (b, nK, q) and covariance (b, nK, q, q)."""
        xp = get_xp(self.backend)
        p = self._pre
        a, bvec, rr, logdet_c = self._terms(theta)
        q = self.n_amp
        lls, means, covs = [], [], []
        for kk, k in enumerate(self.k_values):
            m = q + k + 1
            ak = a[:, :m, :m]
            bk = bvec[:, :m]
            chol = xp.linalg.cholesky(ak)
            logdet = 2.0 * xp.sum(xp.log(xp.diagonal(chol, axis1=1, axis2=2)), axis=1)
            inv = xp.linalg.inv(ak)
            sol = xp.einsum("bij,bj->bi", inv, bk)
            quad = xp.sum(bk * sol, axis=1)
            ll = -0.5 * (rr - quad + logdet_c + p["logdet_lam_k"][kk] + logdet + p["n"] * LOG_2PI)
            lls.append(xp.where(xp.isfinite(ll), ll, NEG_INF))
            means.append(sol[:, :q])
            covs.append(inv[:, :q, :q])
        return xp.stack(lls, axis=1), xp.stack(means, axis=1), xp.stack(covs, axis=1)

    @property
    def _compiled(self) -> bool:
        return self.backend == "numpy" and not self.noise_scale and self.n_amp > 0 and _fast_ll.available()

    def _fast_args(self) -> tuple:
        p = self._pre
        if "fast" not in p:
            chol = self.noise.sigma[None, :] if self.noise.white else self.noise.cholesky_banded()
            const = p["logdet_c"] + p["n"] * LOG_2PI + p["logdet_lam_amp"]
            tw = np.ascontiguousarray(np.asarray(p["tw"], dtype=float).reshape(self.n_base, self.noise.n))
            p["fast"] = (
                np.ascontiguousarray(self.good_index, dtype=np.int64),
                np.ascontiguousarray(chol, dtype=float),
                tw,
                np.ascontiguousarray(p["rw"]),
                np.ascontiguousarray(np.asarray(p["g_tt"]).reshape(self.n_base, self.n_base)),
                np.ascontiguousarray(np.asarray(p["b_t"]).reshape(self.n_base)),
                np.ascontiguousarray(p["amp_prec"]),
                float(p["rr"]),
                float(const),
                np.ascontiguousarray(p["log_lam_b"], dtype=float),
                int(self.k_min if self.k_min is not None else self.baseline_order),
            )
        return p["fast"]

    def _moments_compiled(self, theta):
        basis = np.ascontiguousarray(self.basis(theta))
        return _fast_ll.woodbury(basis, *self._fast_args())

    def moments(self, theta) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gaussian-marginal ln L_K, E[a | theta, y, K] and Cov[a | theta, y, K] for every K.

        Shapes (b, nK), (b, nK, q), (b, nK, q, q).
        """
        th = np.atleast_2d(np.asarray(theta, dtype=float))
        if self._compiled:
            fn = self._moments_compiled
        elif self.backend == "jax":
            fn = self._jit_moments
        else:
            fn = self._moments_generic
        if th.shape[0] <= COMPILED_CHUNK:
            out = fn(th)
            return tuple(np.asarray(o) for o in out)  # type: ignore[return-value]
        # chunks keep the temporary basis arrays in cache
        parts = [fn(th[i : i + COMPILED_CHUNK]) for i in range(0, th.shape[0], COMPILED_CHUNK)]
        return tuple(np.concatenate([np.asarray(pt[k]) for pt in parts]) for k in range(3))  # type: ignore[return-value]

    def _log_weights(self, theta) -> np.ndarray:
        """ln [P(K) p(y | theta, K) (orthant factor)] per K, shape (b, nK)."""
        ll, mean, cov = self.moments(theta)
        lw = ll + self.log_prior_k
        if self.positivity and self.n_amp:
            b, nk, q = mean.shape
            orth = log_orthant(mean.reshape(b * nk, q), cov.reshape(b * nk, q, q)).reshape(b, nk)
            lw = np.where(ll > NEG_INF / 2, lw + orth + q * LOG2, lw)
        return lw

    @staticmethod
    def log_orthant_amp(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """ln P(a >= 0) of the amplitude block as used in the sampling target."""
        return log_orthant(mean, cov)

    def loglike(self, theta) -> np.ndarray:
        """ln p(y | theta) marginalised over the baseline order, vectorised over theta rows.

        With ``positivity`` the half-normal amplitude priors enter through the
        orthant probability of each K (exact for up to two amplitudes).
        """
        lw = self._log_weights(theta)
        out = logsumexp(lw, axis=1)
        return np.where(np.isfinite(out), out, NEG_INF)

    def loglike_gauss(self, theta) -> np.ndarray:
        """ln p(y | theta) with untruncated Gaussian amplitude priors (K marginalised)."""
        ll = self.moments(theta)[0]
        return logsumexp(ll + self.log_prior_k, axis=1)

    def loglike_k(self, theta) -> np.ndarray:
        """ln p(y | theta, K) for every K (b, nK), incl. the orthant factor, without P(K)."""
        return self._log_weights(theta) - self.log_prior_k

    def k_posterior(self, theta) -> np.ndarray:
        """P(K | theta, y), shape (b, nK)."""
        lw = self._log_weights(theta)
        return np.exp(lw - logsumexp(lw, axis=1, keepdims=True))

    def loglike_null_k(self) -> np.ndarray:
        """ln p(y | K) of the baseline-only model for every K (no components, no theta)."""
        if self.n_amp:
            raise ValueError("loglike_null_k is only defined without components")
        if self.n_base == 0:
            n = self._pre["n"]
            return np.array([-0.5 * (self._pre["rr"] + self._pre["logdet_c"] + n * LOG_2PI)])
        theta = np.zeros((1, max(self.n_theta, 1)))
        return np.asarray(self._moments_generic(theta)[0])[0]

    def loglike_null(self) -> float:
        """ln p(y) of the baseline-only model, marginalised over K."""
        return float(logsumexp(self.loglike_null_k() + self.log_prior_k))

    def conditional(self, theta, k: np.ndarray | int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Mean (b, p) and covariance (b, p, p) of beta | theta, y, K, zero-padded beyond K.

        ``k`` is the order per sample (or one for all); ``None`` takes the most
        probable order of each sample.
        """
        xp = get_xp(self.backend)
        th = np.atleast_2d(np.asarray(theta, dtype=float))
        b = th.shape[0]
        p = self.n_linear
        if p == 0:
            return np.zeros((b, 0)), np.zeros((b, 0, 0))
        if k is None:
            kk = np.argmax(self.k_posterior(th), axis=1) if self.n_k > 1 else np.zeros(b, int)
            ks = np.asarray(self.k_values)[kk]
        else:
            ks = np.broadcast_to(np.asarray(k, dtype=int), (b,))
        if self._compiled:
            basis = np.ascontiguousarray(self.basis(th))
            a, bvec = _fast_ll.gram(basis, *self._fast_args()[:7])
        else:
            a, bvec, _, _ = self._terms(xp.asarray(th))
            a = np.asarray(a)
            bvec = np.asarray(bvec)
        mean = np.zeros((b, p))
        cov = np.zeros((b, p, p))
        mu = np.r_[np.zeros(self.n_amp), self.baseline_mean]
        for kv in np.unique(ks):
            rows = np.flatnonzero(ks == kv)
            m = self.n_amp + int(kv) + 1
            inv = np.linalg.inv(a[rows][:, :m, :m])
            inv = 0.5 * (inv + np.swapaxes(inv, 1, 2))
            sol = np.einsum("bij,bj->bi", inv, bvec[rows][:, :m])
            mean[np.ix_(rows, np.arange(m))] = sol + mu[:m]
            cov[np.ix_(rows, np.arange(m), np.arange(m))] = inv
        return mean, cov

    def __call__(self, theta):
        return self.loglike(theta)


def default_amp_scale(
    flux: np.ndarray, good: np.ndarray, sigma_rms: float, dv: float, factor: float = 10.0
) -> float:
    """Scale of the half-normal prior on the integrated flux S: ``factor`` times
    the noise of the flux integrated over the whole band.

    It depends on the noise and the band only, not on the signal, so that
    evidences and detection probabilities of spectra with the same noise are
    comparable. A source brighter than a few times the scale is still found;
    only its evidence pays a penalty, which matters nowhere near the detection
    threshold. Fix ``priors.amplitude_scale`` for a survey.
    """
    n = int(np.sum(good))
    return factor * sigma_rms * dv * math.sqrt(n)


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
    positivity: bool = True,
    k_min: int | None = None,
) -> MarginalLikelihood:
    """Likelihood for the given components with K = k_min..baseline_order marginalised.

    An informative baseline prior (``baseline_mean`` or ``baseline_cov``) fixes
    the order to ``baseline_order``.
    """
    nb = baseline_order + 1
    mean = np.zeros(nb) if baseline_mean is None else np.asarray(baseline_mean, float)[:nb]
    cov = (
        np.eye(nb) * baseline_scale**2 if baseline_cov is None else np.asarray(baseline_cov, float)[:nb, :nb]
    )
    if baseline_mean is not None or baseline_cov is not None:
        k_min = baseline_order
    return MarginalLikelihood(
        flux=flux,
        grid=grid,
        noise=noise,
        components=tuple(components),
        baseline_order=baseline_order,
        amp_scale=np.asarray(amp_scale, dtype=float),
        baseline_mean=mean,
        baseline_cov=cov,
        k_min=k_min,
        noise_scale=noise_scale,
        backend=backend,
        positivity=positivity,
    )
