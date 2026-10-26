"""RFI pre-stage: robust fit with an outlier mixture (Hogg, Bovy & Lang 2010).

Per good channel the likelihood is

    L_j = (1 - P_b) N(y_j | m_j, sigma_j) + P_b B(y_j - m_j)

with a broad background B, either symmetric N(0, sigma_j^2 + s_out^2) or only
positive (twice that density for r > 0, zero otherwise; RFI adds power). The fit
alternates (EM): given the outlier probabilities q_j, the profile + baseline
model is optimised with sigma_j inflated by 1 / sqrt(1 - q_j); given the model,
q_j, P_b and s_out are updated. The main fit then masks channels with
q_j > threshold or keeps them with inflated sigma.

Risk: a narrow, weak line or a single horn can be explained as outliers. The
validation (``hiprofile validate --stage detection``) measures completeness
with and without this stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from hiprofile.inference.priors import ModelPrior
from hiprofile.likelihood.marginal import MarginalLikelihood, build_likelihood
from hiprofile.likelihood.noise import NoiseModel
from hiprofile.model.kernel import SpectralGrid
from hiprofile.samplers.base import CountingLikelihood
from hiprofile.samplers.quick import _Target, find_modes

LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


@dataclass
class RFIResult:
    q: np.ndarray  # (n_chan,) outlier probability, 0 for masked channels
    p_bad: float
    s_out: float
    n_iter: int
    theta: np.ndarray
    model: np.ndarray  # (n_chan,) robust model

    @property
    def fraction(self) -> float:
        return float(np.mean(self.q > 0.5))


def _normal_logpdf(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    return -0.5 * (r / s) ** 2 - np.log(s) - LOG_SQRT_2PI


def outlier_mask(
    q: np.ndarray,
    threshold: float,
    close_holes: bool = True,
    max_width: int | None = 3,
    resid: np.ndarray | None = None,
) -> np.ndarray:
    """Channels to treat as RFI, from the outlier probabilities of the pre-stage.

    A single channel between two outliers is masked with them. A run of more
    than `max_width` outlier channels (gaps of up to two channels count as part
    of the run) is not RFI but signal the one-component model of the pre-stage
    cannot explain, typically a second line, and stays in the fit. With the
    residuals of the pre-stage given, runs whose median residual is negative
    are masked whatever their length: an emission line cannot produce them.
    """
    bad = q > threshold
    if bad.size < 3:
        return bad
    if close_holes:
        hole = np.zeros_like(bad)
        hole[1:-1] = bad[:-2] & bad[2:]
        bad |= hole
    if max_width is not None:
        idx = np.flatnonzero(bad)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 3) + 1):
            if run.size and run[-1] - run[0] + 1 > max_width:
                if resid is not None and np.median(resid[run]) < 0:
                    continue
                bad[run[0] : run[-1] + 1] = False
    return bad


def robust_prefit(
    flux: np.ndarray,
    grid: SpectralGrid,
    noise: NoiseModel,
    prior: ModelPrior,
    baseline_order: int,
    amp_scale: float | np.ndarray,
    baseline_scale: float,
    rng: np.random.Generator,
    background: str = "positive",
    n_iter: int = 8,
    backend: str = "numpy",
    k_min: int | None = None,
) -> RFIResult:
    idx = noise.good_index
    y = np.asarray(flux, float)[idx]
    s = noise.sigma
    q = np.zeros(idx.size)
    p_bad = 0.01
    s_out = 10.0 * float(np.median(s))
    z_best: np.ndarray | None = None
    like: MarginalLikelihood | None = None
    theta = np.zeros(prior.ndim)
    it = 0
    for it in range(1, n_iter + 1):  # noqa: B007 - `it` is reported
        factor = 1.0 / np.sqrt(np.clip(1.0 - q, 1e-4, 1.0))
        like = build_likelihood(
            flux,
            grid,
            noise.scaled(factor),
            prior.components,
            baseline_order,
            amp_scale,
            baseline_scale,
            backend=backend,
            k_min=k_min,
        )
        target = _Target(CountingLikelihood(like.loglike), prior.transform)
        if z_best is None:
            z_best = find_modes(target, prior.ndim, rng, n_init=512, n_starts=2)[0][0]
        else:
            # the outlier weights change little between iterations: refine the previous optimum
            z_best = find_modes_from(target, z_best)[0]
        theta = prior.transform(expit(z_best)[None])[0]
        mean, _ = like.conditional(theta[None])
        model = like.model(theta[None], mean)[0]
        r = y - model[idx]
        log_in = np.log1p(-p_bad) + _normal_logpdf(r, s)
        s_b = np.sqrt(s**2 + s_out**2)
        log_out = np.log(p_bad) + _normal_logpdf(r, s_b)
        if background == "positive":
            log_out = np.where(r > 0, log_out + np.log(2.0), -np.inf)
        q_new = expit(log_out - log_in)
        p_bad = float(np.clip(q_new.mean(), 1e-4, 0.5))
        if q_new.sum() > 1:
            s_out = float(
                np.sqrt(max(np.sum(q_new * r**2) / q_new.sum() - np.mean(s**2), (3 * np.median(s)) ** 2))
            )
        converged = np.max(np.abs(q_new - q)) < 1e-2
        q = q_new
        if converged:
            break
    assert like is not None
    q_full = np.zeros(grid.n_chan)
    q_full[idx] = q
    mean, _ = like.conditional(theta[None])
    return RFIResult(
        q=q_full,
        p_bad=p_bad,
        s_out=s_out,
        n_iter=it,
        theta=theta,
        model=like.model(theta[None], mean)[0],
    )


def find_modes_from(target: _Target, z0: np.ndarray) -> tuple[np.ndarray, float]:
    from scipy import optimize

    from hiprofile.samplers.quick import _grad

    def fun(z: np.ndarray) -> tuple[float, np.ndarray]:
        f, g = _grad(target, z)
        return -f, -g

    res = optimize.minimize(fun, z0, jac=True, method="L-BFGS-B", options={"maxiter": 100})
    return res.x, float(target(res.x)[0])
