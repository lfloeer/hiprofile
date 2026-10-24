"""Fit one spectrum: models N = 0..N_max with the baseline order marginalised, evidences, posteriors.

Pipeline per spectrum

1. normalise the spectrum, build the channel grid and noise model
2. optional RFI pre-stage (outlier mixture) -> mask or inflate channels
3. N = 0: evidence analytic for every baseline order K
4. N >= 1: one sampler run on the likelihood that already marginalises over
   K = K_min..K_max (a quick MAP + Laplace fit first provides the evidence
   gate); the per-K evidences follow from the posterior mean of P(K | theta, y)
5. amplitude positivity: exact in the likelihood for up to two components,
   a Monte Carlo correction beyond (see ``amplitudes``)
6. model probabilities with a Poisson prior on N and a flat prior on K,
   detection probability P(N >= 1 | y)
7. derived quantities, diagnostics and flags
"""

from __future__ import annotations

import logging
import math
import time
import traceback
from typing import Any

import numpy as np
from scipy.special import logsumexp

from hiprofile.config import Config
from hiprofile.inference.amplitudes import draw_amplitudes
from hiprofile.inference.derived import DERIVED, profile_derived
from hiprofile.inference.evaluate import evaluate, model_probabilities
from hiprofile.inference.priors import ModelPrior
from hiprofile.inference.results import (
    FitResult,
    ModelEvidence,
    Posterior,
    spectrum_seed,
)
from hiprofile.inference.setup import FitContext, apply_rfi, estimate_noise, prepare
from hiprofile.likelihood.marginal import NAMP, MarginalLikelihood
from hiprofile.model.kernel import PROFILE_NPARAM
from hiprofile.samplers import run as run_sampler
from hiprofile.samplers.base import SamplerResult, systematic_resample
from hiprofile.spectrum import Spectrum

log = logging.getLogger(__name__)


def _sampler_options(cfg: Config, backend: str) -> dict[str, Any]:
    opts = (
        dict(cfg.sampler.options.get(backend, {}))
        if isinstance(cfg.sampler.options.get(backend), dict)
        else {}
    )
    return opts


def _run(
    ctx: FitContext,
    like: MarginalLikelihood,
    prior: ModelPrior,
    backend: str,
    salt: int,
    **extra: Any,
) -> SamplerResult:
    cfg = ctx.config.sampler
    return run_sampler(
        backend,
        like.loglike,
        prior.transform,
        prior.ndim,
        seed=(ctx.seed + 7919 * salt) % (2**31),
        n_live=cfg.n_live,
        **{**_sampler_options(ctx.config, backend), **extra},
    )


def _truncation(
    like: MarginalLikelihood, theta: np.ndarray, weights: np.ndarray, rng: np.random.Generator, n: int = 1000
) -> tuple[float, float]:
    """(ln Z correction, negative-amplitude fraction) from a resampled subset."""
    if like.n_amp == 0:
        return 0.0, 0.0
    idx = systematic_resample(weights, min(n, max(weights.size, 1)), rng)
    draws = draw_amplitudes(like, theta[idx], rng)
    w = np.full(idx.size, 1.0 / idx.size)
    return draws.log_z_correction(w), draws.negative_fraction(w)


def _compact(
    res: SamplerResult, max_keep: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop negligible-weight samples, cap the number by systematic resampling."""
    w = res.weights
    order = np.argsort(w)
    cum = np.cumsum(w[order])
    keep = np.sort(order[cum > 1e-5])
    theta, lw, ll = res.samples[keep], np.log(w[keep]), res.log_l[keep]
    if keep.size > max_keep:
        idx = systematic_resample(np.exp(lw - logsumexp(lw)), max_keep, rng)
        idx, counts = np.unique(idx, return_counts=True)
        theta, ll = theta[idx], ll[idx]
        lw = np.log(counts.astype(float))
    return theta, lw - logsumexp(lw), ll


def residual_peaks(
    resid: np.ndarray, good: np.ndarray, n_peaks: int, threshold: float = 3.0
) -> list[tuple[int, float]]:
    """(channel, FWHM in channels) of the strongest features of a whitened residual.

    Gaussian kernels of 2 and 6 channels act as matched filters for narrow and
    broad lines; each smoothed residual is scaled to unit noise and its local
    maxima above `threshold` are kept, strongest first, one per neighbourhood.
    """
    r = np.where(good, resid, 0.0)
    found: list[tuple[float, int, float]] = []
    for sig in (2.0, 6.0):
        x = np.arange(-int(4 * sig), int(4 * sig) + 1)
        k = np.exp(-0.5 * (x / sig) ** 2)
        k /= k.sum()
        s = np.convolve(r, k, mode="same") / np.sqrt(np.sum(k**2))
        s[~good] = 0.0
        for i in range(1, s.size - 1):
            if s[i] >= threshold and s[i] >= s[i - 1] and s[i] > s[i + 1]:
                lo = hi = i
                while lo > 0 and s[lo - 1] > 0.5 * s[i]:
                    lo -= 1
                while hi < s.size - 1 and s[hi + 1] > 0.5 * s[i]:
                    hi += 1
                # remove the kernel broadening in quadrature
                width = float(np.sqrt(max((hi - lo + 1) ** 2 - (2.355 * sig) ** 2, 1.0)))
                found.append((float(s[i]), i, width))
    found.sort(reverse=True)
    out: list[tuple[int, float]] = []
    for _, i, w in found:
        if all(abs(i - j) > max(5.0, 0.5 * (w + wj)) for j, wj in out):
            out.append((i, w))
        if len(out) >= n_peaks:
            break
    return out


def prefit_seed(ctx: FitContext, prior: ModelPrior) -> np.ndarray | None:
    """Unit-cube start for N = 1 from the robust pre-fit, which optimised the same model."""
    if ctx.rfi is None or prior.ndim != ctx.rfi.theta.size:
        return None
    u = prior.inverse(ctx.rfi.theta[None])
    return np.clip(u, 1e-4, 1 - 1e-4) if np.all(np.isfinite(u)) else None


def seed_components(ctx: FitContext, n: int, out: FitResult, n_seeds: int) -> np.ndarray | None:
    """Unit-cube starts for N = n: the N-1 posterior medians plus one new profile on each of the
    strongest residual features (a second, fainter line that random starts rarely hit)."""
    prev = out.posteriors.get(n - 1)
    if prev is None or n_seeds <= 0:
        return None
    like = ctx.likelihood(n - 1)
    s = prev.summary()
    theta = np.array([s[name] for name in prev.names])
    mean, _ = like.conditional(theta[None])
    model = like.model(theta[None], mean)[0]
    gi = ctx.noise.good_index
    resid = np.zeros(ctx.spectrum.n_chan)
    resid[gi] = (ctx.spectrum.flux[gi] - model[gi]) / ctx.noise.sigma
    good = np.zeros(ctx.spectrum.n_chan, bool)
    good[gi] = True
    peaks = residual_peaks(resid, good, n_seeds)
    if not peaks:
        return None
    prior = ctx.prior(n)
    n_par = PROFILE_NPARAM + 1
    blocks = theta[: (n - 1) * n_par].reshape(n - 1, n_par)
    tail = theta[(n - 1) * n_par :]
    sig = float(prior.delta_v_rand.transform(np.array([0.5]))[0])
    seeds = []
    for chan, width in peaks:
        w50 = width * ctx.grid.dv
        new = np.array([ctx.spectrum.velocity[chan], min(max(w50 - 2.355 * sig, 5.0), 790.0), sig, 0.0, 0.3])
        allb = np.vstack([blocks, new])
        th = np.r_[allb[np.argsort(allb[:, 0])].ravel(), tail]
        u = prior.inverse(th[None])[0]
        if np.all(np.isfinite(u)):
            seeds.append(np.clip(u, 1e-4, 1 - 1e-4))
    return np.array(seeds) if seeds else None


def fit_n(ctx: FitContext, n: int, out: FitResult) -> None:
    """Evidences for all K at fixed N (one sampler run) and the K-marginalised posterior."""
    cfg = ctx.config
    like = ctx.likelihood(n)
    ks = like.k_values
    if like.n_amp == 0:
        if like.n_theta:  # noise scale only
            raise NotImplementedError("noise_scale requires at least one component")
        for k, lz in zip(ks, like.loglike_null_k(), strict=True):
            out.evidences.append(
                ModelEvidence(
                    n=n, k=k, log_z=float(lz), log_z_err=0.0, log_z_gauss=float(lz), method="analytic"
                )
            )
        return

    prior = ctx.prior(n)
    mode = cfg.sampler.mode
    t0 = time.perf_counter()
    seeds = seed_components(ctx, n, out, cfg.sampler.n_seeds) if n >= 2 else prefit_seed(ctx, prior)
    # with data-driven starts the random multistart only guards against missed modes
    quick_opts: dict[str, Any] = {} if seeds is None else {"init_u": seeds, "n_starts": 2, "n_init": 128}
    quick = _run(ctx, like, prior, "quick", salt=100 * n, **quick_opts)
    gated = False
    if mode != "quick" and cfg.sampler.gate_margin is not None:
        prev = [e.log_z for e in out.evidences if e.n < n]
        gated = bool(prev) and quick.log_z < max(prev) - cfg.sampler.gate_margin
    if mode == "quick" or gated:
        res = quick
        method = "quick-gated" if gated else "quick"
    else:
        extra: dict[str, Any] = {}
        if cfg.sampler.backend == "smc" and cfg.sampler.init_fraction > 0 and quick.proposal:
            extra["init_proposal"] = (quick.proposal, cfg.sampler.init_fraction)
        res = _run(ctx, like, prior, cfg.sampler.backend, salt=10 * n, **extra)
        method = "marginal"
    check: tuple[float, float] | None = None
    if mode == "check" and not gated:
        r2 = _run(ctx, like, prior, cfg.sampler.check_backend, salt=10 * n + 5)
        check = (r2.log_z, r2.log_z_err)

    corr, negf = _truncation(like, res.samples, res.weights, ctx.rng)
    post = make_posterior(ctx, n, like, prior, res)
    out.posteriors[n] = post
    assert post.k_probability is not None
    for k, pk in zip(ks, post.k_probability, strict=True):
        # Z_{N,K} = Z_N E_post[P(K | theta, y)] / P(K)
        shift = math.log(max(float(pk), 1e-300)) - like.log_prior_k
        ev = ModelEvidence(
            n=n,
            k=k,
            log_z=res.log_z + corr + shift,
            log_z_err=res.log_z_err,
            log_z_gauss=res.log_z + shift,
            method=method,
            backend=res.backend,
            negative_fraction=negf,
            runtime=res.runtime,
            n_like=res.n_like,
            budget_exhausted=bool(res.stats.get("budget_exhausted", False)),
        )
        if check is not None:
            ev.log_z_check = check[0] + corr + shift
            ev.log_z_check_err = check[1]
        out.evidences.append(ev)
    log.debug("N=%d done in %.2fs", n, time.perf_counter() - t0)


def _thin(a: np.ndarray, n: int = 1000) -> np.ndarray:
    if a.ndim != 1 or a.size <= n:
        return a
    return a[np.linspace(0, a.size - 1, n).astype(int)]


def _component_derived(
    ctx: FitContext, like: MarginalLikelihood, theta: np.ndarray, beta: np.ndarray, ok: np.ndarray
) -> dict[str, np.ndarray]:
    derived: dict[str, np.ndarray] = {}
    col = 0
    ip = 0
    for kind, sl in zip(like.components, like.theta_slices, strict=True):
        na = NAMP[kind]
        if kind == "profile":
            ip += 1
            s = np.nan_to_num(beta[:, col], nan=0.0)
            f = theta[:, sl.start + PROFILE_NPARAM]
            amps = np.stack([s * (1.0 - f), s * f], axis=1)
            d = profile_derived(ctx.grid, theta[:, sl.start : sl.start + PROFILE_NPARAM], amps)
            for key in DERIVED:
                if key == "f":
                    continue  # f is a sampled parameter
                derived[f"{key}_{ip}"] = np.where(ok, d[key], np.nan)
        else:
            derived[f"rfi_flux_{col}"] = beta[:, col]
        col += na
    if ip > 1:
        derived["S_total"] = np.sum([derived[f"S_{i}"] for i in range(1, ip + 1)], axis=0)
    return derived


def make_posterior(
    ctx: FitContext, n: int, like: MarginalLikelihood, prior: ModelPrior, res: SamplerResult
) -> Posterior:
    """Posterior samples with amplitudes and a baseline order drawn per sample."""
    theta, lw, ll = _compact(res, ctx.config.output.n_posterior, ctx.rng)
    draws = draw_amplitudes(like, theta, ctx.rng)
    beta = draws.beta
    ok = np.all(np.isfinite(beta), axis=1)
    w = np.exp(lw) * draws.factor * ok
    w = w / w.sum() if w.sum() > 0 else np.exp(lw) / np.exp(lw).sum()
    pk = w @ like.k_posterior(theta)
    pk = pk / pk.sum()
    return Posterior(
        n=n,
        k=int(like.k_values[int(np.argmax(pk))]),
        names=prior.names,
        theta=theta,
        u=prior.inverse(theta),
        weights=w,
        beta=np.nan_to_num(beta, nan=0.0),
        log_l=ll,
        derived=_component_derived(ctx, like, theta, beta, ok),
        trace={key: _thin(np.asarray(val)) for key, val in res.trace.items()},
        k_samples=draws.k,
        k_values=list(like.k_values),
        k_probability=pk,
    )


def fit_spectrum(spectrum: Spectrum, config: Config | None = None, seed: int | None = None) -> FitResult:
    """Fit one spectrum with every model and return evidences, posteriors and flags.

    Errors are caught and returned as ``status="failed"`` so that a survey run
    never stops on a single spectrum.
    """
    cfg = config or Config()
    s = spectrum_seed(cfg.seed, spectrum.id) if seed is None else seed
    out = FitResult(spectrum_id=spectrum.id, seed=s, config=cfg.dump(), meta=dict(spectrum.meta))
    t0 = time.perf_counter()
    try:
        ctx = prepare(spectrum, cfg, s)
        out.velocity = ctx.spectrum.velocity
        out.flux = ctx.spectrum.flux
        if cfg.rfi.prefit and cfg.model.n_max >= 1:
            apply_rfi(ctx)
        if cfg.noise.correlation == "estimate":
            estimate_noise(ctx)
        out.mask = ~ctx.good
        sig = np.full(ctx.spectrum.n_chan, np.nan)
        sig[ctx.noise.good_index] = ctx.noise.sigma
        out.sigma = sig
        out.rfi_q = ctx.rfi.q if ctx.rfi is not None else np.zeros(ctx.spectrum.n_chan)
        for n in range(0, cfg.model.n_max + 1):
            fit_n(ctx, n, out)
        evaluate(ctx, out)
    except Exception as exc:
        out.status = "failed"
        out.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        out.flags = {"failed": True}
        log.error("%s failed: %s", spectrum.id, exc)
    out.runtime = time.perf_counter() - t0
    return out


def prior_draw_u(prior: ModelPrior, rng: np.random.Generator, n: int) -> np.ndarray:
    return prior.transform(rng.uniform(size=(n, prior.ndim)))


__all__ = [
    "FitContext",
    "FitResult",
    "ModelEvidence",
    "Posterior",
    "apply_rfi",
    "fit_spectrum",
    "model_probabilities",
    "prepare",
    "spectrum_seed",
]
