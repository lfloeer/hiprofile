"""Fit one spectrum: all models N = 0..N_max, K = K_min..K_max, evidences, posteriors.

Pipeline per spectrum

1. normalise the spectrum, build the channel grid and noise model
2. optional RFI pre-stage (outlier mixture) -> mask or inflate channels
3. N = 0: evidence analytic (unless RFI components are part of every model)
4. N >= 1: choose the preferred baseline order K* from quick Laplace fits, run
   the sampler at K*, importance-reweight the samples to the other K (re-run
   when the reweighted effective sample size is too small)
5. positivity of the amplitudes by reweighting (see ``amplitudes``)
6. model probabilities with a Poisson prior on N and a flat prior on K,
   detection probability P(N >= 1 | y)
7. derived quantities, diagnostics and flags
"""

from __future__ import annotations

import logging
import math
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import expit, logsumexp

from hiprofile import __version__
from hiprofile.config import Config
from hiprofile.diagnostics import metrics as dm
from hiprofile.diagnostics.flags import compute_flags
from hiprofile.inference.amplitudes import draw_amplitudes
from hiprofile.inference.derived import DERIVED, classic_widths, profile_derived
from hiprofile.inference.priors import ModelPrior, build_prior
from hiprofile.likelihood.marginal import (
    NAMP,
    MarginalLikelihood,
    build_likelihood,
    default_amp_scale,
)
from hiprofile.likelihood.noise import NoiseModel, build_noise
from hiprofile.likelihood.rfi import RFIResult, robust_prefit
from hiprofile.model.kernel import SpectralGrid
from hiprofile.samplers import run as run_sampler
from hiprofile.samplers.base import SamplerResult, systematic_resample
from hiprofile.spectrum import Spectrum

log = logging.getLogger(__name__)


@dataclass
class ModelEvidence:
    n: int
    k: int
    log_z: float  # with positivity truncation
    log_z_err: float
    log_z_gauss: float
    method: str  # analytic | sampled | reweighted | quick
    backend: str = ""
    ess_fraction: float = 1.0
    negative_fraction: float = 0.0
    runtime: float = 0.0
    n_like: int = 0
    budget_exhausted: bool = False
    log_z_check: float = float("nan")
    log_z_check_err: float = float("nan")


@dataclass
class Posterior:
    n: int
    k: int
    names: list[str]
    theta: np.ndarray  # (m, d)
    u: np.ndarray  # (m, d) unit-cube coordinates
    weights: np.ndarray  # (m,) normalised
    beta: np.ndarray  # (m, p) linear parameters (amplitudes, baseline)
    log_l: np.ndarray
    derived: dict[str, np.ndarray]  # "<name>_<i>" -> (m,)
    trace: dict[str, np.ndarray] = field(default_factory=dict)
    k_samples: np.ndarray | None = None  # baseline order of each sample (K-marginalised posterior)

    @property
    def k_max(self) -> int:
        return int(self.k_samples.max()) if self.k_samples is not None and self.k_samples.size else self.k

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        cols = {**{nm: self.theta[:, i] for i, nm in enumerate(self.names)}, **self.derived}
        for key, val in cols.items():
            ok = np.isfinite(val)
            if not ok.any():
                continue
            q = weighted_quantile(val[ok], self.weights[ok], [0.16, 0.5, 0.84])
            out[f"{key}_p16"], out[key], out[f"{key}_p84"] = q
        return out


@dataclass
class FitResult:
    spectrum_id: str
    status: str = "ok"
    error: str = ""
    version: str = __version__
    seed: int = 0
    config: str = ""
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    flux: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sigma: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mask: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    rfi_q: np.ndarray = field(default_factory=lambda: np.zeros(0))
    evidences: list[ModelEvidence] = field(default_factory=list)
    p_n: dict[int, float] = field(default_factory=dict)
    p_detect: float = float("nan")
    best_n: int = 0
    best_k: int = 0
    posteriors: dict[int, Posterior] = field(default_factory=dict)
    model_band: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # (5, n_chan)
    components_median: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    baseline_median: np.ndarray = field(default_factory=lambda: np.zeros(0))
    metrics: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)
    classic: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    runtime: float = 0.0

    @property
    def flagged(self) -> bool:
        return any(self.flags.values())

    def catalog_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": self.spectrum_id,
            "status": self.status,
            "best_n": self.best_n,
            "best_k": self.best_k,
            "p_detect": self.p_detect,
            "runtime": self.runtime,
        }
        for n, p in self.p_n.items():
            row[f"p_n{n}"] = p
        for ev in self.evidences:
            row[f"logz_n{ev.n}_k{ev.k}"] = ev.log_z
        post = self.posteriors.get(max(self.best_n, 1)) if self.posteriors else None
        if post is not None:
            for key, val in post.summary().items():
                row[key] = val
        for key, val in self.metrics.items():
            if np.isscalar(val):
                row[f"m_{key}"] = val
        for key, val in self.flags.items():
            row[f"flag_{key}"] = bool(val)
        row["flagged"] = self.flagged
        for key, val in self.classic.items():
            row[f"classic_{key}"] = val
        for key, val in self.meta.items():
            if np.isscalar(val):
                row[f"meta_{key}"] = val
        return row


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: list[float]) -> list[float]:
    order = np.argsort(x)
    x, w = x[order], w[order]
    c = np.cumsum(w)
    c = (c - 0.5 * w) / c[-1]
    return [float(np.interp(qq, c, x)) for qq in q]


def spectrum_seed(global_seed: int, spectrum_id: str) -> int:
    """Seed derived from the global seed and the spectrum id (independent of order)."""
    import hashlib

    h = hashlib.blake2b(f"{global_seed}:{spectrum_id}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") % (2**63)


# --------------------------------------------------------------------------- setup
@dataclass
class FitContext:
    spectrum: Spectrum
    config: Config
    grid: SpectralGrid
    noise: NoiseModel
    good: np.ndarray
    amp_scale: float
    baseline_scale: float
    rng: np.random.Generator
    seed: int
    rfi: RFIResult | None = None

    @property
    def backend(self) -> str:
        return self.config.sampler.likelihood_backend

    def components(self, n: int) -> tuple[str, ...]:
        return ("profile",) * n + ("gauss",) * self.config.model.n_rfi

    def prior(self, n: int) -> ModelPrior:
        return build_prior(
            self.components(n),
            self.config.priors,
            self.spectrum.velocity,
            self.config.model.parametrization,
        )

    def likelihood(self, n: int, k: int, noise: NoiseModel | None = None) -> MarginalLikelihood:
        pr = self.config.priors
        return build_likelihood(
            self.spectrum.flux,
            self.grid,
            noise or self.noise,
            self.components(n),
            k,
            self.amp_scale,
            self.baseline_scale,
            baseline_mean=None if pr.baseline_mean is None else np.asarray(pr.baseline_mean),
            baseline_cov=None if pr.baseline_cov is None else np.asarray(pr.baseline_cov),
            noise_scale=pr.noise_scale is not None,
            backend=self.backend,
        )


def _correlation(cfg: Config) -> str | list[float]:
    corr = cfg.noise.correlation
    if corr == "auto":
        return "hanning" if "hanning" in cfg.model.response.lower() else "white"
    return corr


def prepare(spectrum: Spectrum, config: Config, seed: int) -> FitContext:
    spec = spectrum.normalized()
    cfg = config
    good = spec.good
    pr = cfg.priors
    sig_lo = pr.delta_v_rand.low if pr.delta_v_rand.low is not None else 4.0
    sig_min = sig_lo
    if cfg.model.n_rfi:
        sig_min = min(sig_min, pr.rfi_sigma_channels[0] * spec.channel_width)
    dv_hi = pr.delta_v.high if pr.delta_v.high is not None else 600.0
    sig_hi = pr.delta_v_rand.high if pr.delta_v_rand.high is not None else 60.0
    grid = SpectralGrid.from_velocities(
        spec.velocity,
        supersample=cfg.model.supersample,
        sigma_min=sig_min,
        max_width=dv_hi + 10.0 * sig_hi,
        response=cfg.model.response,
    )
    use_sigma = spec.sigma if cfg.noise.sigma in ("auto", "input") else None
    if cfg.noise.sigma == "input" and spec.sigma is None:
        raise ValueError(f"{spec.id}: noise.sigma = input but the spectrum has no sigma column")
    signal_free = None
    if cfg.noise.signal_free:
        signal_free = np.zeros(spec.n_chan, bool)
        for lo, hi in cfg.noise.signal_free:
            signal_free |= (spec.velocity >= lo) & (spec.velocity <= hi)
    noise = build_noise(spec.flux, good, use_sigma, _correlation(cfg), signal_free)
    rms = float(np.median(noise.sigma))
    amp = (
        default_amp_scale(spec.flux, good, rms, grid.dv)
        if pr.amplitude_scale == "auto"
        else float(pr.amplitude_scale)
    )
    base = 10.0 * rms if pr.baseline_scale == "auto" else float(pr.baseline_scale)
    return FitContext(
        spectrum=spec,
        config=cfg,
        grid=grid,
        noise=noise,
        good=good,
        amp_scale=amp,
        baseline_scale=base,
        rng=np.random.default_rng(seed),
        seed=seed,
    )


def apply_rfi(ctx: FitContext) -> None:
    cfg = ctx.config.rfi
    k_pre = min(max(ctx.config.model.k_min, 1), ctx.config.model.k_max)
    prior = build_prior(("profile",), ctx.config.priors, ctx.spectrum.velocity)
    res = robust_prefit(
        ctx.spectrum.flux,
        ctx.grid,
        ctx.noise,
        prior,
        k_pre,
        ctx.amp_scale,
        ctx.baseline_scale,
        ctx.rng,
        background=cfg.background,
        n_iter=cfg.n_iter,
    )
    ctx.rfi = res
    bad = res.q > cfg.threshold
    if not bad.any():
        return
    if cfg.action == "mask":
        ctx.good = ctx.good & ~bad
        sig_full = np.full(ctx.spectrum.n_chan, np.nan)
        sig_full[ctx.noise.good_index] = ctx.noise.sigma
        ctx.noise = NoiseModel(
            sigma=sig_full[ctx.good], good_index=np.flatnonzero(ctx.good), rho=ctx.noise.rho
        )
    else:
        q = res.q[ctx.noise.good_index]
        ctx.noise = ctx.noise.scaled(1.0 / np.sqrt(np.clip(1.0 - q, 1e-4, 1.0)))


# ------------------------------------------------------------------------- models
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


# The K scan only needs evidences: fewer importance samples, one adaptation round
SCAN_OPTIONS = {"n_samples": 1500, "n_adapt": 1}


def _quick_all_k(
    ctx: FitContext, likes: dict[int, MarginalLikelihood], prior: ModelPrior, n: int
) -> dict[int, SamplerResult]:
    """Quick fits for every K; K > first start from the modes of the previous K."""
    out: dict[int, SamplerResult] = {}
    init = None
    for k in sorted(likes):
        out[k] = _run(ctx, likes[k], prior, "quick", salt=100 * n + k, init_u=init, **SCAN_OPTIONS)
        init = np.asarray(out[k].stats["modes_u"])
    return out


def _truncation(
    like: MarginalLikelihood, theta: np.ndarray, weights: np.ndarray, rng: np.random.Generator, n: int = 1000
) -> tuple[float, float]:
    """(ln Z correction, negative-amplitude fraction) from a resampled subset."""
    if like.n_amp == 0:
        return 0.0, 0.0
    idx = systematic_resample(weights, min(n, max(weights.size, 1)), rng)
    draws = draw_amplitudes(like, theta[idx], rng)
    w = np.full(idx.size, 1.0 / idx.size)
    return draws.log_z_correction(w, like.n_amp), draws.negative_fraction(w)


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


def fit_n(ctx: FitContext, n: int, out: FitResult) -> None:
    """Evidences for all K at fixed N, and the posterior at the best K."""
    cfg = ctx.config
    ks = list(range(cfg.model.k_min, cfg.model.k_max + 1))
    comps = ctx.components(n)
    if len(comps) == 0:
        for k in ks:
            like = ctx.likelihood(n, k)
            if like.n_theta:  # noise scale only
                raise NotImplementedError("noise_scale requires at least one component")
            lz = like.loglike_null()
            out.evidences.append(
                ModelEvidence(n=n, k=k, log_z=lz, log_z_err=0.0, log_z_gauss=lz, method="analytic")
            )
        return

    prior = ctx.prior(n)
    likes = {k: ctx.likelihood(n, k) for k in ks}
    mode = cfg.sampler.mode
    backend = "quick" if mode == "quick" else cfg.sampler.backend
    results: dict[int, SamplerResult] = {}
    t0 = time.perf_counter()

    # 1) quick fits for all K: K* and quick-mode evidences
    if mode == "quick" or len(ks) > 1:
        quick = _quick_all_k(ctx, likes, prior, n)
        k_star = max(ks, key=lambda k: quick[k].log_z)
    else:
        quick = {}
        k_star = ks[0]
    gated = False
    if mode != "quick" and quick and cfg.sampler.gate_margin is not None:
        prev = [e.log_z for e in out.evidences if e.n < n]
        best_quick = max(r.log_z for r in quick.values())
        gated = bool(prev) and best_quick < max(prev) - cfg.sampler.gate_margin
    if mode == "quick" or gated:
        results = dict(quick)
        methods = {k: "quick-gated" if gated else "quick" for k in ks}
        # Full importance-sampling run for the preferred K (posterior samples)
        results[k_star] = _run(
            ctx,
            likes[k_star],
            prior,
            "quick",
            salt=100 * n + k_star + 50,
            init_u=np.asarray(quick[k_star].stats["modes_u"]),
        )
    else:
        results[k_star] = _run(ctx, likes[k_star], prior, backend, salt=10 * n + k_star)
        methods = {k_star: "sampled"}

    # 2) other K: reweight or rerun
    ess_frac = {k: 1.0 for k in ks}
    log_z_g: dict[int, float] = {k: r.log_z for k, r in results.items()}
    log_z_e: dict[int, float] = {k: r.log_z_err for k, r in results.items()}
    if mode != "quick" and not gated:
        ref = results[k_star]
        th, lw, ll = _compact(ref, 20000, ctx.rng)
        base_ess = dm.kish_ess(np.exp(lw))
        for k in ks:
            if k == k_star:
                continue
            if cfg.sampler.baseline_evidence == "rerun":
                results[k] = _run(ctx, likes[k], prior, backend, salt=10 * n + k)
                methods[k] = "sampled"
                log_z_g[k], log_z_e[k] = results[k].log_z, results[k].log_z_err
                continue
            ll_k = np.asarray(likes[k].loglike(th))
            lw_k = lw + ll_k - ll
            lz = ref.log_z + float(logsumexp(lw_k))
            w_k = np.exp(lw_k - logsumexp(lw_k))
            frac = dm.kish_ess(w_k) / max(base_ess, 1.0)
            ess_frac[k] = frac
            best_so_far = max(log_z_g.values())
            if frac < cfg.sampler.reweight_min_ess and lz > best_so_far - 5.0:
                results[k] = _run(ctx, likes[k], prior, backend, salt=10 * n + k)
                methods[k] = "sampled"
                log_z_g[k], log_z_e[k] = results[k].log_z, results[k].log_z_err
                ess_frac[k] = 1.0
            else:
                methods[k] = "reweighted"
                log_z_g[k] = lz
                log_z_e[k] = float(np.hypot(ref.log_z_err, 1.0 / math.sqrt(max(dm.kish_ess(w_k), 1.0))))
                results[k] = SamplerResult(
                    samples=th,
                    log_weights=lw_k,
                    log_l=ll_k,
                    log_z=lz,
                    log_z_err=log_z_e[k],
                    backend=ref.backend + "+reweight",
                )

    # 3) check mode: second backend at K*
    check: tuple[float, float] | None = None
    if mode == "check" and not gated:
        r2 = _run(ctx, likes[k_star], prior, cfg.sampler.check_backend, salt=10 * n + k_star + 5)
        check = (r2.log_z, r2.log_z_err)

    # 4) positivity truncation and bookkeeping
    for k in ks:
        r = results[k]
        corr, negf = _truncation(likes[k], r.samples, r.weights, ctx.rng)
        ev = ModelEvidence(
            n=n,
            k=k,
            log_z=log_z_g[k] + corr,
            log_z_err=log_z_e[k],
            log_z_gauss=log_z_g[k],
            method=methods[k],
            backend=r.backend,
            ess_fraction=ess_frac[k],
            negative_fraction=negf,
            runtime=r.runtime,
            n_like=r.n_like,
            budget_exhausted=bool(r.stats.get("budget_exhausted", False)),
        )
        if check is not None and k == k_star:
            ev.log_z_check = check[0] + corr
            ev.log_z_check_err = check[1]
        out.evidences.append(ev)
    log.debug("N=%d done in %.2fs", n, time.perf_counter() - t0)

    # 5) posterior marginalised over K: mixture of the per-K posteriors with P(K | N, y)
    lz_k = np.array([next(e.log_z for e in out.evidences if e.n == n and e.k == k) for k in ks])
    p_k = np.exp(lz_k - logsumexp(lz_k))
    parts = [(k, likes[k], results[k], float(pk)) for k, pk in zip(ks, p_k, strict=True) if pk > 1e-3]
    out.posteriors[n] = make_posterior(ctx, n, parts, prior)


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
            d = profile_derived(ctx.grid, theta[:, sl], np.nan_to_num(beta[:, col : col + na], nan=0.0))
            for key in DERIVED:
                derived[f"{key}_{ip}"] = np.where(ok, d[key], np.nan)
        else:
            derived[f"rfi_flux_{col}"] = beta[:, col]
        col += na
    if ip > 1:
        derived["S_total"] = np.sum([derived[f"S_{i}"] for i in range(1, ip + 1)], axis=0)
    return derived


def make_posterior(
    ctx: FitContext,
    n: int,
    parts: list[tuple[int, MarginalLikelihood, SamplerResult, float]],
    prior: ModelPrior,
) -> Posterior:
    """Posterior marginalised over the baseline order.

    ``parts`` holds (K, likelihood, sampler result, P(K | N, y)). Each K
    contributes about P(K) * n_posterior samples; the baseline coefficients are
    zero-padded to the largest K so that one design matrix serves all samples.
    """
    n_post = ctx.config.output.n_posterior
    k_max = max(k for k, *_ in parts)
    n_amp = parts[0][1].n_amp
    width = n_amp + k_max + 1
    thetas, lls, ws, betas, kss, ders = [], [], [], [], [], []
    for k, like, res, pk in parts:
        keep = max(50, round(n_post * pk))
        theta, lw, ll = _compact(res, keep, ctx.rng)
        draws = draw_amplitudes(like, theta, ctx.rng)
        beta = draws.beta
        ok = np.all(np.isfinite(beta), axis=1)
        w = np.exp(lw) * draws.accept * ok
        w = w / w.sum() if w.sum() > 0 else np.exp(lw) / np.exp(lw).sum()
        padded = np.zeros((theta.shape[0], width))
        padded[:, : beta.shape[1]] = np.nan_to_num(beta, nan=0.0)
        thetas.append(theta)
        lls.append(ll)
        ws.append(pk * w)
        betas.append(padded)
        kss.append(np.full(theta.shape[0], k))
        ders.append(_component_derived(ctx, like, theta, beta, ok))
    w = np.concatenate(ws)
    k_best = max(parts, key=lambda t: t[3])[0]
    best_res = next(r for k, _, r, _ in parts if k == k_best)
    theta = np.vstack(thetas)
    return Posterior(
        n=n,
        k=k_best,
        names=prior.names,
        theta=theta,
        u=prior.inverse(theta),
        weights=w / w.sum(),
        beta=np.vstack(betas),
        log_l=np.concatenate(lls),
        derived={key: np.concatenate([d[key] for d in ders]) for key in ders[0]},
        trace={key: _thin(np.asarray(val)) for key, val in best_res.trace.items()},
        k_samples=np.concatenate(kss),
    )


# --------------------------------------------------------------------- evaluation
def model_probabilities(evs: list[ModelEvidence], rate: float, n_max: int) -> dict[int, float]:
    ns = sorted({e.n for e in evs})
    log_pn = {n: n * math.log(rate) - rate - math.lgamma(n + 1) for n in ns}
    by_n = {}
    for n in ns:
        lz = np.array([e.log_z for e in evs if e.n == n])
        by_n[n] = float(logsumexp(lz) - math.log(lz.size)) + log_pn[n]
    tot = logsumexp(list(by_n.values()))
    return {n: float(math.exp(v - tot)) for n, v in by_n.items()}


def posterior_models(
    ctx: FitContext, post: Posterior, like: MarginalLikelihood, n_draw: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Posterior predictive draws of full models, components and baselines."""
    idx = systematic_resample(post.weights, n_draw, ctx.rng)
    design = like.design(post.theta[idx])  # (m, p, n_chan)
    beta = post.beta[idx]
    full = np.einsum("mpn,mp->mn", design, beta)
    na = like.n_amp
    base = np.einsum("mpn,mp->mn", design[:, na:], beta[:, na:]) if like.n_base else np.zeros_like(full)
    comps = []
    col = 0
    for kind in like.components:
        k = NAMP[kind]
        comps.append(np.einsum("mpn,mp->mn", design[:, col : col + k], beta[:, col : col + k]))
        col += k
    return full, np.array(comps), base, idx


def evaluate(ctx: FitContext, out: FitResult) -> None:
    cfg = ctx.config
    out.p_n = model_probabilities(out.evidences, cfg.model.poisson_rate, cfg.model.n_max)
    out.p_detect = float(sum(p for n, p in out.p_n.items() if n >= 1))
    out.best_n = max(out.p_n, key=lambda n: out.p_n[n])
    ev_n = [e for e in out.evidences if e.n == out.best_n]
    out.best_k = max(ev_n, key=lambda e: e.log_z).k
    m: dict[str, Any] = {
        "sigma_rms": float(np.median(ctx.noise.sigma)),
        "amp_scale": ctx.amp_scale,
        "baseline_scale": ctx.baseline_scale,
        "n_good": int(ctx.noise.n),
        "supersample": ctx.grid.supersample,
    }
    if ctx.rfi is not None:
        m["rfi_fraction"] = float(np.mean(ctx.rfi.q > cfg.rfi.threshold))
        m["rfi_p_bad"] = ctx.rfi.p_bad
    n_post = max(out.best_n, 1)
    if cfg.model.n_max >= 1 and n_post in out.posteriors:
        post = out.posteriors[n_post]
        like = ctx.likelihood(n_post, post.k_max)
        full, comps, base, _idx = posterior_models(ctx, post, like)
        out.model_band = np.percentile(full, [2.5, 16, 50, 84, 97.5], axis=0)
        out.components_median = np.median(comps, axis=1)
        out.baseline_median = np.median(base, axis=0)
        gi = ctx.noise.good_index
        y = ctx.spectrum.flux[gi]
        med = out.model_band[2][gi]
        rw = ctx.noise.whiten(y - med)
        m["chi2_red"] = dm.chi2_reduced(rw, post.theta.shape[1] + like.n_linear)
        m["ppp"] = dm.posterior_predictive_p(y, full[:, gi], ctx.noise, ctx.rng)
        m["residual_acf"] = dm.residual_acf(rw).tolist()
        m["residual_acf_max"] = float(np.max(np.abs(m["residual_acf"])))
        line = out.components_median[:n_post].sum(axis=0)
        width = float(np.median(post.theta[:, 1])) + 2.35 * float(np.median(post.theta[:, 2]))
        m["wiggle_J"], m["wiggle_snr"] = dm.wiggle_fraction(
            (y - out.baseline_median[gi]), line[gi], ctx.noise.sigma, width, ctx.grid.dv
        )
        m["ess"] = dm.kish_ess(post.weights)
        edge_vals = []
        bounds = ctx.prior(n_post).edge_bounds()
        for i, name in enumerate(post.names):
            if name in bounds:
                lo, hi = bounds[name]
                e = dm.edge_mass(post.theta[:, i], post.weights, lo, hi)
                m[f"edge_{name}"] = e
                edge_vals.append(e)
        for key, val in post.derived.items():
            if key.startswith("f_"):
                ok = np.isfinite(val)
                e = dm.edge_mass(val[ok], post.weights[ok], 0.0, 1.0) if ok.any() else 0.0
                m[f"edge_{key}"] = e
                edge_vals.append(e)
        m["edge_mass"] = max(edge_vals) if edge_vals else 0.0
        mw, sep = dm.multimodality(post.u, post.weights, ctx.rng)
        m["multimodal_weight"], m["multimodal_separation"] = mw, sep
        ev = next(e for e in out.evidences if e.n == n_post and e.k == post.k)
        m["p_best_k"] = (
            float(np.sum(post.weights[post.k_samples == post.k])) if post.k_samples is not None else 1.0
        )
        m["negative_fraction"] = ev.negative_fraction
        m["sampler_budget"] = any(e.budget_exhausted for e in out.evidences)
        m["logz_err"] = ev.log_z_err
        if np.isfinite(ev.log_z_check):
            m["logz_check_sigma"] = abs(ev.log_z - ev.log_z_check) / max(
                math.hypot(ev.log_z_err, ev.log_z_check_err), 1e-3
            )
        s50 = post.summary()
        if "v_sys_1" in s50 and "W20_1" in s50:
            # Classical measurement in a window around the line, as a by-eye
            # window would be chosen: +-(W20/2 + 25 % + 2 channels)
            half = 0.5 * s50["W20_1"] * 1.25 + 2 * ctx.grid.dv
            out.classic = classic_widths(
                ctx.spectrum.velocity,
                ctx.spectrum.flux - out.baseline_median,
                window=(s50["v_sys_1"] - half, s50["v_sys_1"] + half),
                good=ctx.good,
            )
    out.metrics = m
    out.flags = compute_flags(m, cfg.diagnostics, cfg.rfi.flag_fraction)
    if out.best_n == 0:
        # Profile-specific diagnostics are informational for non-detections
        for key in ("prior_edge", "multimodal", "low_ess", "negative_amplitudes", "wiggle"):
            out.flags[key] = False


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
    "FitResult",
    "ModelEvidence",
    "Posterior",
    "expit",
    "fit_spectrum",
    "spectrum_seed",
]
