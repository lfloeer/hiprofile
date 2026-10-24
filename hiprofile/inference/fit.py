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
from hiprofile.likelihood.marginal import NAMP, MarginalLikelihood, build_likelihood, default_amp_scale
from hiprofile.likelihood.noise import (
    NoiseModel,
    build_noise,
    estimate_correlation,
    estimate_sigma,
    kernel_correlation,
)
from hiprofile.likelihood.rfi import RFIResult, outlier_mask, robust_prefit
from hiprofile.model.kernel import PROFILE_NPARAM, SpectralGrid
from hiprofile.model.response import noise_kernel
from hiprofile.samplers import run as run_sampler
from hiprofile.samplers.base import SamplerResult, systematic_resample
from hiprofile.spectrum import Spectrum

log = logging.getLogger(__name__)

# Parameters whose posterior width is compared with the prior width
CONSTRAINT_PARAMS = ("delta_v", "delta_v_rand", "alpha", "f")


@dataclass
class ModelEvidence:
    n: int
    k: int
    log_z: float  # with positivity
    log_z_err: float
    log_z_gauss: float  # evidence of the sampling target before the Monte Carlo correction
    method: str  # analytic | marginal | quick | quick-gated
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
    k: int  # most probable baseline order
    names: list[str]
    theta: np.ndarray  # (m, d)
    u: np.ndarray  # (m, d) unit-cube coordinates
    weights: np.ndarray  # (m,) normalised
    beta: np.ndarray  # (m, p) linear parameters (amplitudes, baseline up to K_max, zero-padded)
    log_l: np.ndarray
    derived: dict[str, np.ndarray]  # "<name>_<i>" -> (m,)
    trace: dict[str, np.ndarray] = field(default_factory=dict)
    k_samples: np.ndarray | None = None  # baseline order drawn for each sample
    k_values: list[int] = field(default_factory=list)
    k_probability: np.ndarray | None = None  # P(K | N, y) over k_values

    @property
    def k_max(self) -> int:
        return int(max(self.k_values)) if self.k_values else self.k

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
            if post.k_probability is not None:
                for k, pk in zip(post.k_values, post.k_probability, strict=True):
                    row[f"p_k{k}"] = float(pk)
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
        )

    def likelihood(self, n: int, noise: NoiseModel | None = None) -> MarginalLikelihood:
        """Likelihood for N components with K = k_min..k_max marginalised."""
        pr = self.config.priors
        comps = self.components(n)
        return build_likelihood(
            self.spectrum.flux,
            self.grid,
            noise or self.noise,
            comps,
            self.config.model.k_max,
            np.full(len(comps), self.amp_scale),
            self.baseline_scale,
            baseline_mean=None if pr.baseline_mean is None else np.asarray(pr.baseline_mean),
            baseline_cov=None if pr.baseline_cov is None else np.asarray(pr.baseline_cov),
            noise_scale=pr.noise_scale is not None,
            backend=self.backend,
            k_min=self.config.model.k_min,
        )


def _correlation(cfg: Config) -> tuple[str | list[float] | np.ndarray, bool]:
    """(correlation spec, estimate a white part) for the first noise model.

    ``auto`` and ``estimate`` start from the correlation of the channel
    smoothing in the response; ``estimate`` replaces it after the pre-fit
    (:func:`estimate_noise`).
    """
    corr = cfg.noise.correlation
    if corr in ("auto", "estimate"):
        k = noise_kernel(cfg.model.response)
        return (kernel_correlation(k), True) if k.size > 1 else ("white", False)
    return corr, False


def prepare(spectrum: Spectrum, config: Config, seed: int) -> FitContext:
    spec = spectrum.normalized()
    cfg = config
    good = spec.good
    pr = cfg.priors
    sig_lo = pr.delta_v_rand.low if pr.delta_v_rand.low is not None else 4.0
    sig_min = sig_lo
    if cfg.model.n_rfi:
        sig_min = min(sig_min, pr.rfi_sigma_channels[0] * spec.channel_width)
    dv_hi = pr.delta_v.high if pr.delta_v.high is not None else 800.0
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
    corr, white = _correlation(cfg)
    noise = build_noise(spec.flux, good, use_sigma, corr, signal_free, estimate_white=white)
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


def run_prefit(ctx: FitContext, n_iter: int) -> RFIResult:
    """Robust one-component fit with the outlier mixture (RFI pre-stage)."""
    cfg = ctx.config.rfi
    # The pre-stage line model must not be narrower than a few channels: a
    # bright spike fitted as a "line" would hide its own centre from the mask.
    pr = ctx.config.priors
    sig = pr.delta_v_rand
    floor = cfg.min_sigma_channels * ctx.grid.dv
    high = sig.high if sig.high is not None else 60.0
    low = max(sig.low if sig.low is not None else 4.0, min(floor, 0.9 * high))
    pre_priors = pr.model_copy(update={"delta_v_rand": sig.model_copy(update={"low": low})})
    prior = build_prior(("profile",), pre_priors, ctx.spectrum.velocity)
    res = robust_prefit(
        ctx.spectrum.flux,
        ctx.grid,
        ctx.noise,
        prior,
        ctx.config.model.k_max,
        np.array([ctx.amp_scale]),
        ctx.baseline_scale,
        ctx.rng,
        background=cfg.background,
        n_iter=n_iter,
        k_min=ctx.config.model.k_min,
    )
    return res


def apply_rfi(ctx: FitContext) -> None:
    cfg = ctx.config.rfi
    res = run_prefit(ctx, cfg.n_iter)
    ctx.rfi = res
    bad = outlier_mask(res.q, cfg.threshold, cfg.close_holes, cfg.max_width_channels)
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
        q = np.where(bad, res.q, 0.0)[ctx.noise.good_index]
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


def estimate_noise(ctx: FitContext) -> None:
    """Replace the noise model by one with the correlation measured on the pre-fit residuals.

    Uses the robust model of the RFI pre-stage (or runs a one-iteration
    pre-fit) and the channels it left unmasked. sigma comes, unless the
    spectrum provides one, from first differences corrected with the new
    correlation, like the correlation itself insensitive to the baseline
    ripple and line residues that the main fit models separately.
    """
    res = ctx.rfi if ctx.rfi is not None else run_prefit(ctx, 1)
    resid = ctx.spectrum.flux - res.model
    rho = estimate_correlation(resid, ctx.good)
    idx = np.flatnonzero(ctx.good)
    if ctx.spectrum.sigma is not None and ctx.config.noise.sigma in ("auto", "input"):
        sig = np.asarray(ctx.spectrum.sigma, float)[idx]
    else:
        sig = np.full(idx.size, estimate_sigma(resid, ctx.good, rho))
    ctx.noise = NoiseModel(sigma=sig, good_index=idx, rho=rho)


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
        if cfg.sampler.backend in ("smc", "jax-smc") and cfg.sampler.init_fraction > 0 and quick.proposal:
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


# --------------------------------------------------------------------- evaluation
def model_probabilities(evs: list[ModelEvidence], rate: float, n_max: int) -> dict[int, float]:
    """P(N | y) from the per-(N, K) evidences with a Poisson prior on N and a flat prior on K."""
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


def prior_constraints(prior: ModelPrior, post: Posterior, rng: np.random.Generator) -> dict[str, float]:
    """1 - (posterior 16-84 % width) / (prior 16-84 % width) for the first component's parameters.

    0 means the data did not narrow the prior at all, 1 a perfectly measured parameter.
    """
    draws = prior.transform(rng.uniform(size=(4000, prior.ndim)))
    out: dict[str, float] = {}
    for i, name in enumerate(post.names):
        if not name.endswith("_1"):
            continue
        base = name[:-2]
        if base not in CONSTRAINT_PARAMS:
            continue
        p16, p84 = np.percentile(draws[:, i], [16, 84])
        q16, _, q84 = weighted_quantile(post.theta[:, i], post.weights, [0.16, 0.5, 0.84])
        width_prior = max(p84 - p16, 1e-12)
        out[f"constraint_{base}"] = float(np.clip(1.0 - (q84 - q16) / width_prior, 0.0, 1.0))
    return out


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
        "noise_rho": ctx.noise.rho.tolist(),
        "noise_rho1": float(ctx.noise.rho[1]) if ctx.noise.rho.size > 1 else 0.0,
    }
    if ctx.rfi is not None:
        m["rfi_fraction"] = float(np.mean(ctx.rfi.q > cfg.rfi.threshold))
        m["rfi_p_bad"] = ctx.rfi.p_bad
    n_post = max(out.best_n, 1)
    if cfg.model.n_max >= 1 and n_post in out.posteriors:
        post = out.posteriors[n_post]
        like = ctx.likelihood(n_post)
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
        s50 = post.summary()
        widths = [s50[f"W50_{i}"] for i in range(1, n_post + 1) if f"W50_{i}" in s50]
        if widths:
            m["w50_min_channels"] = float(min(widths) / ctx.grid.dv)
        prior = ctx.prior(n_post)
        edge_vals = []
        bounds = prior.edge_bounds()
        for i, name in enumerate(post.names):
            if name in bounds:
                lo, hi = bounds[name]
                e = dm.edge_mass(post.theta[:, i], post.weights, lo, hi)
                m[f"edge_{name}"] = e
                edge_vals.append(e)
        m["edge_mass"] = max(edge_vals) if edge_vals else 0.0
        m.update(prior_constraints(prior, post, ctx.rng))
        mw, sep = dm.multimodality(post.u, post.weights, ctx.rng)
        m["multimodal_weight"], m["multimodal_separation"] = mw, sep
        ev = next(e for e in out.evidences if e.n == n_post and e.k == post.k)
        if post.k_probability is not None:
            m["p_best_k"] = float(post.k_probability[post.k_values.index(post.k)])
        m["negative_fraction"] = ev.negative_fraction
        m["sampler_budget"] = any(e.budget_exhausted for e in out.evidences)
        m["logz_err"] = ev.log_z_err
        if np.isfinite(ev.log_z_check):
            m["logz_check_sigma"] = abs(ev.log_z - ev.log_z_check) / max(
                math.hypot(ev.log_z_err, ev.log_z_check_err), 1e-3
            )
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
        for key in (
            "prior_edge",
            "multimodal",
            "low_ess",
            "negative_amplitudes",
            "wiggle",
            "prior_dominated",
            "unresolved",
        ):
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
    "FitResult",
    "ModelEvidence",
    "Posterior",
    "expit",
    "fit_spectrum",
    "spectrum_seed",
]
