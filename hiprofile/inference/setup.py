"""Set-up of a fit: channel grid, noise model, RFI pre-stage and the fit context."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from hiprofile.config import Config
from hiprofile.inference.priors import ModelPrior, build_prior
from hiprofile.likelihood.marginal import MarginalLikelihood, build_likelihood, default_amp_scale
from hiprofile.likelihood.noise import (
    NoiseModel,
    build_noise,
    estimate_correlation,
    estimate_sigma,
    kernel_correlation,
)
from hiprofile.likelihood.rfi import RFIResult, outlier_mask, robust_prefit
from hiprofile.model.kernel import SpectralGrid
from hiprofile.model.response import noise_kernel
from hiprofile.spectrum import Spectrum

log = logging.getLogger(__name__)


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
    max_run = None if cfg.max_width is None else max(1, int(cfg.max_width / ctx.grid.dv))
    resid = ctx.spectrum.flux - res.model if cfg.mask_negative_runs else None
    bad = outlier_mask(res.q, cfg.threshold, cfg.close_holes, max_run, resid)
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
