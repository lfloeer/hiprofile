"""Evaluation of a fitted spectrum: model probabilities, derived quantities, metrics and flags."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from scipy.special import logsumexp

from hiprofile.diagnostics import metrics as dm
from hiprofile.diagnostics.flags import compute_flags
from hiprofile.inference.derived import classic_measurements
from hiprofile.inference.priors import ModelPrior
from hiprofile.inference.results import (
    CONSTRAINT_PARAMS,
    FitResult,
    ModelEvidence,
    Posterior,
    weighted_quantile,
)
from hiprofile.inference.setup import FitContext
from hiprofile.likelihood.marginal import NAMP, MarginalLikelihood
from hiprofile.samplers.base import systematic_resample

log = logging.getLogger(__name__)


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
        m["residual_lowfreq_lnp"], m["residual_lowfreq_ratio"] = dm.residual_lowfreq(
            rw, cfg.diagnostics.structure_fmax
        )
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
        if prior.band is not None:  # a centre pressed against the containment limit: line cut by the window
            r = prior.band_position(post.theta)
            for j in range(r.shape[1]):
                m[f"edge_band_{j + 1}"] = e = dm.edge_mass(r[:, j], post.weights, 0.0, 1.0)
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
            out.classic = classic_measurements(
                ctx.spectrum.velocity,
                ctx.spectrum.flux - out.baseline_median,
                float(np.median(ctx.noise.sigma)),
                s50["v_sys_1"],
                s50["W20_1"],
                good=ctx.good,
                hanning="hanning" in cfg.model.response.lower(),
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
