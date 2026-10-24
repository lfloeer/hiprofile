"""Nested-sampling backends: nautilus (default), dynesty, UltraNest."""

from __future__ import annotations

import logging
import tempfile
import warnings
from typing import Any

import numpy as np

from hiprofile.samplers.base import CountingLikelihood, LogLike, SamplerResult, Timer, Transform


def run_nautilus(
    loglike: LogLike,
    prior_transform: Transform,
    ndim: int,
    *,
    seed: int,
    n_live: int = 400,
    **options: Any,
) -> SamplerResult:
    import nautilus

    like = CountingLikelihood(loglike)
    # One network is enough in <= 10 dimensions and cuts the training overhead,
    # which dominates for this cheap likelihood.
    options.setdefault("n_networks", 1)
    run_opts = {"f_live": 0.01, "n_eff": 4000, "verbose": False, "timeout": 600.0, "n_like_max": 1_000_000}
    for key in ("f_live", "n_eff", "n_like_max", "timeout", "discard_exploration"):
        if key in options:
            run_opts[key] = options.pop(key)
    with Timer() as t:
        sampler = nautilus.Sampler(
            lambda u: prior_transform(np.atleast_2d(u)),
            like,
            n_dim=ndim,
            n_live=n_live,
            vectorized=True,
            pass_dict=False,
            seed=seed,
            **options,
        )
        sampler.run(**run_opts)
        points, log_w, log_l = sampler.posterior()
    n_eff = float(sampler.n_eff)
    budget = like.n_calls >= run_opts["n_like_max"] or t.elapsed >= run_opts["timeout"]
    return SamplerResult(
        samples=np.asarray(points),
        log_weights=np.asarray(log_w),
        log_l=np.asarray(log_l),
        log_z=float(sampler.log_z),
        # nautilus: the evidence uncertainty scales as ~1/sqrt(n_eff)
        log_z_err=float(1.0 / np.sqrt(max(n_eff, 1.0))),
        backend="nautilus",
        runtime=t.elapsed,
        n_like=like.n_calls,
        stats={"n_eff": n_eff, "n_live": n_live, "budget_exhausted": bool(budget)},
        trace={"log_l": np.sort(np.asarray(log_l))},
    )


def run_dynesty(
    loglike: LogLike,
    prior_transform: Transform,
    ndim: int,
    *,
    seed: int,
    n_live: int = 400,
    **options: Any,
) -> SamplerResult:
    import dynesty

    like = CountingLikelihood(loglike)
    dlogz = options.pop("dlogz", 0.1)
    maxcall = options.pop("maxcall", 1_000_000)
    with Timer() as t, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sampler = dynesty.NestedSampler(
            like.single,
            lambda u: prior_transform(u[None, :])[0],
            ndim,
            nlive=n_live,
            rstate=np.random.default_rng(seed),
            **options,
        )
        sampler.run_nested(dlogz=dlogz, maxcall=maxcall, print_progress=False)
        res = sampler.results
    return SamplerResult(
        samples=np.asarray(res.samples),
        log_weights=np.asarray(res.logwt),
        log_l=np.asarray(res.logl),
        log_z=float(res.logz[-1]),
        log_z_err=float(res.logzerr[-1]),
        backend="dynesty",
        runtime=t.elapsed,
        n_like=like.n_calls,
        stats={
            "n_live": n_live,
            "niter": int(res.niter),
            "eff": float(res.eff),
            "budget_exhausted": like.n_calls >= maxcall,
        },
        trace={"log_l": np.asarray(res.logl), "log_x": np.asarray(res.logvol), "log_z": np.asarray(res.logz)},
    )


def run_ultranest(
    loglike: LogLike,
    prior_transform: Transform,
    ndim: int,
    *,
    seed: int,
    n_live: int = 400,
    **options: Any,
) -> SamplerResult:
    import ultranest

    logging.getLogger("ultranest").setLevel(logging.WARNING)
    like = CountingLikelihood(loglike)
    np.random.seed(seed % (2**32))  # noqa: NPY002 - UltraNest uses the global NumPy RNG
    names = [f"p{i}" for i in range(ndim)]
    run_opts = {
        "show_status": False,
        "viz_callback": False,
        "min_num_live_points": n_live,
        "max_ncalls": 1_000_000,
    }
    run_opts.update(options)
    with Timer() as t, tempfile.TemporaryDirectory() as tmp:
        sampler = ultranest.ReactiveNestedSampler(
            names,
            like,
            transform=lambda u: prior_transform(np.atleast_2d(u)),
            vectorized=True,
            log_dir=tmp,
            resume="overwrite",
        )
        sampler.logger.setLevel(logging.WARNING)
        for handler in sampler.logger.handlers:
            handler.setLevel(logging.WARNING)
        result = sampler.run(**run_opts)
    ws = result["weighted_samples"]
    w = np.asarray(ws["weights"])
    return SamplerResult(
        samples=np.asarray(ws["points"]),
        log_weights=np.log(np.maximum(w, 1e-300)),
        log_l=np.asarray(ws["logl"]),
        log_z=float(result["logz"]),
        log_z_err=float(result["logzerr"]),
        backend="ultranest",
        runtime=t.elapsed,
        n_like=like.n_calls,
        stats={
            "ess": float(result.get("ess", np.nan)),
            "n_live": n_live,
            "budget_exhausted": like.n_calls >= run_opts["max_ncalls"],
        },
        trace={"log_l": np.asarray(ws["logl"])},
    )
