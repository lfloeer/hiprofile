"""Interchangeable samplers behind ``run(loglike, prior_transform, ndim) -> SamplerResult``."""

from __future__ import annotations

from typing import Any

from hiprofile.samplers.base import LogLike, SamplerResult, Transform

BACKENDS = ("nautilus", "dynesty", "ultranest", "smc", "quick")


def run(
    backend: str,
    loglike: LogLike,
    prior_transform: Transform,
    ndim: int,
    *,
    seed: int,
    n_live: int = 400,
    **options: Any,
) -> SamplerResult:
    """Run ``backend`` and return samples, weights and log-evidence."""
    if backend == "nautilus":
        from hiprofile.samplers.nested import run_nautilus as fn
    elif backend == "dynesty":
        from hiprofile.samplers.nested import run_dynesty as fn
    elif backend == "ultranest":
        from hiprofile.samplers.nested import run_ultranest as fn
    elif backend == "smc":
        from hiprofile.samplers.smc import run_smc as fn
    elif backend == "quick":
        from hiprofile.samplers.quick import run_quick as fn
    else:
        raise ValueError(f"unknown sampler backend {backend!r}; choose from {BACKENDS}")
    return fn(loglike, prior_transform, ndim, seed=seed, n_live=n_live, **options)


__all__ = ["BACKENDS", "SamplerResult", "run"]
