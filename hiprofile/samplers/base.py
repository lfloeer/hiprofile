"""Common sampler interface.

Every backend implements ``run(loglike, prior_transform, ndim, *, seed, n_live,
**options) -> SamplerResult``. ``loglike`` maps a batch of physical parameter
vectors (b, ndim) to log-likelihoods (b,); ``prior_transform`` maps points of the
unit cube (b, ndim) to physical parameters.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import logsumexp

LogLike = Callable[[np.ndarray], np.ndarray]
Transform = Callable[[np.ndarray], np.ndarray]


@dataclass
class SamplerResult:
    samples: np.ndarray  # (n, ndim) physical parameters
    log_weights: np.ndarray  # (n,) unnormalised log importance weights
    log_l: np.ndarray  # (n,)
    log_z: float
    log_z_err: float
    backend: str
    runtime: float = 0.0
    n_like: int = 0
    stats: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def weights(self) -> np.ndarray:
        lw = self.log_weights - logsumexp(self.log_weights)
        return np.exp(lw)

    @property
    def ess(self) -> float:
        w = self.weights
        return float(1.0 / np.sum(w**2)) if w.size else 0.0

    def resample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Indices of ``n`` equally weighted draws (systematic resampling)."""
        return systematic_resample(self.weights, n, rng)


def systematic_resample(weights: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    pos = (rng.uniform() + np.arange(n)) / n
    return np.minimum(np.searchsorted(cdf, pos), weights.size - 1)


class CountingLikelihood:
    """Wrap a batched log-likelihood, count calls and sanitise non-finite values."""

    def __init__(self, loglike: LogLike) -> None:
        self.loglike = loglike
        self.n_calls = 0

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta)
        self.n_calls += theta.shape[0]
        ll = np.asarray(self.loglike(theta), dtype=float)
        return np.where(np.isfinite(ll), ll, -1e300)

    def single(self, theta: np.ndarray) -> float:
        return float(self(theta[None, :])[0])


class Timer:
    def __enter__(self) -> Timer:
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self.t0
