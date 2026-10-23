"""Noise models: per-channel sigma, channel correlations and whitening.

The covariance between good channels i and j (original channel indices) is
``C_ij = sigma_i sigma_j rho(|i - j|)``. ``rho`` follows from a smoothing kernel
(Hanning: rho = [1, 2/3, 1/6]; Hanning then decimated by 2: rho = [1, 1/6]) or is
given explicitly. Masked channels are removed, which keeps C banded.

Whitening ``L^-1 x`` with ``C = L L^T`` uses a banded Cholesky factor for the
NumPy backend and a dense inverse factor for JAX (shapes must be static there).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg

from hiprofile._backend import get_xp

MAD_TO_STD = 1.482602218505602


def kernel_correlation(kernel: np.ndarray, decimate: int = 1) -> np.ndarray:
    """Correlation rho[l] of white noise smoothed with ``kernel`` (optionally decimated)."""
    k = np.asarray(kernel, dtype=float)
    acf = np.correlate(k, k, mode="full")[k.size - 1 :]
    acf = acf[::decimate]
    return acf / acf[0]


def correlation_from_spec(spec: str | list[float]) -> np.ndarray:
    if isinstance(spec, (list, tuple, np.ndarray)):
        rho = np.asarray(spec, dtype=float)
        if rho.size == 0 or rho[0] != 1.0:
            rho = np.r_[1.0, rho]
        return rho
    if spec == "white":
        return np.array([1.0])
    if spec == "hanning":
        return kernel_correlation(np.array([0.25, 0.5, 0.25]))
    if spec == "hanning-decimated":
        return kernel_correlation(np.array([0.25, 0.5, 0.25]), decimate=2)
    raise ValueError(f"unknown noise correlation {spec!r}")


def estimate_sigma(
    flux: np.ndarray, good: np.ndarray, rho: np.ndarray, signal_free: np.ndarray | None = None
) -> float:
    """Robust noise estimate from first differences (MAD), corrected for correlation.

    ``var(y[i+1] - y[i]) = 2 sigma^2 (1 - rho_1)``. Differences are robust to
    smooth signal and to most of a line profile. If ``signal_free`` is given,
    only those channels are used.
    """
    use = good.copy()
    if signal_free is not None:
        use &= signal_free
    y = np.where(use, flux, np.nan)
    d = np.diff(y)
    d = d[np.isfinite(d)]
    if d.size < 8:
        raise ValueError("too few good channels to estimate the noise")
    rho1 = rho[1] if rho.size > 1 else 0.0
    mad = np.median(np.abs(d - np.median(d)))
    return float(MAD_TO_STD * mad / np.sqrt(2.0 * (1.0 - rho1)))


@dataclass
class NoiseModel:
    """Gaussian noise on the good channels of one spectrum."""

    sigma: np.ndarray  # (n_good,)
    good_index: np.ndarray  # (n_good,) indices into the channel grid
    rho: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return int(self.sigma.size)

    @property
    def white(self) -> bool:
        return self.rho.size == 1

    @property
    def bandwidth(self) -> int:
        return int(self.rho.size - 1)

    def _banded_cov(self) -> np.ndarray:
        """Lower banded storage ab[i - j, j] = C[i, j]."""
        b = self.bandwidth
        n = self.n
        idx = self.good_index
        ab = np.zeros((b + 1, n))
        ab[0] = self.sigma**2
        for d in range(1, b + 1):
            if d >= n:
                break
            lag = idx[d:] - idx[:-d]
            r = np.where(lag <= b, self.rho[np.minimum(lag, b)], 0.0)
            ab[d, : n - d] = r * self.sigma[d:] * self.sigma[:-d]
        return ab

    def _chol_banded(self) -> np.ndarray:
        if "chol" not in self._cache:
            self._cache["chol"] = scipy.linalg.cholesky_banded(self._banded_cov(), lower=True)
        return self._cache["chol"]

    @property
    def logdet(self) -> float:
        if self.white:
            return float(2.0 * np.sum(np.log(self.sigma)))
        return float(2.0 * np.sum(np.log(self._chol_banded()[0])))

    def dense_whitener(self) -> np.ndarray:
        """Dense L^-1 (n_good x n_good)."""
        if "dense" not in self._cache:
            if self.white:
                self._cache["dense"] = np.diag(1.0 / self.sigma)
            else:
                ab = self._chol_banded()
                n = self.n
                low = np.zeros((n, n))
                for d in range(ab.shape[0]):
                    low += np.diag(ab[d, : n - d], -d)
                self._cache["dense"] = scipy.linalg.solve_triangular(low, np.eye(n), lower=True)
        return self._cache["dense"]

    def whiten(self, x, backend: str = "numpy"):
        """Apply L^-1 along the last axis of ``x`` (shape (..., n_good))."""
        xp = get_xp(backend)
        if self.white:
            return x / xp.asarray(self.sigma)
        if backend == "jax":
            return x @ xp.asarray(self.dense_whitener()).T
        x = np.asarray(x)
        shape = x.shape
        flat = x.reshape(-1, self.n).T
        out = scipy.linalg.solve_banded((self.bandwidth, 0), self._chol_banded(), flat)
        return out.T.reshape(shape)

    def scaled(self, factor: np.ndarray) -> NoiseModel:
        """Noise model with sigma multiplied by ``factor`` (per good channel)."""
        return NoiseModel(sigma=self.sigma * factor, good_index=self.good_index, rho=self.rho)

    def covariance(self) -> np.ndarray:
        """Dense covariance (tests and simulation only)."""
        ab = self._banded_cov()
        n = self.n
        c = np.zeros((n, n))
        for d in range(ab.shape[0]):
            c += np.diag(ab[d, : n - d], -d)
            if d:
                c += np.diag(ab[d, : n - d], d)
        return c


def build_noise(
    flux: np.ndarray,
    good: np.ndarray,
    sigma: np.ndarray | None,
    correlation: str | list[float] = "white",
    signal_free: np.ndarray | None = None,
) -> NoiseModel:
    rho = correlation_from_spec(correlation)
    idx = np.flatnonzero(good)
    if sigma is None:
        s = estimate_sigma(flux, good, rho, signal_free)
        sig = np.full(idx.size, s)
    else:
        sig = np.asarray(sigma, dtype=float)[idx]
    return NoiseModel(sigma=sig, good_index=idx, rho=rho)
