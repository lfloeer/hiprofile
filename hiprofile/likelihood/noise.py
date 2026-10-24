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


def correlation_from_spec(spec: str | list[float] | np.ndarray) -> np.ndarray:
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


WHITE_FLOOR = 0.005


def _filtered_variance(f: np.ndarray, rho: np.ndarray) -> float:
    """var(sum_j f_j y_{i+j}) / sigma^2 for correlation rho."""
    out = 0.0
    for a in range(f.size):
        for b in range(f.size):
            lag = abs(a - b)
            out += f[a] * f[b] * (rho[lag] if lag < rho.size else 0.0)
    return float(out)


def estimate_white_fraction(
    flux: np.ndarray, good: np.ndarray, rho: np.ndarray, signal_free: np.ndarray | None = None
) -> float:
    """White-noise fraction eps of the mixture rho_eps = (1 - eps) rho + eps delta.

    Real Hanning-smoothed spectra are never exactly Hanning: the model has
    zero power at the Nyquist frequency, the data (re-gridding, quantisation,
    the spectrometer's own window) do not, and whitening with the exact
    kernel blows that component up by orders of magnitude. The ratio of the
    variances of fourth and first differences (robust MAD estimates) is
    linear-fractional in eps and solved for it; fourth differences pick out
    the Nyquist band, where the two components differ most, and suppress
    smooth signal. eps is kept >= WHITE_FLOOR so that the covariance stays
    well conditioned.
    """
    use = good.copy()
    if signal_free is not None:
        use &= signal_free
    y = np.where(use, flux, np.nan)
    f1 = np.array([-1.0, 1.0])
    f4 = np.array([1.0, -4.0, 6.0, -4.0, 1.0])
    d1 = np.convolve(y, f1, mode="valid")
    d4 = np.convolve(y, f4, mode="valid")
    d1, d4 = d1[np.isfinite(d1)], d4[np.isfinite(d4)]
    if d4.size < 16:
        return WHITE_FLOOR
    v1 = (MAD_TO_STD * np.median(np.abs(d1 - np.median(d1)))) ** 2
    v4 = (MAD_TO_STD * np.median(np.abs(d4 - np.median(d4)))) ** 2
    if v1 <= 0:
        return WHITE_FLOOR
    white = np.array([1.0])
    a1, b1 = _filtered_variance(f1, rho), _filtered_variance(f1, white)
    a4, b4 = _filtered_variance(f4, rho), _filtered_variance(f4, white)
    ratio = v4 / v1
    # ratio ((1-e) a1 + e b1) = (1-e) a4 + e b4
    den = (b4 - a4) - ratio * (b1 - a1)
    eps = (ratio * a1 - a4) / den if den != 0 else 1.0
    return float(np.clip(eps, WHITE_FLOOR, 1.0))


def correlation_spectrum(rho: np.ndarray, n_freq: int = 512) -> np.ndarray:
    """S(omega) = 1 + 2 sum_k rho_k cos(k omega) on [0, pi]: the noise power spectrum (unit variance)."""
    w = np.linspace(0.0, np.pi, n_freq)
    k = np.arange(1, rho.size)
    return 1.0 + 2.0 * np.cos(np.outer(w, k)) @ rho[1:]


def make_positive(rho: np.ndarray, floor: float = 0.02) -> np.ndarray:
    """Mix in the least white noise that keeps the power spectrum >= ``floor``.

    A Toeplitz covariance is positive definite (for every length and every
    subset of channels, i.e. also with masks) if its power spectrum is
    positive; ``floor`` also bounds the condition number, which keeps the
    whitening from amplifying frequencies the correlation model says are
    empty (the failure of exact Hanning on real data).
    """
    rho = np.asarray(rho, float).copy()
    rho[0] = 1.0
    s_min = float(correlation_spectrum(rho).min())
    if s_min >= floor:
        return rho
    eps = (floor - s_min) / (1.0 - s_min)
    return with_white(rho, eps)


def _raw_acf(resid: np.ndarray, good: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample correlation rho_0..rho_max_lag over pairs of good channels and the pair counts."""
    r = np.where(good, np.asarray(resid, float), np.nan)
    med = np.nanmedian(r)
    mad = MAD_TO_STD * np.nanmedian(np.abs(r - med))
    r = np.where(np.abs(r - med) < 5.0 * mad, r - med, np.nan)  # clip what the mask missed
    ok = np.isfinite(r)
    c0 = float(np.mean(r[ok] ** 2))
    rho, counts = [1.0], [int(ok.sum())]
    for k in range(1, max_lag + 1):
        pair = ok[:-k] & ok[k:]
        n = int(pair.sum())
        if n < 20:
            break
        rho.append(float(np.mean(r[:-k][pair] * r[k:][pair]) / c0))
        counts.append(n)
    return np.array(rho), np.array(counts)


def estimate_correlation(
    resid: np.ndarray, good: np.ndarray, max_lag: int = 6, floor: float = 0.01
) -> np.ndarray:
    """Noise correlation rho_0..rho_m from residuals (model already subtracted).

    The noise is modelled as white noise smoothed by an unknown short
    symmetric kernel h (taps -q..q, q = max_lag // 2), which is what
    spectrometer windows, Hanning smoothing and regridding do. h is fitted
    so that rho = h * h / |h|^2 matches the sample correlation of the
    residuals over pairs of good channels (masks leave gaps), weighted by the
    pair counts. Unlike a truncated sample correlation this is positive
    definite by construction and needs no choice of the last lag;
    :func:`make_positive` then adds the white floor that bounds the
    condition number.
    """
    from scipy import optimize

    raw, counts = _raw_acf(resid, good, max_lag)
    lags = raw.size - 1
    if lags == 0:
        return np.array([1.0])
    q = max(lags // 2, 1)
    wts = np.sqrt(counts[1:])

    def model(half: np.ndarray) -> np.ndarray:
        h = np.r_[half[:0:-1], half]  # symmetric taps -q..q
        acf = np.correlate(h, h, mode="full")[h.size - 1 :]
        acf = np.pad(acf, (0, max(0, lags + 1 - acf.size)))[: lags + 1]
        return acf / acf[0]

    start = np.r_[1.0, np.zeros(q)]
    res = optimize.least_squares(lambda x: (model(x)[1:] - raw[1:]) * wts, start, method="lm")
    rho = model(res.x)
    keep = np.flatnonzero(np.abs(rho) > 1e-4)
    rho = rho[: int(keep[-1]) + 1] if keep.size else np.array([1.0])
    return make_positive(rho, floor)


def with_white(rho: np.ndarray, eps: float) -> np.ndarray:
    out = (1.0 - eps) * np.asarray(rho, float)
    out[0] = 1.0
    return out


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

    def cholesky_banded(self) -> np.ndarray:
        """Lower banded Cholesky factor, storage ab[d, j] = L[j + d, j]."""
        return self._chol_banded()

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
    correlation: str | list[float] | np.ndarray = "white",
    signal_free: np.ndarray | None = None,
    estimate_white: bool = False,
) -> NoiseModel:
    """Noise model; a correlation spec ending in ``+white`` (e.g. ``hanning+white``)
    or ``estimate_white`` adds a white component whose fraction is estimated
    from the data."""
    if isinstance(correlation, str) and correlation.endswith("+white"):
        estimate_white = True
        correlation = correlation[: -len("+white")]
    rho = correlation_from_spec(correlation)
    if estimate_white and rho.size > 1:
        rho = with_white(rho, estimate_white_fraction(flux, good, rho, signal_free))
    idx = np.flatnonzero(good)
    if sigma is None:
        s = estimate_sigma(flux, good, rho, signal_free)
        sig = np.full(idx.size, s)
    else:
        sig = np.asarray(sigma, dtype=float)[idx]
    return NoiseModel(sigma=sig, good_index=idx, rho=rho)
