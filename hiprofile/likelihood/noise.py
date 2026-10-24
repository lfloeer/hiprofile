"""Noise models: per-channel sigma, channel correlations and whitening.

The covariance between good channels i and j (original channel indices) is
``C_ij = sigma_i sigma_j rho(|i - j|)``. ``rho`` follows from a smoothing kernel
(Hanning: rho = [1, 2/3, 1/6]; Hanning then decimated by 2: rho = [1, 1/6]) or is
given explicitly. Masked channels are removed, which keeps C banded.

Whitening ``L^-1 x`` with ``C = L L^T`` uses a banded Cholesky factor for the
NumPy backend and a dense inverse factor for JAX (shapes must be static there).
"""

from __future__ import annotations

import math
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
    robust variances of fourth and first differences is linear-fractional in
    eps and solved for it; fourth differences pick out the Nyquist band,
    where the two components differ most, and suppress smooth signal. eps is
    kept >= WHITE_FLOOR so that the covariance stays well conditioned.
    """
    use = good if signal_free is None else good & signal_free
    v1, v4 = _difference_variances(flux, use, (1, 4))
    if not (np.isfinite(v1) and np.isfinite(v4)) or v1 <= 0:
        return WHITE_FLOOR
    f1, f4 = np.array([-1.0, 1.0]), np.array([1.0, -4.0, 6.0, -4.0, 1.0])
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


def _difference_variances(resid: np.ndarray, good: np.ndarray, orders: tuple[int, ...]) -> np.ndarray:
    """Robust (MAD) variances of the k-th differences of the residuals over good channels."""
    y = np.where(good, np.asarray(resid, float), np.nan)
    out = []
    for k in orders:
        f = np.array([(-1.0) ** (k - j) * math.comb(k, j) for j in range(k + 1)])
        d = np.convolve(y, f, mode="valid")
        d = d[np.isfinite(d)]
        out.append((MAD_TO_STD * np.median(np.abs(d - np.median(d)))) ** 2 if d.size >= 16 else np.nan)
    return np.array(out)


def _kernel_rho(h1: float, h2: float, eps: float) -> np.ndarray:
    taps = np.array([h2, h1, 1.0, h1, h2])
    acf = np.correlate(taps, taps, mode="full")[taps.size - 1 :]
    return with_white(acf / acf[0], eps)


def estimate_correlation(resid: np.ndarray, good: np.ndarray, floor: float = WHITE_FLOOR) -> np.ndarray:
    """Noise correlation rho_0..rho_4 from residuals (model already subtracted).

    The noise is modelled as white noise smoothed by an unknown symmetric
    kernel [h2, h1, 1, h1, h2] with h >= 0 (spectrometer windows, Hanning,
    Gaussian or interpolation smoothing all have non-negative taps) plus a
    white fraction eps. The three parameters are fitted to the ratios of the
    robust variances of the 2nd, 3rd and 4th to the 1st differences of the
    residuals. Differences suppress what the residuals contain besides
    noise at low frequencies (baseline ripple, line residues), which biases a
    sample autocorrelation, and they weight the high frequencies whose power
    the whitening divides by; the Nyquist band in particular, where smoothing
    kernels have (almost) no power and real data some. The result is positive
    definite by construction; eps >= ``floor`` bounds the condition number.
    Masked channels leave gaps: differences across them are skipped.
    """
    from scipy import optimize

    orders = (1, 2, 3, 4)
    v = _difference_variances(resid, good, orders)
    if not np.all(np.isfinite(v)) or v[0] <= 0:
        return np.array([1.0])
    obs = np.log(v[1:] / v[0])
    n = int(np.sum(good))
    # standard error of a log MAD variance (efficiency 0.37) ~ sqrt(2 / (0.37 n))
    err = np.sqrt(5.4 / max(n, 16))
    filters = [np.array([(-1.0) ** (k - j) * math.comb(k, j) for j in range(k + 1)]) for k in orders]

    def predicted(h1: float, h2: float, eps: float) -> np.ndarray:
        rho = _kernel_rho(h1, h2, eps)
        fv = np.array([_filtered_variance(f, rho) for f in filters])
        return np.log(fv[1:] / fv[0])

    # Nested models, chosen by AIC: white; 3 taps + white part; 5 taps + white part.
    # Without the selection, white noise is fitted by a kernel plus a large white
    # part that mimics it (the two are degenerate there).
    candidates: list[tuple[float, tuple[float, float, float]]] = [
        (float(np.sum(((predicted(0.0, 0.0, 1.0) - obs) / err) ** 2)), (0.0, 0.0, 1.0))
    ]
    for free_h2 in (False, True):
        k_par = 3 if free_h2 else 2

        def resid_fn(x: np.ndarray, free_h2: bool = free_h2) -> np.ndarray:
            h1, h2, eps = (x[0], x[1], x[2]) if free_h2 else (x[0], 0.0, x[1])
            return (predicted(h1, h2, eps) - obs) / err

        lo = [0.0, 0.0, floor] if free_h2 else [0.0, floor]
        hi = [5.0, 5.0, 1.0] if free_h2 else [5.0, 1.0]
        starts = ([0.5, 0.05, 0.05], [1.0, 0.4, 0.02]) if free_h2 else ([0.5, 0.05], [0.2, 0.3])
        for start in starts:
            res = optimize.least_squares(resid_fn, start, bounds=(lo, hi), method="trf")
            x = res.x
            par = (x[0], x[1], x[2]) if free_h2 else (x[0], 0.0, x[1])
            candidates.append((2.0 * res.cost + 2 * k_par, par))
    _, best = min(candidates, key=lambda c: c[0])
    rho = _kernel_rho(*best)
    keep = np.flatnonzero(np.abs(rho) > 1e-4)
    return rho[: int(keep[-1]) + 1] if keep.size else np.array([1.0])


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
