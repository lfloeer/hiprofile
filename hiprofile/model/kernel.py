"""Fourier-space kernel for the Stewart et al. (2014) line profile.

The profile of one component is linear in two amplitudes,

    m(v) = a_ring R(v; theta) + a_disk D(v; theta),     theta = (v_ctr, dv, dv_rand, alpha)

with a_ring = S (1 - f) and a_disk = S f. R and D both have unit integral. With
x = dv * omega / 2 their Fourier transforms are

    R^(omega) = [J0(x) - i alpha J1(x)]             * P(omega)
    D^(omega) = [2 J1(x)/x - 2 i alpha J2(x)/x]     * P(omega)
    P(omega)  = exp(-i omega v_ctr) exp(-dv_rand^2 omega^2 / 2) H(omega)

where H is the channel/spectrometer response. The ring is the double horn
2/(pi dv) (1 + 2 alpha u/dv) / sqrt(1 - (2u/dv)^2), the disk the half ellipse
4/(pi dv) (1 + 2 alpha u/dv) sqrt(1 - (2u/dv)^2), u = v - v_ctr.

The inverse transform is evaluated with a real FFT on a periodic fine grid with
spacing dv_chan / supersample, zero-padded so that profiles inside the band do
not wrap around, and sampled at the channel centres. Channel integration is
exact (it is part of H), so supersampling only controls the truncation of the
spectrum at the fine-grid Nyquist frequency.

Narrow Gaussian components (used for RFI) have a single unit-integral basis
function G^(omega) = P(omega) with theta = (v_ctr, sigma).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.fft

from hiprofile._backend import get_xp
from hiprofile.model import fast as _fast
from hiprofile.model.bessel import bessel_terms
from hiprofile.model.response import transfer_function

PROFILE_NPARAM = 4
# Use the compiled (numba) kernel for the NumPy backend when numba is installed
USE_COMPILED = True
GAUSS_NPARAM = 2


def next_fast_len(n: int) -> int:
    """Smallest 5-smooth integer >= n (fast for every FFT library)."""
    best = None
    p5 = 1
    while p5 < 2 * n:
        p35 = p5
        while p35 < 2 * n:
            p = p35
            while p < n:
                p *= 2
            if best is None or p < best:
                best = p
            p35 *= 3
        p5 *= 5
    assert best is not None
    return best


def default_supersample(dv_chan: float, sigma_min: float, tol: float = 1e-7) -> int:
    """Supersampling so that the Gaussian truncation at the fine Nyquist is < tol."""
    need = math.sqrt(-2.0 * math.log(tol)) / math.pi  # sigma * omega_max >= that * pi
    return max(1, math.ceil(need * dv_chan / max(sigma_min, 1e-6)))


@dataclass(frozen=True)
class SpectralGrid:
    """Uniform channel grid plus the fine FFT grid used to evaluate the model.

    Parameters
    ----------
    v0 : centre of the first channel [km/s]
    dv : channel width [km/s], > 0
    n_chan : number of channels
    supersample : fine-grid points per channel
    n_total : length of the periodic grid in channels (>= n_chan, padding)
    response : channel/spectrometer response, see :mod:`hiprofile.model.response`
    """

    v0: float
    dv: float
    n_chan: int
    supersample: int = 4
    n_total: int = 0
    response: str = "boxcar"
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.dv <= 0:
            raise ValueError("channel width must be positive (sort velocities ascending)")
        if self.n_total == 0:
            object.__setattr__(self, "n_total", next_fast_len(2 * self.n_chan))
        if self.n_total < self.n_chan:
            raise ValueError("n_total must be >= n_chan")

    @classmethod
    def from_velocities(
        cls,
        velocity: np.ndarray,
        *,
        supersample: int | None = None,
        sigma_min: float = 4.0,
        pad: float = 0.1,
        max_width: float | None = 1000.0,
        response: str = "boxcar",
    ) -> SpectralGrid:
        """Build the grid for an ascending, uniform velocity axis.

        ``pad`` is the minimum padding as a fraction of the band; it is increased
        so that a profile of total extent ``max_width`` [km/s] (line width plus
        Gaussian wings) centred inside the band cannot wrap into the band.
        """
        v = np.asarray(velocity, dtype=float)
        dv = float((v[-1] - v[0]) / (v.size - 1))
        n = int(v.size)
        n_pad = math.ceil(pad * n)
        if max_width is not None:
            n_pad = max(n_pad, math.ceil(0.5 * max_width / dv))
        ss = supersample if supersample is not None else default_supersample(dv, sigma_min)
        return cls(
            v0=float(v[0]),
            dv=dv,
            n_chan=n,
            supersample=int(ss),
            n_total=next_fast_len(n + n_pad),
            response=response,
        )

    @property
    def velocity(self) -> np.ndarray:
        return self.v0 + self.dv * np.arange(self.n_chan)

    @property
    def h(self) -> float:
        """Fine-grid spacing."""
        return self.dv / self.supersample

    @property
    def n_fft(self) -> int:
        return self.n_total * self.supersample

    @property
    def omega(self) -> np.ndarray:
        if "omega" not in self._cache:
            n = self.n_fft
            self._cache["omega"] = 2.0 * np.pi * np.arange(n // 2 + 1) / (n * self.h)
        return self._cache["omega"]

    @property
    def transfer(self) -> np.ndarray:
        if "transfer" not in self._cache:
            self._cache["transfer"] = transfer_function(self.response, self.omega, self.dv)
        return self._cache["transfer"]

    def with_response(self, response: str, supersample: int | None = None) -> SpectralGrid:
        return SpectralGrid(
            v0=self.v0,
            dv=self.dv,
            n_chan=self.n_chan,
            supersample=self.supersample if supersample is None else supersample,
            n_total=self.n_total,
            response=response,
        )


def _split_theta(xp, theta, nparam):
    theta = xp.asarray(theta)
    if theta.shape[-1] != nparam:
        raise ValueError(f"theta must have trailing dimension {nparam}")
    return [theta[..., i : i + 1] for i in range(nparam)]


def profile_fourier(grid: SpectralGrid, theta, backend: str = "numpy"):
    """Fourier coefficients of (R, D), shape (..., 2, n_fft//2 + 1).

    The phase reference is the first channel centre ``grid.v0``.
    """
    if backend == "numpy":
        return _profile_fourier_numpy(grid, np.asarray(theta, dtype=float))
    xp = get_xp(backend)
    vc, width, sig, alpha = _split_theta(xp, theta, PROFILE_NPARAM)
    omega = xp.asarray(grid.omega)
    x = 0.5 * xp.abs(width) * omega
    j0, j1, j1x, j2x = bessel_terms(x, backend)
    common = (
        xp.exp(-1j * omega * (vc - grid.v0)) * xp.exp(-0.5 * (sig * omega) ** 2) * xp.asarray(grid.transfer)
    )
    ring = (j0 - 1j * alpha * j1) * common
    disk = (2.0 * j1x - 2j * alpha * j2x) * common
    return xp.stack([ring, disk], axis=-2)


# Fourier coefficients with |exp(-sigma^2 omega^2 / 2) H(omega)| below this are
# set to zero in the NumPy path (the Bessel terms are bounded by 1).
NEGLIGIBLE = 1e-13


# exp(-z^2 / 2) < NEGLIGIBLE for z > _ZCUT
_ZCUT = math.sqrt(-2.0 * math.log(NEGLIGIBLE))


def _needed_pairs(grid: SpectralGrid, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flattened (row, k) indices of the Fourier coefficients above NEGLIGIBLE."""
    nk = grid.omega.size
    d_omega = grid.omega[1]
    lengths = np.minimum(nk, np.ceil(_ZCUT / (np.abs(sigma) * d_omega + 1e-300)) + 1)
    lengths = lengths.astype(np.int64)
    rows = np.repeat(np.arange(sigma.size), lengths)
    starts = np.cumsum(lengths) - lengths
    ks = np.arange(rows.size) - np.repeat(starts, lengths)
    return rows, ks


def _profile_terms_numpy(grid: SpectralGrid, th: np.ndarray):
    """Needed (row, k) pairs and the real/imag parts of the ring and disk terms."""
    omega = grid.omega
    rows, ks = _needed_pairs(grid, th[:, 2])
    w = omega[ks]
    d = np.exp(-0.5 * (th[rows, 2] * w) ** 2) * grid.transfer[ks]
    x = 0.5 * np.abs(th[rows, 1]) * w
    j0, j1, j1x, j2x = bessel_terms(x, "numpy")
    arg = w * (th[rows, 0] - grid.v0)
    cr = np.cos(arg) * d
    ci = np.sin(arg) * d
    ci *= -1.0
    a = th[rows, 3]
    aj1 = a * j1
    d0 = 2.0 * j1x
    d1 = 2.0 * a * j2x
    terms = (
        (j0 * cr + aj1 * ci, j0 * ci - aj1 * cr),
        (d0 * cr + d1 * ci, d0 * ci - d1 * cr),
    )
    return rows, ks, terms


def _profile_fourier_numpy(grid: SpectralGrid, theta: np.ndarray) -> np.ndarray:
    if theta.shape[-1] != PROFILE_NPARAM:
        raise ValueError(f"theta must have trailing dimension {PROFILE_NPARAM}")
    batch = theta.shape[:-1]
    th = theta.reshape(-1, PROFILE_NPARAM)
    nk = grid.omega.size
    rows, ks, terms = _profile_terms_numpy(grid, th)
    out = np.zeros((th.shape[0], 2, nk), dtype=complex)
    for i, (re, im) in enumerate(terms):
        out[rows, i, ks] = re + 1j * im
    return out.reshape((*batch, 2, nk))


def _fold_to_channels(grid: SpectralGrid, n_rows: int, n_basis: int, rows, ks, terms):
    """Sample the fine-grid inverse transform at the channel centres.

    Taking every s-th point of a length-N real inverse DFT equals the length-M
    (M = N / s) inverse DFT of the spectrum folded modulo M. Only the non-zero
    coefficients are scattered, so the cost scales with their number.
    """
    m = grid.n_total
    n = grid.n_fft
    half = m // 2 + 1
    # Positive frequencies k and their conjugate partners N - k (0 < k < N/2).
    pos = ks % m
    neg = (-ks) % m
    has_neg = (ks > 0) & (2 * ks != n)
    out = np.empty((n_rows, n_basis, m))
    size = n_rows * half
    for b, (re, im) in enumerate(terms):
        keep_p = pos < half
        keep_n = has_neg & (neg < half)
        idx = np.concatenate([rows[keep_p] * half + pos[keep_p], rows[keep_n] * half + neg[keep_n]])
        wr = np.concatenate([re[keep_p], re[keep_n]])
        wi = np.concatenate([im[keep_p], -im[keep_n]])
        yr = np.bincount(idx, wr, minlength=size).reshape(n_rows, half)
        yi = np.bincount(idx, wi, minlength=size).reshape(n_rows, half)
        out[:, b, :] = scipy.fft.irfft(yr + 1j * yi, n=m, axis=-1)
    out *= 1.0 / (grid.h * grid.supersample)
    return out[..., : grid.n_chan]


def _profile_basis_numpy(grid: SpectralGrid, theta: np.ndarray) -> np.ndarray:
    if theta.shape[-1] != PROFILE_NPARAM:
        raise ValueError(f"theta must have trailing dimension {PROFILE_NPARAM}")
    batch = theta.shape[:-1]
    th = theta.reshape(-1, PROFILE_NPARAM)
    rows, ks, terms = _profile_terms_numpy(grid, th)
    out = _fold_to_channels(grid, th.shape[0], 2, rows, ks, terms)
    return out.reshape((*batch, 2, grid.n_chan))


def gaussian_fourier(grid: SpectralGrid, theta, backend: str = "numpy"):
    """Fourier coefficients of a unit-integral Gaussian, shape (..., 1, n_k)."""
    xp = get_xp(backend)
    vc, sig = _split_theta(xp, theta, GAUSS_NPARAM)
    omega = xp.asarray(grid.omega)
    g = xp.exp(-1j * omega * (vc - grid.v0)) * xp.exp(-0.5 * (sig * omega) ** 2) * xp.asarray(grid.transfer)
    return g[..., None, :]


def _to_channels(grid: SpectralGrid, coeffs, backend: str):
    if backend == "numpy":
        fine = scipy.fft.irfft(coeffs, n=grid.n_fft, axis=-1, overwrite_x=True)
        fine *= 1.0 / grid.h
        return fine[..., : grid.n_chan * grid.supersample : grid.supersample]
    xp = get_xp(backend)
    fine = xp.fft.irfft(coeffs, n=grid.n_fft, axis=-1) / grid.h
    return fine[..., : grid.n_chan * grid.supersample : grid.supersample]


def profile_basis(grid: SpectralGrid, theta, backend: str = "numpy"):
    """Channel values of the unit-flux ring and disk basis, shape (..., 2, n_chan).

    ``theta[..., :] = (v_ctr, dv, dv_rand, alpha)``; any leading batch shape.
    """
    if backend == "numpy":
        if USE_COMPILED and _fast.available():
            return _fast.profile_basis_fast(grid, theta)
        return _profile_basis_numpy(grid, np.asarray(theta, dtype=float))
    return _to_channels(grid, profile_fourier(grid, theta, backend), backend)


def gaussian_basis(grid: SpectralGrid, theta, backend: str = "numpy"):
    """Channel values of a unit-flux Gaussian, shape (..., 1, n_chan)."""
    return _to_channels(grid, gaussian_fourier(grid, theta, backend), backend)


def profile(grid: SpectralGrid, params, backend: str = "numpy"):
    """Full profile for physical parameters (v_ctr, dv, dv_rand, alpha, S, f)."""
    xp = get_xp(backend)
    params = xp.asarray(params)
    basis = profile_basis(grid, params[..., :4], backend)
    s = params[..., 4:5]
    f = params[..., 5:6]
    return s * (1.0 - f) * basis[..., 0, :] + s * f * basis[..., 1, :]


def chebyshev_basis(n_chan: int, order: int) -> np.ndarray:
    """Chebyshev polynomials T_0..T_order on x in [-1, 1] over the band, (order+1, n)."""
    if order < 0:
        return np.zeros((0, n_chan))
    x = np.linspace(-1.0, 1.0, n_chan)
    return np.polynomial.chebyshev.chebvander(x, order).T.copy()
