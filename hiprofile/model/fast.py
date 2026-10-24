"""Compiled (numba) kernel for the ring/disk basis functions.

The NumPy kernel spends its time in ~30 array passes over the needed Fourier
coefficients. Here one compiled loop per parameter vector evaluates the
Bessel terms, the Gaussian, the phase and the response, and scatters the
result directly into the spectrum folded onto the channel grid; one batched
real FFT then gives the channel values.

J0 and J1 come from tables of piecewise polynomials (Chebyshev interpolants of
degree 9 on intervals of width 1/4 up to x = 2048, built at import from SciPy's
Cephes routines, evaluated with Estrin's scheme; accuracy ~1e-14). J1/x and
J2/x follow from them, with short power series below x = 1; beyond the table
the Hankel expansion is used. The phase factor and the Gaussian are advanced by
exact recurrences over the equally spaced frequencies and re-anchored every 32
steps. The result agrees with the NumPy kernel to ~1e-13.

numba is optional: :func:`available` reports whether this path can be used.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.fft
from scipy import special

try:
    import numba

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False

TABLE_MAX = 2048
STEP = 0.25
N_INT = int(TABLE_MAX / STEP)
DEGREE = 9  # Estrin evaluation below is written out for degree 9
ANCHOR = 32
NEGLIGIBLE = 1e-13
_ZCUT = math.sqrt(-2.0 * math.log(NEGLIGIBLE))


def available() -> bool:
    return HAVE_NUMBA


def _j_terms(x: np.ndarray) -> np.ndarray:  # reference for tests
    """(4, n): J0, J1, J1/x, J2/x without cancellation at small x."""
    j0, j1 = special.j0(x), special.j1(x)
    small = x < 1e-3
    xs = np.where(small, 1.0, x)
    j1x = np.where(small, 0.5 - x**2 / 16 + x**4 / 384, j1 / xs)
    j2x = np.where(small, x / 8 - x**3 / 96 + x**5 / 3072, special.jv(2, x) / xs)
    return np.stack([j0, j1, j1x, j2x])


def _build_tables() -> np.ndarray:
    """Polynomial coefficients of J0 and J1 on [i, i + 1] * STEP, flat (N_INT * (DEGREE + 1) * 2,)."""
    n = DEGREE + 1
    k = np.arange(n)
    nodes = np.cos(np.pi * (k + 0.5) / n)  # Chebyshev nodes on [-1, 1]
    x = (np.arange(N_INT)[:, None] + 0.5 * (nodes[None] + 1.0)) * STEP  # (T, n)
    vals = np.stack([special.j0(x), special.j1(x)])  # (2, T, n)
    # c_j = 2/n sum_k f(x_k) cos(j theta_k), c_0 halved
    cos = np.cos(np.pi * np.outer(np.arange(n), k + 0.5) / n)  # (j, k)
    coef = 2.0 / n * np.einsum("ftk,jk->tfj", vals, cos)
    coef[..., 0] *= 0.5
    # monomial coefficients in t in [-1, 1] for Estrin's scheme; interleaved
    # as (interval, power, function) so one row is contiguous in memory
    mono = np.polynomial.chebyshev.cheb2poly
    out = np.empty((N_INT, n, 2))
    for f in range(2):
        conv = np.zeros((n, n))  # (Chebyshev index j, power)
        for j in range(n):
            pj = mono(np.eye(n)[j])
            conv[j, : pj.size] = pj
        out[:, :, f] = coef[:, f, :] @ conv
    return np.ascontiguousarray(out).ravel()  # flat: interval i starts at 2 * (DEGREE + 1) * i


def _hankel_coeffs(nu: int, n: int) -> np.ndarray:
    mu = 4.0 * nu * nu
    out = np.empty(n)
    acc = 1.0
    out[0] = 1.0
    for k in range(1, n):
        acc *= (mu - (2 * k - 1) ** 2) / (k * 8.0)
        out[k] = acc
    return out


if HAVE_NUMBA:
    _TABLE = _build_tables()
    _H0 = _hankel_coeffs(0, 16)
    _H1 = _hankel_coeffs(1, 16)

    @numba.njit(cache=True)
    def _hankel(nu, coeffs, x):
        inv2 = 1.0 / (x * x)
        p = 0.0
        q = 0.0
        sign = 1.0
        pw = 1.0
        for k in range(8):
            p += sign * coeffs[2 * k] * pw
            q += sign * coeffs[2 * k + 1] * pw
            pw *= inv2
            sign = -sign
        q /= x
        chi = x - (0.5 * nu + 0.25) * math.pi
        return math.sqrt(2.0 / (math.pi * x)) * (p * math.cos(chi) - q * math.sin(chi))

    @numba.njit(cache=True)
    def _small_x(x):
        """J1(x)/x and J2(x)/x from their power series (x < 1)."""
        z = 0.25 * x * x
        s1 = 0.0
        s2 = 0.0
        c1 = 0.5
        c2 = 0.125
        for k in range(9):
            s1 += c1
            s2 += c2
            c1 *= -z / ((k + 1) * (k + 2))
            c2 *= -z / ((k + 1) * (k + 3))
        return s1, x * s2

    @numba.njit(cache=True)
    def _folded(theta, d_omega, n_fft, m, transfer, v0, table, h0, h1):
        """Ring and disk spectra folded modulo m, flat (B * 2 * (m // 2 + 1) * 2) real/imag.

        Written as one flat loop with the table evaluation inlined; calling a
        helper per frequency or indexing a 4-D output array made it 3x slower.
        """
        nb = theta.shape[0]
        half = m // 2 + 1
        nk = transfer.shape[0]
        out = np.zeros(nb * 4 * half)
        inv_step = 1.0 / STEP
        for r in range(nb):
            vc = theta[r, 0] - v0
            width = abs(theta[r, 1])
            sig = abs(theta[r, 2])
            alpha = theta[r, 3]
            kmax = nk
            if sig * d_omega > 0:
                kc = int(_ZCUT / (sig * d_omega)) + 2
                if kc < kmax:
                    kmax = kc
            # recurrences: phase e^{-i w_k vc} = z^k, Gaussian g_k = exp(-a k^2)
            a = 0.5 * (sig * d_omega) ** 2
            zr = math.cos(d_omega * vc)
            zi = -math.sin(d_omega * vc)
            ratio_step = math.exp(-2.0 * a)
            pr = 1.0
            pi_ = 0.0
            g = 1.0
            ratio = math.exp(-a)
            hx = 0.5 * width * d_omega
            base = r * 4 * half
            pos = 0  # k mod m, advanced incrementally
            for k in range(kmax):
                if (k & (ANCHOR - 1)) == 0:
                    arg = k * d_omega * vc
                    pr = math.cos(arg)
                    pi_ = -math.sin(arg)
                    g = math.exp(-a * k * k)
                    ratio = math.exp(-a * (2 * k + 1))
                d = g * transfer[k]
                cr = pr * d
                ci = pi_ * d
                x = hx * k
                if x < TABLE_MAX:
                    u = x * inv_step
                    i = int(u)
                    t = 2.0 * (u - i) - 1.0
                    t2 = t * t
                    t4 = t2 * t2
                    t8 = t4 * t4
                    o = i * 20
                    # Estrin's scheme (degree 9): short dependency chains
                    j0 = (
                        (table[o] + table[o + 2] * t)
                        + (table[o + 4] + table[o + 6] * t) * t2
                        + ((table[o + 8] + table[o + 10] * t) + (table[o + 12] + table[o + 14] * t) * t2) * t4
                        + (table[o + 16] + table[o + 18] * t) * t8
                    )
                    j1 = (
                        (table[o + 1] + table[o + 3] * t)
                        + (table[o + 5] + table[o + 7] * t) * t2
                        + ((table[o + 9] + table[o + 11] * t) + (table[o + 13] + table[o + 15] * t) * t2) * t4
                        + (table[o + 17] + table[o + 19] * t) * t8
                    )
                else:
                    j0 = _hankel(0, h0, x)
                    j1 = _hankel(1, h1, x)
                if x < 1.0:
                    j1x, j2x = _small_x(x)
                else:
                    inv = 1.0 / x
                    j1x = j1 * inv
                    j2x = (2.0 * j1x - j0) * inv
                aj1 = alpha * j1
                d0 = 2.0 * j1x
                d1 = 2.0 * alpha * j2x
                rr = j0 * cr + aj1 * ci
                ri = j0 * ci - aj1 * cr
                dr = d0 * cr + d1 * ci
                di = d0 * ci - d1 * cr
                if pos < half:
                    q = base + 2 * pos
                    out[q] += rr
                    out[q + 1] += ri
                    out[q + 2 * half] += dr
                    out[q + 2 * half + 1] += di
                if k > 0 and 2 * k != n_fft:
                    neg = m - pos if pos > 0 else 0
                    if neg < half:
                        q = base + 2 * neg
                        out[q] += rr
                        out[q + 1] -= ri
                        out[q + 2 * half] += dr
                        out[q + 2 * half + 1] -= di
                pos += 1
                if pos == m:
                    pos = 0
                # advance the recurrences
                tmp = pr * zr - pi_ * zi
                pi_ = pr * zi + pi_ * zr
                pr = tmp
                g *= ratio
                ratio *= ratio_step
        return out


def profile_basis_fast(grid, theta: np.ndarray) -> np.ndarray:
    """Same as ``profile_basis(grid, theta, "numpy")`` using the compiled kernel."""
    theta = np.asarray(theta, dtype=float)
    batch = theta.shape[:-1]
    th = np.ascontiguousarray(theta.reshape(-1, 4))
    y = _folded(th, float(grid.omega[1]), grid.n_fft, grid.n_total, grid.transfer, grid.v0, _TABLE, _H0, _H1)
    y = y.view(np.complex128).reshape(th.shape[0], 2, grid.n_total // 2 + 1)
    out = scipy.fft.irfft(y, n=grid.n_total, axis=-1, overwrite_x=True)[..., : grid.n_chan]
    out *= 1.0 / (grid.h * grid.supersample)
    return out.reshape((*batch, 2, grid.n_chan))
