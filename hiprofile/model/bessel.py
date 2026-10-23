"""Bessel functions J0, J1/x and J2/x for the NumPy and JAX backends.

The model needs, at x = Delta v * omega / 2,

* J0(x)              ring
* J1(x)              ring asymmetry
* 2 J1(x) / x        disk
* 2 J2(x) / x        disk asymmetry, J2(x)/x = (2 J1(x)/x - J0(x)) / x

``J2/x`` and ``J1/x`` are evaluated without cancellation for x -> 0 (Delta v -> 0).

NumPy uses the Cephes routines from SciPy. JAX has no j0/j1, so this module
implements them with a power series for x < X_SWITCH and the Hankel asymptotic
expansion above. Both branches are evaluated on "safe" arguments so that the
discarded branch never produces inf/nan, which keeps gradients finite.
Accuracy against SciPy is about 3e-12 absolute (see tests/test_bessel.py).
"""

from __future__ import annotations

import math
from types import ModuleType

import numpy as np
from scipy import special

X_SWITCH = 12.5
N_SERIES = 48
N_ASYMPTOTIC = 12


def _series_coeffs(nu: int, n: int) -> np.ndarray:
    """c_k = (-1)^k / (k! (k+nu)!) for J_nu(x) = (x/2)^nu sum_k c_k (x^2/4)^k."""
    return np.array(
        [(-1) ** k / (math.factorial(k) * math.factorial(k + nu)) for k in range(n)],
        dtype=float,
    )


def _hankel_coeffs(nu: int, n: int) -> np.ndarray:
    """a_k(nu) = prod_{j=1..k} (4 nu^2 - (2j-1)^2) / (k! 8^k)."""
    mu = 4.0 * nu * nu
    out = np.empty(n)
    acc = 1.0
    out[0] = 1.0
    for k in range(1, n):
        acc *= (mu - (2 * k - 1) ** 2) / (k * 8.0)
        out[k] = acc
    return out


_S0 = _series_coeffs(0, N_SERIES)
_S1 = _series_coeffs(1, N_SERIES)
_S2 = _series_coeffs(2, N_SERIES)
_H0 = _hankel_coeffs(0, 2 * N_ASYMPTOTIC)
_H1 = _hankel_coeffs(1, 2 * N_ASYMPTOTIC)


def _horner(xp: ModuleType, coeffs: np.ndarray, z):
    acc = xp.zeros_like(z) + coeffs[-1]
    for c in coeffs[-2::-1]:
        acc = acc * z + c
    return acc


def _hankel(xp: ModuleType, coeffs: np.ndarray, nu: int, x):
    """Hankel asymptotic expansion of J_nu for large x."""
    inv = 1.0 / x
    inv2 = inv * inv
    p_c = coeffs[0::2] * np.array([(-1) ** k for k in range(N_ASYMPTOTIC)])
    q_c = coeffs[1::2] * np.array([(-1) ** k for k in range(N_ASYMPTOTIC)])
    p = _horner(xp, p_c, inv2)
    q = _horner(xp, q_c, inv2) * inv
    chi = x - (0.5 * nu + 0.25) * math.pi
    return xp.sqrt(2.0 / (math.pi * x)) * (p * xp.cos(chi) - q * xp.sin(chi))


def bessel_terms_jax(x):
    """Return (J0, J1, J1/x, J2/x) for x >= 0 with jax.numpy."""
    import jax.numpy as jnp

    small = x < X_SWITCH
    xs = jnp.where(small, x, 0.0)
    xl = jnp.where(small, X_SWITCH, x)

    z = 0.25 * xs * xs
    j0_s = _horner(jnp, _S0, z)
    j1x_s = 0.5 * _horner(jnp, _S1, z)
    j2x_s = 0.25 * xs * _horner(jnp, _S2, z)

    j0_l = _hankel(jnp, _H0, 0, xl)
    j1_l = _hankel(jnp, _H1, 1, xl)
    j1x_l = j1_l / xl
    j2x_l = (2.0 * j1x_l - j0_l) / xl

    j0 = jnp.where(small, j0_s, j0_l)
    j1x = jnp.where(small, j1x_s, j1x_l)
    j1 = jnp.where(small, j1x_s * xs, j1_l)
    j2x = jnp.where(small, j2x_s, j2x_l)
    return j0, j1, j1x, j2x


_SMALL = 0.05
_S2_SHORT = _S2[:6]
_S1_SHORT = _S1[:6]


def bessel_terms_numpy(x: np.ndarray):
    """Return (J0, J1, J1/x, J2/x) for x >= 0 with NumPy/SciPy."""
    x = np.asarray(x, dtype=float)
    j0 = special.j0(x)
    j1 = special.j1(x)
    small = x < _SMALL
    if small.any():
        xs = np.where(small, 1.0, x)
        j1x = j1 / xs
        j2x = (2.0 * j1x - j0) / xs
        z = 0.25 * x[small] ** 2
        j1x[small] = 0.5 * _horner(np, _S1_SHORT, z)
        j2x[small] = 0.25 * x[small] * _horner(np, _S2_SHORT, z)
    else:
        j1x = j1 / x
        j2x = (2.0 * j1x - j0) / x
    return j0, j1, j1x, j2x


def bessel_terms(x, backend: str = "numpy"):
    if backend == "jax":
        return bessel_terms_jax(x)
    return bessel_terms_numpy(x)
