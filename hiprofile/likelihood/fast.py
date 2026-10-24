"""Compiled (numba) Woodbury algebra of the marginal likelihood.

For each parameter vector: Gram matrices of the (whitened) component basis
with itself, with the whitened Chebyshev basis and with the whitened data,
assembly of A = Lambda^-1 + M~^T M~, a small Cholesky factorisation, ln|A| and
b^T A^-1 b, all in one compiled loop. White noise is applied on the fly
by the banded Cholesky factor of the noise covariance (forward substitution;
white noise is the case of bandwidth 0).
"""

from __future__ import annotations

import math

import numpy as np

try:
    import numba

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


def available() -> bool:
    return HAVE_NUMBA


if HAVE_NUMBA:

    @numba.njit(cache=True)
    def _assemble(basis, r, idx, chol, tw, rw, g_tt, b_t, amp_prec, bw, a, bvec):
        """Whiten basis[r] and fill A = Lambda^-1 + M~^T M~ and b = M~^T r~ in place."""
        q = basis.shape[1]
        nb = tw.shape[0]
        n = idx.shape[0]
        band = chol.shape[0] - 1
        for i in range(q):
            for j in range(n):
                s = basis[r, i, idx[j]]
                for d in range(1, min(band, j) + 1):
                    s -= chol[d, j - d] * bw[i, j - d]
                bw[i, j] = s / chol[0, j]
        for i in range(q):
            for k in range(i, q):
                s = 0.0
                for j in range(n):
                    s += bw[i, j] * bw[k, j]
                a[i, k] = s
                a[k, i] = s
            a[i, i] += amp_prec[i]
            for k in range(nb):
                s = 0.0
                for j in range(n):
                    s += bw[i, j] * tw[k, j]
                a[i, q + k] = s
                a[q + k, i] = s
            s = 0.0
            for j in range(n):
                s += bw[i, j] * rw[j]
            bvec[i] = s
        for i in range(nb):
            for k in range(nb):
                a[q + i, q + k] = g_tt[i, k]
            bvec[q + i] = b_t[i]

    @numba.njit(cache=True)
    def gram(basis, idx, chol, tw, rw, g_tt, b_t, amp_prec):
        """A (B, p, p) and b (B, p) for every parameter vector (used for beta | theta, y)."""
        nbatch, q, _ = basis.shape
        p = q + tw.shape[0]
        a = np.empty((nbatch, p, p))
        bvec = np.empty((nbatch, p))
        bw = np.empty((q, idx.shape[0]))
        for r in range(nbatch):
            _assemble(basis, r, idx, chol, tw, rw, g_tt, b_t, amp_prec, bw, a[r], bvec[r])
        return a, bvec

    @numba.njit(cache=True)
    def woodbury(basis, idx, chol, tw, rw, g_tt, b_t, amp_prec, rr, const):
        """ln L for basis (B, q, n_chan) evaluated on the good channels idx.

        chol (bw + 1, n_good) is the lower banded Cholesky factor of C in the
        storage chol[d, j] = L[j + d, j]; tw (nb, n_good), rw (n_good) are
        whitened; g_tt = tw tw^T + Lambda_b^-1, b_t = tw rw;
        const = ln|C| + ln|Lambda| + n ln 2 pi.
        """
        nbatch, q, _ = basis.shape
        p = q + tw.shape[0]
        out = np.empty(nbatch)
        bw = np.empty((q, idx.shape[0]))
        a = np.empty((p, p))
        bvec = np.empty(p)
        for r in range(nbatch):
            _assemble(basis, r, idx, chol, tw, rw, g_tt, b_t, amp_prec, bw, a, bvec)
            # Cholesky A = L L^T in place (lower triangle), ln|A|, b^T A^-1 b
            logdet = 0.0
            ok = True
            for j in range(p):
                s = a[j, j]
                for k in range(j):
                    s -= a[j, k] * a[j, k]
                if s <= 0.0:
                    ok = False
                    break
                d = math.sqrt(s)
                a[j, j] = d
                logdet += 2.0 * math.log(d)
                for i in range(j + 1, p):
                    s = a[i, j]
                    for k in range(j):
                        s -= a[i, k] * a[j, k]
                    a[i, j] = s / d
            if not ok:
                out[r] = -1e300
                continue
            # forward solve L z = b; b^T A^-1 b = z^T z
            quad = 0.0
            for i in range(p):
                s = bvec[i]
                for k in range(i):
                    s -= a[i, k] * bvec[k]
                bvec[i] = s / a[i, i]
                quad += bvec[i] * bvec[i]
            out[r] = -0.5 * (rr - quad + const + logdet)
        return out
