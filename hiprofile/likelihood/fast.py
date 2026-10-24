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
    def woodbury(basis, idx, chol, tw, rw, g_tt, b_t, amp_prec, rr, const, log_lam_b, k_min):
        """ln L_K and the amplitude moments for every baseline order K = k_min..k_max.

        basis (B, q, n_chan) is evaluated on the good channels idx; chol
        (bw + 1, n_good) is the lower banded Cholesky factor of C in the storage
        chol[d, j] = L[j + d, j]; tw (nb, n_good), rw (n_good) are whitened;
        g_tt = tw tw^T + Lambda_b^-1 (Lambda_b diagonal), b_t = tw rw;
        const = ln|C| + n ln 2 pi + ln|Lambda_amp|; log_lam_b[k] = ln lambda_k of
        coefficient k. A_K is the leading (q + K + 1) block of A, so one
        Cholesky factorisation serves all K.

        Returns ll (B, nK), E[a | theta, y, K] (B, nK, q) and
        Cov[a | theta, y, K] (B, nK, q, q); an order whose block is not positive
        definite gets ll = -1e300 and the identity covariance.
        """
        nbatch, q, _ = basis.shape
        nb = tw.shape[0]
        p = q + nb
        n_k = nb - k_min
        out = np.full((nbatch, n_k), -1e300)
        amp_mean = np.zeros((nbatch, n_k, q))
        amp_cov = np.zeros((nbatch, n_k, q, q))
        for r in range(nbatch):
            for kk in range(n_k):
                for i in range(q):
                    amp_cov[r, kk, i, i] = 1.0
        bw = np.empty((q, idx.shape[0]))
        a = np.empty((p, p))
        bvec = np.empty(p)
        z = np.empty(p)
        work = np.empty(p)
        logdet_cum = np.empty(p + 1)
        quad_cum = np.empty(p + 1)
        lam_cum = np.empty(nb + 1)
        lam_cum[0] = 0.0
        for k in range(nb):
            lam_cum[k + 1] = lam_cum[k] + log_lam_b[k]
        for r in range(nbatch):
            _assemble(basis, r, idx, chol, tw, rw, g_tt, b_t, amp_prec, bw, a, bvec)
            # Cholesky A = L L^T in place (lower triangle); stop at the first
            # non-positive pivot, the leading blocks before it stay valid
            jfail = p
            logdet_cum[0] = 0.0
            for j in range(p):
                s = a[j, j]
                for k in range(j):
                    s -= a[j, k] * a[j, k]
                if s <= 0.0:
                    jfail = j
                    break
                d = math.sqrt(s)
                a[j, j] = d
                logdet_cum[j + 1] = logdet_cum[j] + 2.0 * math.log(d)
                for i in range(j + 1, p):
                    s = a[i, j]
                    for k in range(j):
                        s -= a[i, k] * a[j, k]
                    a[i, j] = s / d
            # forward solve L z = b; b_K^T A_K^-1 b_K = sum_{i < m} z_i^2
            quad_cum[0] = 0.0
            for i in range(jfail):
                s = bvec[i]
                for k in range(i):
                    s -= a[i, k] * z[k]
                z[i] = s / a[i, i]
                quad_cum[i + 1] = quad_cum[i] + z[i] * z[i]
            for kk in range(n_k):
                kord = k_min + kk
                m = q + kord + 1
                if m > jfail:
                    continue
                out[r, kk] = -0.5 * (rr - quad_cum[m] + const + lam_cum[kord + 1] + logdet_cum[m])
                # mean = L_m^-T z_m
                for i in range(m - 1, -1, -1):
                    s = z[i]
                    for k in range(i + 1, m):
                        s -= a[k, i] * work[k]
                    work[i] = s / a[i, i]
                for i in range(q):
                    amp_mean[r, kk, i] = work[i]
                # covariance block: amplitude columns of A_m^-1
                for c in range(q):
                    for i in range(m):
                        s = 1.0 if i == c else 0.0
                        for k in range(i):
                            s -= a[i, k] * work[k]
                        work[i] = s / a[i, i]
                    for i in range(m - 1, -1, -1):
                        s = work[i]
                        for k in range(i + 1, m):
                            s -= a[k, i] * work[k]
                        work[i] = s / a[i, i]
                    for i in range(q):
                        amp_cov[r, kk, i, c] = work[i]
        return out, amp_mean, amp_cov
