"""The compiled (numba) kernel and likelihood agree with the NumPy reference paths."""

import numpy as np
import pytest

from hiprofile.likelihood import fast as fast_ll
from hiprofile.likelihood.marginal import build_likelihood
from hiprofile.likelihood.noise import build_noise
from hiprofile.model import SpectralGrid, kernel
from hiprofile.model import fast as fast_k

pytestmark = pytest.mark.skipif(not fast_k.available(), reason="numba not installed")


def _numpy_basis(grid, theta):
    return kernel._profile_basis_numpy(grid, np.asarray(theta, float))


def test_bessel_tables():
    x = np.concatenate([np.linspace(0, 3000, 300001), np.geomspace(1e-9, 5000, 5000)])
    ref = fast_k._j_terms(x)
    t = fast_k._TABLE
    assert t.ndim == 1
    # evaluate through the kernel on a single-row grid is indirect; check the table itself
    n = fast_k.DEGREE + 1
    i = np.minimum((x / fast_k.STEP).astype(int), fast_k.N_INT - 1)
    tt = 2.0 * (x / fast_k.STEP - i) - 1.0
    inside = x < fast_k.TABLE_MAX
    coef = t.reshape(fast_k.N_INT, n, 2)[i[inside]]  # (m, power, f)
    powers = tt[inside][:, None] ** np.arange(n)[None]
    vals = np.einsum("mp,mpf->fm", powers, coef)
    np.testing.assert_allclose(vals, ref[:2, inside], atol=2e-14)


@pytest.mark.parametrize(
    "n,dv,response", [(512, 5.0, "boxcar"), (300, 10.24, "boxcar+hanning"), (1024, 2.0, "tukey(0.5)")]
)
def test_kernel_matches_numpy(rng, n, dv, response):
    v = 100.0 + np.arange(n) * dv
    g = SpectralGrid.from_velocities(v, response=response)
    th = np.c_[
        rng.uniform(v[0], v[-1], 64),
        rng.uniform(0, 600, 64),
        np.exp(rng.uniform(np.log(2), np.log(60), 64)),
        rng.uniform(-1, 1, 64),
    ]
    th[:4, 1] = [0.0, 1e-5, 0.5, 5000.0]  # Delta v -> 0, small x series, Hankel branch
    a = _numpy_basis(g, th)
    b = fast_k.profile_basis_fast(g, th)
    assert np.max(np.abs(a - b)) < 1e-12 * np.max(np.abs(a))


def _theta(rng, n, n_comp):
    cols = []
    for _ in range(n_comp):  # distinct components (identical ones make A singular)
        cols.append(
            np.c_[
                rng.uniform(1200, 2000, n),
                rng.uniform(0, 400, n),
                rng.uniform(5, 40, n),
                rng.uniform(-1, 1, n),
                rng.uniform(0, 1, n),
            ]
        )
    return np.concatenate(cols, axis=1)


def _noise(rng, n, corr, sigma=0.02):
    """Noise realisation consistent with the correlation model."""
    eps = rng.normal(size=n + 2)
    if corr == "hanning":
        return np.convolve(eps, [0.25, 0.5, 0.25], mode="valid") / np.sqrt(0.375) * sigma
    return eps[:n] * sigma


@pytest.mark.parametrize(
    "corr,comps,k,k_min",
    [
        ("white", ["profile"], 2, 2),
        ("hanning", ["profile", "profile"], 0, 0),
        ("white", ["profile"], -1, -1),
        ("white", ["profile"], 3, 0),
        ("hanning", ["profile", "profile"], 3, -1),
    ],
)
def test_compiled_likelihood_matches_numpy(rng, monkeypatch, corr, comps, k, k_min):
    v = 1000.0 + 5.0 * np.arange(256)
    g = SpectralGrid.from_velocities(v)
    y = _noise(rng, 256, corr)
    good = np.ones(256, bool)
    good[[3, 4, 100]] = False
    nm = build_noise(y, good, None, corr)
    like = build_likelihood(y, g, nm, comps, k, 10.0, 0.1, k_min=k_min)
    th = _theta(rng, 32, len(comps))
    assert like._compiled
    fast = like.loglike(th)
    fast_k = like.loglike_k(th)
    m_fast, c_fast = like.moments(th)[1:]
    monkeypatch.setattr(fast_ll, "HAVE_NUMBA", False)
    monkeypatch.setattr(kernel, "USE_COMPILED", False)
    slow = like.loglike(th)
    # ln L ~ -5e5 for this noise-only data: the two code paths sum in different orders
    np.testing.assert_allclose(fast, slow, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(fast_k, like.loglike_k(th), rtol=1e-10, atol=1e-8)
    m_slow, c_slow = like.moments(th)[1:]
    # identical components give nearly collinear columns (condition number ~ 1e8)
    np.testing.assert_allclose(m_fast, m_slow, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(c_fast, c_slow, rtol=1e-7, atol=1e-12)


def test_compiled_conditional_matches_numpy(rng, monkeypatch):
    v = 1000.0 + 5.0 * np.arange(200)
    g = SpectralGrid.from_velocities(v)
    y = _noise(rng, 200, "hanning")
    nm = build_noise(y, np.ones(200, bool), None, "hanning")
    like = build_likelihood(y, g, nm, ["profile"], 1, 10.0, 0.1, k_min=0)
    th = _theta(rng, 8, 1)
    m1, c1 = like.conditional(th)
    monkeypatch.setattr(fast_ll, "HAVE_NUMBA", False)
    m2, c2 = like.conditional(th)
    np.testing.assert_allclose(m1, m2, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(c1, c2, rtol=1e-9, atol=1e-15)
