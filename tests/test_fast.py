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


@pytest.mark.parametrize(
    "corr,comps,k",
    [("white", ["profile"], 2), ("hanning", ["profile", "profile"], 0), ("white", ["profile"], -1)],
)
def test_compiled_likelihood_matches_numpy(rng, monkeypatch, corr, comps, k):
    v = 1000.0 + 5.0 * np.arange(256)
    g = SpectralGrid.from_velocities(v)
    y = rng.normal(size=256) * 0.02
    good = np.ones(256, bool)
    good[[3, 4, 100]] = False
    nm = build_noise(y, good, None, corr)
    like = build_likelihood(y, g, nm, comps, k, 10.0, 0.1)
    th = np.c_[
        rng.uniform(1200, 2000, 32), rng.uniform(0, 400, 32), rng.uniform(5, 40, 32), rng.uniform(-1, 1, 32)
    ]
    th = np.tile(th, (1, len(comps)))
    assert like._compiled
    fast = like.loglike(th)
    monkeypatch.setattr(fast_ll, "HAVE_NUMBA", False)
    monkeypatch.setattr(kernel, "USE_COMPILED", False)
    slow = like.loglike(th)
    np.testing.assert_allclose(fast, slow, rtol=1e-11, atol=1e-8)


def test_compiled_conditional_matches_numpy(rng, monkeypatch):
    v = 1000.0 + 5.0 * np.arange(200)
    g = SpectralGrid.from_velocities(v)
    y = rng.normal(size=200) * 0.02
    nm = build_noise(y, np.ones(200, bool), None, "hanning")
    like = build_likelihood(y, g, nm, ["profile"], 1, 10.0, 0.1)
    th = np.c_[
        rng.uniform(1200, 1800, 8), rng.uniform(0, 400, 8), rng.uniform(5, 40, 8), rng.uniform(-1, 1, 8)
    ]
    m1, c1 = like.conditional(th)
    monkeypatch.setattr(fast_ll, "HAVE_NUMBA", False)
    m2, c2 = like.conditional(th)
    np.testing.assert_allclose(m1, m2, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(c1, c2, rtol=1e-9, atol=1e-15)
