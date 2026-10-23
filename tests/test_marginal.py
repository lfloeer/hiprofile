import itertools

import numpy as np
import pytest
from scipy import stats

from conftest import requires_jax
from hiprofile.likelihood.marginal import build_likelihood
from hiprofile.likelihood.noise import NoiseModel, correlation_from_spec
from hiprofile.model import SpectralGrid, profile


def _setup(rng, corr="white", mask=False, n=120):
    v = 1000.0 + 5.0 * np.arange(n)
    g = SpectralGrid.from_velocities(v)
    eps = rng.normal(size=n + 2)
    if corr == "hanning":
        noise = np.convolve(eps, [0.25, 0.5, 0.25], mode="valid") / np.sqrt(0.375)
    else:
        noise = eps[:n]
    y = profile(g, [1300.0, 120.0, 10.0, 0.2, 5.0, 0.3]) + 0.05 * noise
    good = np.ones(n, bool)
    if mask:
        good[[10, 11, 50]] = False
    idx = np.flatnonzero(good)
    nm = NoiseModel(sigma=np.full(idx.size, 0.05), good_index=idx, rho=correlation_from_spec(corr))
    return g, y, nm


def _dense(like, theta):
    """ln N(y | M mu, C + M Lambda M^T) with dense matrices."""
    idx = like.good_index
    m = like.design(theta[None])[0][:, idx].T
    lam = np.zeros((like.n_linear, like.n_linear))
    lam[: like.n_amp, : like.n_amp] = np.diag(like.amp_scale**2)
    lam[like.n_amp :, like.n_amp :] = like.baseline_cov
    mu = np.r_[np.zeros(like.n_amp), like.baseline_mean]
    cov = like.noise.covariance() + m @ lam @ m.T
    return stats.multivariate_normal(m @ mu, cov).logpdf(like.flux[idx])


@pytest.mark.parametrize("corr,mask,k", [("white", False, 0), ("white", True, 2), ("hanning", True, 1)])
def test_against_dense_formula(rng, corr, mask, k):
    g, y, nm = _setup(rng, corr, mask)
    like = build_likelihood(y, g, nm, ["profile"], k, 8.0, 0.1, baseline_mean=np.full(k + 1, 0.01))
    th = np.array([1300.0, 120.0, 10.0, 0.2])
    np.testing.assert_allclose(like.loglike(th[None])[0], _dense(like, th), rtol=0, atol=1e-7)


def test_two_components_and_rfi(rng):
    g, y, nm = _setup(rng)
    like = build_likelihood(y, g, nm, ["profile", "profile", "gauss"], 1, 8.0, 0.1)
    th = np.array([1250.0, 60.0, 8.0, 0.0, 1350.0, 80.0, 12.0, 0.4, 1400.0, 3.0])
    np.testing.assert_allclose(like.loglike(th[None])[0], _dense(like, th), atol=1e-7)


def test_against_numerical_integration_over_beta(rng):
    """3-D Gauss-Hermite integral over (a_ring, a_disk, c_0) equals the closed form."""
    g, y, nm = _setup(rng, n=60)
    like = build_likelihood(y, g, nm, ["profile"], 0, 3.0, 0.2)
    th = np.array([1150.0, 100.0, 10.0, 0.2])
    mean, cov = like.conditional(th[None])
    mean, cov = mean[0], cov[0]
    chol = np.linalg.cholesky(cov * 4.0)  # integrate with a wider Gaussian kernel
    x, w = np.polynomial.hermite_e.hermegauss(40)
    idx = like.good_index
    design = like.design(th[None])[0][:, idx]
    prior = stats.multivariate_normal(np.zeros(3), np.diag([9.0, 9.0, 0.04]))
    vals, wts = [], []
    for (i, a), (j, b), (k, c) in itertools.product(enumerate(x), enumerate(x), enumerate(x)):
        z = np.array([a, b, c])
        beta = mean + chol @ z
        ll = stats.norm(beta @ design, 0.05).logpdf(y[idx]).sum() + prior.logpdf(beta)
        q = stats.multivariate_normal(mean, cov * 4.0).logpdf(beta)
        vals.append(ll - q)
        wts.append(np.log(w[i] * w[j] * w[k]) - 1.5 * np.log(2 * np.pi))
    est = np.logaddexp.reduce(np.array(vals) + np.array(wts))
    np.testing.assert_allclose(est, like.loglike(th[None])[0], atol=1e-6)


def test_conditional_matches_dense(rng):
    g, y, nm = _setup(rng, "hanning")
    like = build_likelihood(y, g, nm, ["profile"], 1, 8.0, 0.1)
    th = np.array([1300.0, 120.0, 10.0, 0.2])
    mean, cov = like.conditional(th[None])
    idx = like.good_index
    m = like.design(th[None])[0][:, idx].T
    cinv = np.linalg.inv(nm.covariance())
    prec = m.T @ cinv @ m + np.diag(np.r_[1 / 64.0, 1 / 64.0, 1 / 0.01, 1 / 0.01])
    np.testing.assert_allclose(cov[0], np.linalg.inv(prec), rtol=1e-8)
    np.testing.assert_allclose(mean[0], np.linalg.solve(prec, m.T @ cinv @ y[idx]), rtol=1e-8)


def test_noise_scale_parameter(rng):
    g, y, nm = _setup(rng)
    th = np.array([1300.0, 120.0, 10.0, 0.2])
    free = build_likelihood(y, g, nm, ["profile"], 1, 8.0, 0.1, noise_scale=True)
    fixed = build_likelihood(y, g, nm.scaled(np.full(nm.n, 10**0.1)), ["profile"], 1, 8.0, 0.1)
    np.testing.assert_allclose(free.loglike(np.r_[th, 0.1][None]), fixed.loglike(th[None]), atol=1e-8)


def test_null_model(rng):
    g, y, nm = _setup(rng)
    like = build_likelihood(y, g, nm, [], 2, 8.0, 0.1)
    assert np.isfinite(like.loglike_null())


@requires_jax
def test_jax_backend_matches_numpy(rng):
    g, y, nm = _setup(rng, "hanning", True)
    th = np.array([[1300.0, 120.0, 10.0, 0.2], [1310.0, 100.0, 20.0, -0.2]])
    a = build_likelihood(y, g, nm, ["profile"], 1, 8.0, 0.1).loglike(th)
    b = np.asarray(build_likelihood(y, g, nm, ["profile"], 1, 8.0, 0.1, backend="jax").loglike(th))
    np.testing.assert_allclose(a, b, atol=1e-8)
