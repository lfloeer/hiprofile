import numpy as np
import pytest
from scipy import stats
from scipy.special import logsumexp

from hiprofile.likelihood.marginal import build_likelihood
from hiprofile.likelihood.noise import NoiseModel, correlation_from_spec
from hiprofile.model import SpectralGrid, profile

from conftest import requires_jax

TH = np.array([1300.0, 120.0, 10.0, 0.2, 0.3])  # v_ctr, delta_v, delta_v_rand, alpha, f


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


def _dense(like, theta, k=None):
    """ln N(y | M mu, C + M Lambda M^T) with dense matrices for the order k (default K_max)."""
    k = like.baseline_order if k is None else k
    m_cols = like.n_amp + k + 1
    idx = like.good_index
    m = like.design(theta[None])[0][:m_cols, idx].T
    lam = np.zeros((m_cols, m_cols))
    lam[: like.n_amp, : like.n_amp] = np.diag(like.amp_scale**2)
    lam[like.n_amp :, like.n_amp :] = like.baseline_cov[: k + 1, : k + 1]
    mu = np.r_[np.zeros(like.n_amp), like.baseline_mean[: k + 1]]
    cov = like.noise.covariance() + m @ lam @ m.T
    return stats.multivariate_normal(m @ mu, cov).logpdf(like.flux[idx])


@pytest.mark.parametrize("corr,mask,k", [("white", False, 0), ("white", True, 2), ("hanning", True, 1)])
def test_against_dense_formula(rng, corr, mask, k):
    g, y, nm = _setup(rng, corr, mask)
    like = build_likelihood(y, g, nm, ["profile"], k, 8.0, 0.1, baseline_mean=np.full(k + 1, 0.01))
    assert like.n_k == 1  # an informative baseline prior fixes the order
    np.testing.assert_allclose(like.loglike_gauss(TH[None])[0], _dense(like, TH), rtol=0, atol=1e-7)


def test_basis_column_combines_ring_and_disk(rng):
    g, y, nm = _setup(rng)
    like = build_likelihood(y, g, nm, ["profile"], 0, 8.0, 0.1)
    col = like.basis(TH[None])[0, 0]
    s_f = profile(g, [*TH[:4], 1.0, TH[4]])
    np.testing.assert_allclose(col, s_f, atol=1e-12)
    assert abs(col.sum() * g.dv - 1.0) < 1e-9


def test_k_marginal_equals_mixture_of_single_orders(rng):
    """The K-marginalised likelihood is the flat mixture of the single-K likelihoods."""
    g, y, nm = _setup(rng, "hanning", True)
    th = np.array([TH, [1310.0, 100.0, 20.0, -0.2, 0.8]])
    mix = build_likelihood(y, g, nm, ["profile"], 3, 8.0, 0.1, k_min=0)
    assert mix.k_values == [0, 1, 2, 3]
    singles = [build_likelihood(y, g, nm, ["profile"], k, 8.0, 0.1) for k in range(4)]
    ll_k = np.stack([s.loglike(th) for s in singles], axis=1)
    np.testing.assert_allclose(mix.loglike(th), logsumexp(ll_k, axis=1) - np.log(4), atol=1e-8)
    np.testing.assert_allclose(mix.loglike_k(th), ll_k, atol=1e-8)
    ll_g = np.stack([s.loglike_gauss(th) for s in singles], axis=1)
    np.testing.assert_allclose(mix.loglike_gauss(th), logsumexp(ll_g, axis=1) - np.log(4), atol=1e-8)
    for k in range(4):
        np.testing.assert_allclose(ll_g[0, k], _dense(mix, TH, k), atol=1e-7)
    pk = mix.k_posterior(th)
    np.testing.assert_allclose(pk.sum(axis=1), 1.0)
    np.testing.assert_allclose(pk, np.exp(ll_k - logsumexp(ll_k, axis=1, keepdims=True)), atol=1e-9)
    # conditional moments for a given order agree with the single-order likelihood
    for k in (0, 3):
        m_mix, c_mix = mix.conditional(th, k=k)
        m_one, c_one = singles[k].conditional(th)
        m = mix.n_amp + k + 1
        np.testing.assert_allclose(m_mix[:, :m], m_one, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(c_mix[:, :m, :m], c_one, rtol=1e-9, atol=1e-15)
        assert np.all(m_mix[:, m:] == 0) and np.all(c_mix[:, m:, :] == 0)


def test_null_model_per_order(rng):
    g, y, nm = _setup(rng)
    like = build_likelihood(y, g, nm, [], 2, 8.0, 0.1, k_min=0)
    lz = like.loglike_null_k()
    assert lz.shape == (3,) and np.all(np.isfinite(lz))
    for k in range(3):
        single = build_likelihood(y, g, nm, [], k, 8.0, 0.1)
        np.testing.assert_allclose(lz[k], single.loglike_null(), atol=1e-9)
    np.testing.assert_allclose(like.loglike_null(), logsumexp(lz) - np.log(3))


def test_two_components_and_rfi(rng):
    g, y, nm = _setup(rng)
    like = build_likelihood(y, g, nm, ["profile", "profile", "gauss"], 1, 8.0, 0.1)
    th = np.array([1250.0, 60.0, 8.0, 0.0, 0.5, 1350.0, 80.0, 12.0, 0.4, 0.1, 1400.0, 3.0])
    np.testing.assert_allclose(like.loglike_gauss(th[None])[0], _dense(like, th), atol=1e-7)


def test_against_numerical_integration_over_beta(rng):
    """2-D Gauss-Hermite integral over (S, c_0) equals the closed form."""
    g, y, nm = _setup(rng, n=60)
    like = build_likelihood(y, g, nm, ["profile"], 0, 3.0, 0.2)
    th = np.array([1150.0, 100.0, 10.0, 0.2, 0.4])
    mean, cov = like.conditional(th[None])
    mean, cov = mean[0], cov[0]
    chol = np.linalg.cholesky(cov * 4.0)  # integrate with a wider Gaussian kernel
    x, w = np.polynomial.hermite_e.hermegauss(60)
    idx = like.good_index
    design = like.design(th[None])[0][:, idx]
    prior = stats.multivariate_normal(np.zeros(2), np.diag([9.0, 0.04]))
    vals, wts = [], []
    for i, a in enumerate(x):
        for j, b in enumerate(x):
            beta = mean + chol @ np.array([a, b])
            ll = stats.norm(beta @ design, 0.05).logpdf(y[idx]).sum() + prior.logpdf(beta)
            q = stats.multivariate_normal(mean, cov * 4.0).logpdf(beta)
            vals.append(ll - q)
            wts.append(np.log(w[i] * w[j]) - np.log(2 * np.pi))
    est = np.logaddexp.reduce(np.array(vals) + np.array(wts))
    np.testing.assert_allclose(est, like.loglike_gauss(th[None])[0], atol=1e-6)


def test_conditional_matches_dense(rng):
    g, y, nm = _setup(rng, "hanning")
    like = build_likelihood(y, g, nm, ["profile"], 1, 8.0, 0.1)
    mean, cov = like.conditional(TH[None])
    idx = like.good_index
    m = like.design(TH[None])[0][:, idx].T
    cinv = np.linalg.inv(nm.covariance())
    prec = m.T @ cinv @ m + np.diag(np.r_[1 / 64.0, 1 / 0.01, 1 / 0.01])
    np.testing.assert_allclose(cov[0], np.linalg.inv(prec), rtol=1e-8)
    np.testing.assert_allclose(mean[0], np.linalg.solve(prec, m.T @ cinv @ y[idx]), rtol=1e-8)


def test_noise_scale_parameter(rng):
    g, y, nm = _setup(rng)
    free = build_likelihood(y, g, nm, ["profile"], 1, 8.0, 0.1, noise_scale=True, k_min=0)
    fixed = build_likelihood(y, g, nm.scaled(np.full(nm.n, 10**0.1)), ["profile"], 1, 8.0, 0.1, k_min=0)
    np.testing.assert_allclose(free.loglike(np.r_[TH, 0.1][None]), fixed.loglike(TH[None]), atol=1e-8)


@requires_jax
def test_jax_backend_matches_numpy(rng):
    g, y, nm = _setup(rng, "hanning", True)
    th = np.array([TH, [1310.0, 100.0, 20.0, -0.2, 0.9]])
    a = build_likelihood(y, g, nm, ["profile"], 2, 8.0, 0.1, k_min=0).loglike(th)
    b = np.asarray(build_likelihood(y, g, nm, ["profile"], 2, 8.0, 0.1, k_min=0, backend="jax").loglike(th))
    np.testing.assert_allclose(a, b, atol=1e-8)


def test_positivity_against_integral_over_half_normal_prior(rng):
    """ln p(y | theta) with a half-normal flux prior equals a direct 1-D integral."""
    g, y, nm = _setup(rng, n=80)
    y = y - 0.9 * profile(g, [1300.0, 120.0, 10.0, 0.2, 5.0, 0.3])  # weak line, S near 0
    like = build_likelihood(y, g, nm, ["profile"], -1, 3.0, 0.1)
    th = np.array([1150.0, 60.0, 12.0, -0.3, 0.2])
    design = like.design(th[None])[0][0, like.good_index]
    yy = y[like.good_index]
    s_grid = np.linspace(0, 15, 30001)
    ll = stats.norm(s_grid[:, None] * design[None], 0.05).logpdf(yy).sum(axis=1)
    f = ll + stats.halfnorm(scale=3.0).logpdf(s_grid)
    est = np.log(np.trapezoid(np.exp(f - f.max()), s_grid)) + f.max()
    np.testing.assert_allclose(like.loglike(th[None])[0], est, atol=1e-4)
    # the Gaussian version differs (it allows negative fluxes)
    assert abs(like.loglike_gauss(th[None])[0] - est) > 0.1


def test_two_component_positivity_is_exact(rng):
    """For two amplitudes the bivariate orthant makes the target exact: the MC factor is 1."""
    from hiprofile.inference.amplitudes import draw_amplitudes

    g, y, nm = _setup(rng)
    like = build_likelihood(y, g, nm, ["profile", "profile"], 1, 8.0, 0.1, k_min=0)
    th = np.array([[1250.0, 60.0, 8.0, 0.0, 0.5, 1350.0, 80.0, 12.0, 0.4, 0.1]] * 3)
    d = draw_amplitudes(like, th, rng)
    assert np.all(d.factor == 1.0) and d.k.shape == (3,)
    assert np.all(d.beta[:, :2] >= 0)
