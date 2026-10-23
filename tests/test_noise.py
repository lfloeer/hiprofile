import numpy as np

from hiprofile.likelihood.noise import NoiseModel, build_noise, correlation_from_spec, estimate_sigma


def test_hanning_correlation():
    np.testing.assert_allclose(correlation_from_spec("hanning"), [1, 2 / 3, 1 / 6])
    np.testing.assert_allclose(correlation_from_spec("hanning-decimated"), [1, 1 / 6])


def test_whitening_inverts_covariance(rng):
    idx = np.delete(np.arange(60), [5, 6, 30])
    nm = NoiseModel(sigma=rng.uniform(0.5, 2, idx.size), good_index=idx, rho=correlation_from_spec("hanning"))
    c = nm.covariance()
    w = nm.dense_whitener()
    np.testing.assert_allclose(w @ c @ w.T, np.eye(idx.size), atol=1e-8)
    x = rng.normal(size=(3, idx.size))
    np.testing.assert_allclose(nm.whiten(x), x @ w.T, atol=1e-9)
    np.testing.assert_allclose(nm.logdet, np.linalg.slogdet(c)[1], atol=1e-8)


def test_sigma_estimate_hanning(rng):
    raw = rng.normal(size=20002)
    y = np.convolve(raw, [0.25, 0.5, 0.25], mode="valid") / np.sqrt(0.375) * 0.3
    s = estimate_sigma(y, np.ones(y.size, bool), correlation_from_spec("hanning"))
    assert abs(s / 0.3 - 1) < 0.03
    nm = build_noise(y, np.ones(y.size, bool), None, "hanning")
    assert nm.bandwidth == 2
