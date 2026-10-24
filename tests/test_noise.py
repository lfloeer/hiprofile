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


def test_white_fraction_is_recovered(rng):
    from hiprofile.likelihood.noise import estimate_white_fraction

    rho = correlation_from_spec("hanning")
    good = np.ones(20000, bool)
    eps_white = rng.normal(size=20002)
    hann = np.convolve(eps_white, [0.25, 0.5, 0.25], mode="valid") / np.sqrt(0.375)
    for eps in (0.0, 0.05, 0.3, 1.0):
        y = np.sqrt(1 - eps) * hann + np.sqrt(eps) * rng.normal(size=hann.size)
        est = estimate_white_fraction(y, good, rho)
        assert abs(est - max(eps, 0.005)) < 0.01 + 0.1 * eps, (eps, est)


def test_hanning_plus_white_whitens_real_like_noise(rng):
    """Hanning plus 3 % white noise: exact Hanning whitening explodes (chi2/n ~ 1e4), hanning+white not."""
    n = 400
    e = rng.normal(size=n + 2)
    y = np.sqrt(0.97) * np.convolve(e, [0.25, 0.5, 0.25], mode="valid") / np.sqrt(0.375)
    y += np.sqrt(0.03) * rng.normal(size=n)
    good = np.ones(n, bool)
    chi2 = {}
    for spec in ("hanning", "hanning+white"):
        nm = build_noise(y, good, None, spec)
        chi2[spec] = float(np.sum(nm.whiten(y) ** 2) / n)
    assert chi2["hanning"] > 3.0
    assert 0.6 < chi2["hanning+white"] < 1.7  # 5-95 % range for 400 channels: 0.8-1.6


def _smoothed_noise(rng, n, kernel):
    k = np.asarray(kernel, float) / np.sum(kernel)
    y = np.convolve(rng.normal(size=n + k.size - 1), k, mode="valid")
    return y / np.sqrt(np.sum(k**2))


def test_estimated_correlation_recovers_smoothing_kernels(rng):
    from hiprofile.likelihood.noise import estimate_correlation, kernel_correlation

    gauss = np.exp(-0.5 * (np.arange(-3, 4) / 0.8) ** 2)
    for kernel in ([1.0], [0.25, 0.5, 0.25], gauss, np.convolve([0.5, 0.5], [0.25, 0.5, 0.25])):
        true = np.pad(kernel_correlation(np.asarray(kernel) / np.sum(kernel)), (0, 8))[:4]
        good = rng.uniform(size=20000) > 0.05  # masked channels leave gaps
        est = np.pad(estimate_correlation(_smoothed_noise(rng, 20000, kernel), good), (0, 8))[:4]
        # differences pin the high-frequency shape; rho_2 of a 7-tap kernel only to ~0.03
        np.testing.assert_allclose(est, true, atol=0.05)


def test_estimated_correlation_is_positive_definite(rng):
    from hiprofile.likelihood.noise import correlation_spectrum, estimate_correlation, make_positive

    for _ in range(20):
        y = _smoothed_noise(rng, 120, [0.25, 0.5, 0.25])
        rho = estimate_correlation(y, rng.uniform(size=120) > 0.2)
        assert correlation_spectrum(rho).min() >= 0.005 - 1e-9
    rho = make_positive(np.array([1.0, 0.9]))  # not positive definite on its own
    assert correlation_spectrum(rho).min() >= 0.02 - 1e-9


def test_auto_correlation_follows_the_response_kernel():
    from hiprofile.config import Config
    from hiprofile.inference.fit import _correlation

    rho, white = _correlation(Config().updated(model={"response": "boxcar+kernel(0.6, 0.2)"}))
    np.testing.assert_allclose(rho[:3], [1.0, 0.24 / 0.44, 0.04 / 0.44])
    assert white
    assert _correlation(Config()) == ("white", False)
