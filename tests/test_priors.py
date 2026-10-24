import numpy as np

from hiprofile.config import Priors, Range
from hiprofile.inference.priors import build_prior, ordered_uniform


def test_round_trip_and_ordering(rng):
    v = np.arange(512) * 5.0
    for par in ("direct", "effective"):
        p = build_prior(("profile", "profile", "gauss"), Priors(), v, par)
        u = rng.uniform(size=(2000, p.ndim))
        t = p(u)
        np.testing.assert_allclose(p.inverse(t), u, atol=1e-9)
        assert np.all(t[:, 0] < t[:, 4])
        assert np.all(t[:, 2] >= 4.0 - 1e-9)


def test_order_statistics_are_uniform(rng):
    x = ordered_uniform(rng.uniform(size=(200000, 3)))
    np.testing.assert_allclose(x.mean(axis=0), [0.25, 0.5, 0.75], atol=3e-3)


def test_normal_centre_prior():
    v = np.arange(100) * 5.0
    pr = Priors(v_ctr=Range(kind="normal", mean=250.0, std=20.0))
    p = build_prior(("profile",), pr, v)
    t = p(np.full((1, 4), 0.5))
    assert abs(t[0, 0] - 250.0) < 1e-9


def test_default_shapes_follow_things(rng):
    """Defaults: Delta v_rand ~ 12.6 km/s, alpha ~ N(0, 0.25), Delta v ~ 160 km/s with a narrow tail."""
    v = np.arange(1000) * 5.0
    p = build_prior(("profile",), Priors(), v)
    t = p(rng.uniform(size=(200000, 4)))
    dv, sig, a = t[:, 1], t[:, 2], t[:, 3]
    assert abs(np.median(sig) - 12.6) < 0.3 and sig.min() >= 4.0 and sig.max() <= 60.0
    assert abs(np.std(a) - 0.25) < 0.01 and np.all(np.abs(a) <= 1)
    assert 0.0 <= dv.min() and dv.max() <= 800.0
    assert abs(np.mean(dv < 60) - (0.1 + 0.9 * 0.114)) < 0.01  # uniform part + log-normal tail
    np.testing.assert_allclose(p.inverse(t[:1000]), p.inverse(p(p.inverse(t[:1000]))), atol=1e-6)


def test_truncated_lognormal_round_trip(rng):
    from hiprofile.inference.priors import Bounds1D

    b = Bounds1D("lognormal", 4.0, 60.0, mean=1.1, std=0.2)
    u = rng.uniform(size=1000)
    np.testing.assert_allclose(b.inverse(b.transform(u)), u, atol=1e-10)
    m = Bounds1D("lognormal", 0.0, 800.0, mean=2.2, std=0.35, mix_weight=0.1, mix_high=60.0)
    np.testing.assert_allclose(m.inverse(m.transform(u)), u, atol=1e-4)
