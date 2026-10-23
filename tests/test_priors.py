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
