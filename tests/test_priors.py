import numpy as np

from hiprofile.config import Priors, Range
from hiprofile.inference.priors import build_prior, ordered_uniform


def test_round_trip_and_ordering(rng):
    v = np.arange(512) * 5.0
    for _ in range(1):
        p = build_prior(("profile", "profile", "gauss"), Priors(), v)
        assert p.names == [
            "v_ctr_1", "delta_v_1", "delta_v_rand_1", "alpha_1", "f_1",
            "v_ctr_2", "delta_v_2", "delta_v_rand_2", "alpha_2", "f_2",
            "rfi_v_1", "rfi_sigma_1",
        ]  # fmt: skip
        u = rng.uniform(size=(2000, p.ndim))
        t = p(u)
        np.testing.assert_allclose(p.inverse(t), u, atol=1e-9)
        r = p.band_position(t)
        assert np.all(r[:, 0] < r[:, 1])  # labels ordered by the relative position in the band
        assert np.all(t[:, 2] >= 4.0 - 1e-9)
        assert np.all((t[:, 4] >= 0) & (t[:, 4] <= 1))


def test_order_statistics_are_uniform(rng):
    x = ordered_uniform(rng.uniform(size=(200000, 3)))
    np.testing.assert_allclose(x.mean(axis=0), [0.25, 0.5, 0.75], atol=3e-3)


def test_normal_centre_prior():
    v = np.arange(100) * 5.0
    pr = Priors(v_ctr=Range(kind="normal", mean=250.0, std=20.0))
    p = build_prior(("profile",), pr, v)
    t = p(np.full((1, 5), 0.5))
    assert abs(t[0, 0] - 250.0) < 1e-9
    assert abs(t[0, 4] - (1 - 0.5 ** (1 / 3))) < 1e-12  # median of Beta(1, 3)


def test_default_shapes_follow_things(rng):
    """Defaults: Delta v ~ Gamma(2.3, 83 km/s), Delta v_rand ~ 12.6 km/s, alpha ~ N(0, 0.25), f ~ B(1, 3)."""
    from scipy import stats

    v = np.arange(1000) * 5.0
    p = build_prior(("profile",), Priors(), v)
    t = p(rng.uniform(size=(200000, 5)))
    dv, sig, a, f = t[:, 1], t[:, 2], t[:, 3], t[:, 4]
    # median of Gamma(2.3, scale 83 km/s) truncated at 800 km/s (ALFALFA calibration)
    g = stats.gamma(2.3, scale=83.0)
    assert abs(np.median(dv) - g.ppf(0.5 * g.cdf(800.0))) < 3 and dv.max() <= 800.0
    assert abs(np.median(sig) - 12.6) < 0.3 and sig.min() >= 4.0 and sig.max() <= 60.0
    assert abs(np.std(a) - 0.25) < 0.01 and np.all(np.abs(a) <= 1)
    assert abs(np.mean(f) - 0.25) < 0.003 and abs(np.median(f) - (1 - 0.5 ** (1 / 3))) < 0.003
    assert abs(np.mean(f > 0.7) - 0.3**3) < 0.002 and f.min() >= 0 and f.max() <= 1


def test_edges_exclude_definitional_bounds_of_f():
    p = build_prior(("profile",), Priors(), np.arange(100) * 5.0)
    edges = p.edge_bounds()
    assert "f_1" not in edges
    assert edges["delta_v_rand_1"] == (4.0, 60.0) and edges["alpha_1"] == (-1.0, 1.0)


def test_truncated_round_trip(rng):
    from hiprofile.inference.priors import Bounds1D

    u = rng.uniform(size=1000)
    for b in (
        Bounds1D("lognormal", 4.0, 60.0, mean=1.1, std=0.2),
        Bounds1D("normal", -1.0, 1.0, std=0.25),
        Bounds1D("beta", 0.0, 1.0, a=1.0, b=3.0),
        Bounds1D("beta", 0.0, 1.0, a=0.55, b=1.86),
        Bounds1D("gamma", 0.0, 800.0, a=2.3, scale=83.0),
        Bounds1D("gamma", 20.0, np.inf, a=0.8, scale=50.0),
    ):
        np.testing.assert_allclose(b.inverse(b.transform(u)), u, atol=1e-10)


def test_components_lie_inside_the_band(rng):
    """Containment: v_ctr +- (delta_v / 2 + 3 delta_v_rand) inside the band minus the margins; the
    widths keep their prior where they fit, and the labels may be permuted on input."""
    from hiprofile.inference.priors import CONTAIN_SIGMA

    for n_chan in (512, 60):
        v = np.arange(n_chan) * 5.0
        p = build_prior(("profile", "profile"), Priors(), v)
        lo, hi = p.band
        assert lo == 0.05 * v[-1] and hi == 0.95 * v[-1]
        u = rng.uniform(size=(20000, p.ndim))
        t = p(u)
        np.testing.assert_allclose(p.inverse(t), u, atol=1e-9)
        swapped = np.c_[t[:, 5:], t[:, :5]]
        np.testing.assert_allclose(p(p.inverse(swapped)), t, atol=1e-8)
        for c in (0, 5):
            h = 0.5 * t[:, c + 1] + CONTAIN_SIGMA * t[:, c + 2]
            assert np.all(t[:, c] - h >= lo - 1e-6) and np.all(t[:, c] + h <= hi + 1e-6)
    # a wide band leaves the width prior untouched
    p = build_prior(("profile",), Priors(), np.arange(1000) * 5.0)
    free = build_prior(("profile",), Priors(contain=False), np.arange(1000) * 5.0)
    u = rng.uniform(size=(1000, 5))
    np.testing.assert_allclose(p(u)[:, 1:], free(u)[:, 1:], atol=1e-9)
    # an explicit centre range is kept, and a normal centre prior switches containment off
    v = np.arange(512) * 5.0
    p = build_prior(("profile",), Priors(v_ctr=Range(kind="uniform", low=1000, high=1100)), v)
    t = p(rng.uniform(size=(5000, 5)))
    assert t[:, 0].min() >= 1000 and t[:, 0].max() <= 1100 and np.median(t[:, 1]) > 140
    assert build_prior(("profile",), Priors(v_ctr=Range(kind="normal", mean=1e3, std=30)), v).band is None
