import numpy as np
from scipy import special

from conftest import requires_jax
from hiprofile.model.bessel import bessel_terms_numpy


def _reference(x):
    j0, j1 = special.j0(x), special.j1(x)
    xs = np.where(x > 1e-3, x, 1.0)
    j1x = np.where(x > 1e-3, j1 / xs, 0.5 - x**2 / 16)
    j2x = np.where(x > 1e-3, special.jv(2, x) / xs, x / 8 - x**3 / 96)
    return j0, j1, j1x, j2x


X = np.concatenate([np.linspace(0, 60, 60001), np.geomspace(1e-9, 1e4, 4000)])


def test_numpy_terms():
    for a, b in zip(bessel_terms_numpy(X), _reference(X), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)


@requires_jax
def test_jax_terms_match_scipy():
    import jax.numpy as jnp

    from hiprofile._backend import get_xp
    from hiprofile.model.bessel import bessel_terms_jax

    get_xp("jax")
    for a, b in zip(bessel_terms_jax(jnp.asarray(X)), _reference(X), strict=True):
        np.testing.assert_allclose(np.asarray(a), b, atol=5e-12, rtol=0)


@requires_jax
def test_jax_gradients_finite_at_switch_and_zero():
    import jax

    from hiprofile._backend import get_xp
    from hiprofile.model.bessel import X_SWITCH, bessel_terms_jax

    get_xp("jax")
    for x0 in (0.0, X_SWITCH, 3.0, 40.0):
        for i in range(4):
            g = jax.grad(lambda t, i=i: bessel_terms_jax(t)[i])(x0)
            assert np.isfinite(float(g))
    g = jax.grad(lambda t: bessel_terms_jax(t)[0])(5.0)
    np.testing.assert_allclose(float(g), -special.j1(5.0), atol=1e-11)
