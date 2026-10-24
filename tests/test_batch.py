import numpy as np

from hiprofile import Spectrum
from hiprofile.config import Config
from hiprofile.model import SpectralGrid, profile

from conftest import requires_jax


@requires_jax
def test_batched_likelihood_and_screen_match_single_spectrum():
    from hiprofile.inference.batch import BatchedLikelihood, screen_batch
    from hiprofile.inference.fit import prepare

    rng = np.random.default_rng(0)
    cfg = Config().updated(rfi={"prefit": False})
    specs = []
    for i, s in enumerate([0.0, 20.0]):
        v = 1000 + 5.0 * np.arange(200) + 7 * i
        y = profile(SpectralGrid.from_velocities(v), [1500, 150, 10, 0.1, s, 0.3]) + 0.02 * rng.normal(
            size=200
        )
        specs.append(Spectrum(v, y, sigma=np.full(200, 0.02), id=f"s{i}"))
    ctxs = [prepare(sp, cfg, 1) for sp in specs]
    g0 = ctxs[0].grid
    like = BatchedLikelihood(
        grid=g0,
        v_shift=np.array([c.grid.v0 - g0.v0 for c in ctxs]),
        flux=np.array([c.spectrum.flux for c in ctxs]),
        weight=np.array([1 / np.full(200, 0.02)] * 2),
        baseline_order=1,
        amp_scale=np.array([c.amp_scale for c in ctxs]),
        baseline_scale=np.array([c.baseline_scale for c in ctxs]),
    )
    th = np.array([[[1500.0, 150, 10, 0.1, 0.3], [1480.0, 100, 20, -0.3, 0.8]]] * 2)
    ll = like(th)
    cfg1 = cfg.updated(model={"k_min": 1, "k_max": 1})
    ctxs1 = [prepare(sp, cfg1, 1) for sp in specs]
    for i, c in enumerate(ctxs1):
        np.testing.assert_allclose(ll[i], c.likelihood(1).loglike(th[i]), atol=1e-7)
        np.testing.assert_allclose(like.log_z_null[i], c.likelihood(0).loglike_null(), atol=1e-7)
    out = screen_batch(specs, cfg, k=1, n_particles=400)
    assert out[0]["p_detect"] < 0.5 < out[1]["p_detect"]
