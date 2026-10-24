import numpy as np
import pytest

from hiprofile.model import SpectralGrid, gaussian_basis, profile, profile_basis
from hiprofile.model.kernel import _profile_fourier_numpy, default_supersample, next_fast_len

from conftest import requires_jax


def brute_force(v, theta, s, f, dv, sub=400):
    """Analytic ring/disk shape x asymmetry, convolved with a Gaussian, channel-averaged.

    The shapes are integrated exactly over each fine cell via their CDFs (in
    u = 2 (v - v_ctr) / dv_width), which handles the 1/sqrt singularity of the ring.
    """
    vc, width, sig, alpha = theta
    h = dv / sub
    edges = np.arange(v[0] - dv / 2 - 10 * sig, v[-1] + dv / 2 + 10 * sig + h, h)
    u = np.clip((edges - vc) / (width / 2), -1, 1)
    rho = np.sqrt(1 - u**2)
    cdf_ring = (np.arcsin(u) - alpha * rho) / np.pi
    cdf_disk = (u * rho + np.arcsin(u)) / np.pi - 2 * alpha * rho**3 / (3 * np.pi)
    mass = s * ((1 - f) * np.diff(cdf_ring) + f * np.diff(cdf_disk))
    x = 0.5 * (edges[1:] + edges[:-1])
    kx = np.arange(-8 * sig, 8 * sig + h / 2, h)
    kern = np.exp(-0.5 * (kx / sig) ** 2)
    kern /= kern.sum()
    conv = np.convolve(mass / h, kern, mode="same")
    out = []
    for vi in v:
        m = (x >= vi - dv / 2) & (x < vi + dv / 2)
        out.append(conv[m].mean())
    return np.array(out)


@pytest.mark.parametrize("theta,f", [((1500.0, 200.0, 15.0, 0.3), 0.2), ((1500.0, 80.0, 25.0, -0.6), 0.8)])
def test_against_brute_force(theta, f):
    v = 1000.0 + 5.0 * np.arange(200)
    g = SpectralGrid.from_velocities(v, supersample=8)
    ours = profile(g, np.r_[theta, 10.0, f])
    ref = brute_force(v, theta, 10.0, f, 5.0)
    assert np.max(np.abs(ours - ref)) < 1e-4 * ours.max()


def test_unit_flux_and_asymmetry(grid):
    th = np.array([[2000.0, 200.0, 10.0, 0.5]])
    b = profile_basis(grid, th)[0]
    np.testing.assert_allclose(b.sum(axis=1) * grid.dv, 1.0, atol=1e-9)
    v = grid.velocity
    for row in b:
        assert row[v > 2000].sum() > row[v < 2000].sum()  # alpha > 0: more flux at high v
    g = gaussian_basis(grid, np.array([[2000.0, 3.0]]))[0, 0]
    assert abs(g.sum() * grid.dv - 1) < 1e-9


def test_hanning_response_equals_channel_smoothing(grid):
    th = np.array([2100.0, 150.0, 8.0, -0.2])
    box = profile_basis(grid, th)
    han = profile_basis(grid.with_response("boxcar+hanning"), th)
    smooth = np.apply_along_axis(lambda r: np.convolve(r, [0.25, 0.5, 0.25], mode="same"), -1, box)
    np.testing.assert_allclose(han[:, 1:-1], smooth[:, 1:-1], atol=1e-10)


def test_dense_and_folded_paths_agree(grid, rng):
    th = np.c_[
        rng.uniform(1200, 2800, 50),
        rng.uniform(0, 500, 50),
        np.exp(rng.uniform(1.4, 4, 50)),
        rng.uniform(-1, 1, 50),
    ]
    folded = profile_basis(grid, th)
    coeffs = _profile_fourier_numpy(grid, th)
    dense = (
        np.fft.irfft(coeffs, n=grid.n_fft, axis=-1)[..., : grid.n_chan * grid.supersample : grid.supersample]
        / grid.h
    )
    np.testing.assert_allclose(folded, dense, atol=1e-12)


def test_batch_shapes(grid):
    th = np.tile([2000.0, 200.0, 10.0, 0.0], (3, 2, 1))
    assert profile_basis(grid, th).shape == (3, 2, 2, grid.n_chan)


@requires_jax
def test_jax_matches_numpy_and_is_differentiable(grid, rng):
    import jax
    import jax.numpy as jnp

    th = np.c_[
        rng.uniform(1200, 2800, 20),
        rng.uniform(0, 500, 20),
        np.exp(rng.uniform(1.4, 4, 20)),
        rng.uniform(-1, 1, 20),
    ]
    np.testing.assert_allclose(
        np.asarray(profile_basis(grid, th, "jax")), profile_basis(grid, th), atol=1e-12
    )
    g = jax.grad(lambda t: jnp.sum(profile_basis(grid, t, "jax") ** 2))(jnp.asarray(th[0]))
    assert np.all(np.isfinite(np.asarray(g)))


def test_helpers():
    assert next_fast_len(1000) == 1000
    assert next_fast_len(1001) == 1024
    assert default_supersample(5.0, 4.0) >= 2


def test_kernel_response_equals_hanning_and_explicit_smoothing():
    from hiprofile.model import SpectralGrid, profile
    from hiprofile.model.response import noise_kernel

    v = 1000 + 5.0 * np.arange(300)
    th = [1500.0, 150.0, 10.0, 0.2, 5.0, 0.3]
    a = profile(SpectralGrid.from_velocities(v, response="boxcar+hanning"), th)
    b = profile(SpectralGrid.from_velocities(v, response="boxcar+kernel(0.5, 0.25)"), th)
    np.testing.assert_allclose(a, b, atol=1e-14)
    w = [0.1, 0.2, 0.4, 0.2, 0.1]
    c = profile(SpectralGrid.from_velocities(v, response="boxcar+kernel(0.4, 0.2, 0.1)"), th)
    d = np.convolve(profile(SpectralGrid.from_velocities(v, response="boxcar"), th), w, mode="same")
    np.testing.assert_allclose(c[5:-5], d[5:-5], atol=1e-12)
    np.testing.assert_allclose(noise_kernel("kernel(0.6, 0.2)+hanning"), [0.05, 0.25, 0.4, 0.25, 0.05])
