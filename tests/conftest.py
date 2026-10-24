import numpy as np
import pytest

from hiprofile.model import SpectralGrid, profile


def has_jax() -> bool:
    try:
        import jax  # noqa: F401
    except ImportError:
        return False
    return True


requires_jax = pytest.mark.skipif(not has_jax(), reason="JAX not installed")


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def grid():
    v = 1000.0 + 5.0 * np.arange(400)
    return SpectralGrid.from_velocities(v)


@pytest.fixture
def line_spectrum(grid, rng):
    from hiprofile import Spectrum

    truth = np.array([2000.0, 180.0, 12.0, 0.15, 20.0, 0.25])
    y = profile(grid, truth) + 0.02 * rng.normal(size=grid.n_chan)
    return Spectrum(grid.velocity, y, sigma=np.full(grid.n_chan, 0.02), id="line"), truth
