"""The new kernel reproduces the frozen legacy model (tests/reference)."""

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_jax
from hiprofile.model import legacy_profile

REF = Path(__file__).parent / "reference" / "golden_profiles.npz"


def _grids():
    ref = np.load(REF)
    i = 0
    while f"grid{i}_velocities" in ref:
        yield (
            ref[f"grid{i}_velocities"],
            ref[f"grid{i}_params"],
            ref[f"grid{i}_profiles"],
            int(ref[f"grid{i}_supersample"]),
        )
        i += 1


def _check(backend):
    n = 0
    for v, params, prof, ss in _grids():
        new = np.asarray(legacy_profile(v, params, ss, backend))
        scale = np.max(np.abs(prof), axis=1, keepdims=True)
        err = np.max(np.abs(new - prof) / scale, axis=1)
        assert err.max() < 1e-6, f"max relative deviation {err.max():.2e}"
        n += params.shape[0]
    assert n >= 400


def test_golden_numpy():
    _check("numpy")


@requires_jax
def test_golden_jax():
    _check("jax")


def test_golden_grid_covers_edges():
    allp = np.vstack([p for _, p, _, _ in _grids()])
    assert {0.0, 1.0} <= set(allp[:, 4])
    assert {-1.0, 1.0} <= set(allp[:, 5])
    assert allp[:, 2].min() <= 1e-3


# legacy LineModel.__dealloc__ calls fftw_cleanup() and crashes other live plans
_KEEP_ALIVE: list = []


@pytest.mark.legacy
def test_legacy_code_reproduces_golden():
    lp = pytest.importorskip("lineprofile.model")
    for v, params, prof, ss in _grids():
        m = lp.LineModel(v, n_disks=1, supersample=ss)
        _KEEP_ALIVE.append(m)
        out = np.array([m.model(np.ascontiguousarray(p)) for p in params[:20]])
        np.testing.assert_allclose(out, prof[:20], rtol=0, atol=1e-12 * np.abs(prof).max())
