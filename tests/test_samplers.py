import numpy as np
import pytest

from hiprofile.samplers import run

MU = np.array([0.3, 0.6, 0.5])
SD = np.array([0.05, 0.02, 0.1])


def gaussian_ll(t):
    return -0.5 * np.sum(((t - MU) / SD) ** 2, axis=1) - np.sum(np.log(np.sqrt(2 * np.pi) * SD))


def _check(backend, tol):
    r = run(backend, gaussian_ll, lambda u: u, 3, seed=3, n_live=300)
    assert abs(r.log_z) < tol, (backend, r.log_z, r.log_z_err)
    m = np.sum(r.weights[:, None] * r.samples, axis=0)
    np.testing.assert_allclose(m, MU, atol=0.02)
    assert r.n_like > 0


@pytest.mark.parametrize("backend,tol", [("quick", 0.05), ("smc", 0.3)])
def test_fast_backends(backend, tol):
    _check(backend, tol)


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["nautilus", "dynesty", "ultranest"])
def test_nested_backends(backend):
    _check(backend, 0.5)


def test_smc_warm_start_keeps_the_evidence_exact():
    """A quarter of the particles from the quick proposal (defensive mixture) gives the same ln Z."""
    q = run("quick", gaussian_ll, lambda u: u, 3, seed=3, n_live=300)
    assert q.proposal is not None
    r = run("smc", gaussian_ll, lambda u: u, 3, seed=4, n_live=300, init_proposal=(q.proposal, 0.25))
    assert abs(r.log_z) < 0.3, (r.log_z, r.log_z_err)
    m = np.sum(r.weights[:, None] * r.samples, axis=0)
    np.testing.assert_allclose(m, MU, atol=0.02)
