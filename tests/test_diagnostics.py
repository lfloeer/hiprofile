import numpy as np

from hiprofile.diagnostics import metrics as dm


def test_edge_mass_flags_only_posteriors_cut_by_a_bound(rng):
    w = np.ones(20000)
    # half-normal against the lower bound: cut off
    assert dm.edge_mass(4.0 + np.abs(rng.normal(size=w.size)), w, 4.0, 60.0) > 0.3
    # mean 1 sigma inside the bound: still cut off
    x = rng.normal(5.0, 1.0, size=w.size)
    assert dm.edge_mass(x[x >= 4.0], w[x >= 4.0], 4.0, 60.0) > 0.2
    # narrow posterior far (in its own width) from the bound of a wide range: not an edge
    assert dm.edge_mass(rng.normal(20.0, 3.0, size=w.size), w, 0.0, 800.0) < 1e-3
    # flat, prior-like posterior: touches both bounds but is not cut by them
    assert dm.edge_mass(rng.uniform(size=w.size), w, 0.0, 1.0) < 0.07
    # unbounded parameter
    assert dm.edge_mass(rng.normal(size=w.size), w, -np.inf, np.inf) == 0.0
