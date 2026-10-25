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


def test_residual_structure_ignores_the_nyquist_band_but_finds_baselines(rng):
    """Low-frequency test: calibrated on white noise, blind to a wrong white-noise fraction
    (pure Hanning noise whitened with a 0.5 % white part, as on real ALFALFA spectra, where
    the lag-1 autocorrelation fired in a quarter of the fits), sensitive to a baseline ripple."""
    from scipy.linalg import cholesky, solve_triangular, toeplitz

    n = 380
    lnp_white = np.array([dm.residual_lowfreq(rng.standard_normal(n))[0] for _ in range(2000)])
    assert abs(np.mean(lnp_white < np.log(0.01)) - 0.01) < 0.008
    rho = np.zeros(n)
    rho[:3] = 1.0, 2 / 3 * 0.995, 1 / 6 * 0.995
    chol = cholesky(toeplitz(rho), lower=True)
    nyq = []
    for _ in range(500):
        h = np.convolve(rng.standard_normal(n + 2), [0.25, 0.5, 0.25], "valid") / np.sqrt(0.375)
        nyq.append(dm.residual_lowfreq(solve_triangular(chol, h, lower=True))[0])
    assert np.mean(np.array(nyq) < np.log(1e-3)) < 0.01
    ripple = rng.standard_normal(n) + 0.5 * np.sin(2 * np.pi * np.arange(n) / 60.0)
    assert dm.residual_lowfreq(ripple)[0] < np.log(1e-3)
