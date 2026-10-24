import numpy as np

from hiprofile.inference.derived import classic_widths, profile_derived, springob_lambda, springob_widths
from hiprofile.model import SpectralGrid, profile


def _line(response="boxcar+hanning"):
    v = 1000.0 + 5.0 * np.arange(400)
    grid = SpectralGrid.from_velocities(v, supersample=8, response=response)
    theta = [2000.0, 250.0, 12.0, 0.0, 5.0, 0.25]  # symmetric: per-horn levels = global
    y = profile(grid, theta)
    d = profile_derived(grid, np.array([theta[:4]]), np.array([[5.0 * 0.75, 5.0 * 0.25]]))
    return v, y, {k: float(d[k][0]) for k in ("W50", "W20", "v_sys")}


def test_springob_lambda_table2():
    # Hanning, dv < 5 km/s: lambda1 below log SNR 0.6, lambda3 above 1.1, linear between
    assert springob_lambda(2.0, 2.6) == 0.005
    assert springob_lambda(100.0, 2.6) == 0.395
    assert abs(springob_lambda(10**0.8, 2.6) - (-0.4685 + 0.785 * 0.8)) < 1e-12
    # 5-11 km/s interpolates linearly in dv between the two outer rows at its ends
    assert abs(springob_lambda(100.0, 8.0) - (0.023 * 8 + 0.28)) < 1e-12


def test_springob_recovers_noiseless_widths():
    v, y, truth = _line()
    sp = springob_widths(v, y, 1e-6, (1700.0, 2300.0), 2000.0, hanning=False)
    assert abs(sp["W50"] - truth["W50"]) < 2.5  # straight line on a curved flank
    assert abs(sp["W20"] - truth["W20"]) < 5.0
    assert abs(sp["v_sys"] - truth["v_sys"]) < 1.0


def test_springob_is_robust_where_naive_w20_is_not(rng):
    v, y, truth = _line()
    err_sp, err_naive = [], []
    for _ in range(30):
        noisy = y + 0.004 * rng.standard_normal(v.size)
        sp = springob_widths(v, noisy, 0.004, (1700.0, 2300.0), 2000.0, hanning=False)
        err_sp.append(sp["W20"] - truth["W20"])
        err_naive.append(classic_widths(v, noisy, window=(1700.0, 2300.0))["W20"] - truth["W20"])
    assert abs(np.nanmedian(err_sp)) < 8.0
    assert np.median(err_naive) > 50.0  # noise peaks above 20 % of the peak channel widen it


def test_springob_nan_without_flank():
    v = np.arange(100) * 5.0
    out = springob_widths(v, np.zeros(100), 0.01, (100.0, 400.0), 250.0)
    assert np.isnan(out["W50"]) and np.isnan(out["W20"])
