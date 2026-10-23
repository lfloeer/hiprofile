"""Fit one simulated spectrum from Python and print the catalogue row."""

import numpy as np

from hiprofile import Spectrum
from hiprofile.config import Config
from hiprofile.inference.fit import fit_spectrum
from hiprofile.model import SpectralGrid, profile
from hiprofile.plots.spectrum import plot_all

rng = np.random.default_rng(1)
v = 1000.0 + 5.0 * np.arange(512)
grid = SpectralGrid.from_velocities(v)
# (v_ctr, delta_v, delta_v_rand, alpha, S, f)
truth = [2200.0, 250.0, 12.0, -0.2, 30.0, 0.2]
flux = profile(grid, truth) + rng.normal(0, 0.02, v.size)
spec = Spectrum(v, flux, id="example")

cfg = Config().updated(sampler={"mode": "quick"})
res = fit_spectrum(spec, cfg)
row = res.catalog_row()
print(f"P(detect) = {res.p_detect:.4f}, N = {res.best_n}, K = {res.best_k}")
for key in ("v_ctr_1", "delta_v_1", "S_1", "f_1", "W50_1", "W20_1"):
    print(f"{key:>10s} = {row[key]:9.3f}  [{row[key + '_p16']:.3f}, {row[key + '_p84']:.3f}]")
plot_all(res, "example_plots")
