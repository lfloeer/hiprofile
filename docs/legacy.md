# Legacy code (Phase 0)

The original Cython implementation (`lineprofile`, emcee-based) lives in
`legacy/` as a frozen reference. CI builds it and checks that it still
reproduces the golden files.

## Build with current toolchains

Tested with Python 3.12, NumPy 2.5, Cython 3.3, FFTW 3.3.10. Changes needed:

| problem | fix |
|---|---|
| implicit relative `cimport` (`from model cimport LineModel`, `cimport ln_likes`) fails in Cython 3 | absolute `from lineprofile.model cimport ...` |
| `from numpy.math cimport INFINITY` no longer available | `from libc.math cimport INFINITY` |
| `xrange` in `utils.py` (Python 2) | `range` |
| `fftw3.h` required | `libfftw3-dev` |

```bash
cd legacy && python setup.py build_ext --inplace
python make_reference.py --check
```

**Bug:** `LineModel.__dealloc__` calls `fftw_cleanup()`, which frees FFTW's
global state while other plans are alive, so deallocating one model can crash
the next (observed segfaults). The reference script keeps all models alive.

## Model, units and parameters (legacy)

* `LineModel(velocities, n_disks, n_gaussians, n_baseline, supersample=2)`;
  parameter vector per disk `(log10 S, v_sys, v_rot, v_turb, f_solid, asym)` =
  `(log10 S, v_ctr, Δv, Δv_rand, f, α)`, then per Gaussian `(log10 S, v, log10 σ)`,
  then baseline coefficients (plain polynomial in x ∈ [-1, 1], highest order
  first, Horner scheme — not Chebyshev).
* Evaluation on `n_chan * supersample` points with spacing `dv / supersample`
  starting at the first channel centre, periodic without padding, then the mean
  of each block of `supersample` points. Consequences: the profile is shifted
  by `dv (s-1) / (2s)` (a quarter channel for s = 2) and wraps at the band edges.
* Fit classes `FitGaussian`, `FitLaplacian`, `FitMixture` (Hogg-style outlier
  mixture with parameters `(log10 P_b, log10 σ_in, μ_out, log10 σ_out)`), priors:
  Gamma on Δv (k = 4.38, θ = 55.4, from HIPASS), shifted Gamma on Δv_rand,
  Beta(1, 5) on f, Beta(10, 10) on (α+1)/2, normal on v_ctr and baseline.
  Note: `FitGaussian.ln_bounds_likelihood` tests `p <= -3 and p >= 3`, which is
  never true, so the bound is not applied.

## Golden files and benchmark

`legacy/make_reference.py` writes `tests/reference/golden_profiles.npz`: a full
factorial grid Δv ∈ {10⁻³, 50, 230, 500}, Δv_rand ∈ {2, 8, 25},
f ∈ {0, 0.3, 1}, α ∈ {−1, −0.2, 0, 1} (random S and non-integer centres) on
four grids (dv = 2, 5, 10.24 km/s; supersample 2 and 4), plus legacy
log-posterior values. The new kernel reproduces all 432 profiles to
1×10⁻¹¹ relative to the peak (acceptance: 10⁻⁶).

`tests/reference/benchmark_legacy.json`: one legacy model evaluation takes
27 / 50 / 112 / 206 µs for 256 / 512 / 1024 / 2048 channels (supersample 2,
single profile, x86_64). `benchmarks/bench_kernel.py` writes the comparison to
`tests/reference/benchmark_new.json`. On the same grid as the legacy code
(supersample 2, no padding) the compiled kernel computes ring *and* disk in
3.5 / 7.2 / 16 / 39 µs per parameter vector in batches (5–8x faster than
legacy) and 13 / 17 / 21 / 59 µs as single calls (2–3.5x faster).

## Not reproduced

The fit of a THINGS spectrum from Table 1 of the paper needs the THINGS global
profiles, which are not part of this repository; `hiprofile simulate` with
`template_files` accepts such profiles as templates once available.
