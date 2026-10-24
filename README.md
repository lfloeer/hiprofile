# hiprofile

Bayesian fitting of global HI line profiles with the model of
Stewart, Blyth & de Blok (2014, A&A 567, A61): posteriors, evidences for
N = 0, 1, 2, … components, detection probabilities, catalogue, diagnostics and
plots for whole spectrum lists.

```bash
uv pip install -e ".[all]"
hiprofile fit spectra.txt --config examples/survey.yaml --jobs 32 --out results/
```

Per spectrum, `results/spectra/<id>.h5` holds weighted posterior samples
(including W50, W20, flux and systemic velocity), evidences for every N and
Chebyshev baseline order K, P(detect), quality metrics and flags;
`results/catalog.fits` and `results/report.html` summarise all spectra.

## How it works

* **Model kernel** (`hiprofile.model`): ring and disk basis functions evaluated in
  Fourier space (Bessel functions, Gaussian, exact channel integration and
  Hanning/Tukey responses); compiled with numba by default, NumPy fallback, and
  JAX (`jit`, `vmap`, gradients). Only four parameters are nonlinear: v_ctr,
  Δv, Δv_rand, α. A full fit of a 512-channel spectrum (N ≤ 2, K ≤ 3) takes
  3–10 s on one core.
* **Likelihood** (`hiprofile.likelihood`): ring/disk amplitudes and the Chebyshev
  baseline are marginalised analytically (Woodbury); white or Hanning-correlated
  noise; masks; a robust outlier-mixture pre-stage for RFI.
* **Samplers** (`hiprofile.samplers`): tempered SMC (default, fastest), nautilus
  (most precise evidences), dynesty, UltraNest and a quick MAP + Laplace +
  importance-sampling mode behind one interface; `--mode quick | full | check`.
* **Model comparison** (`hiprofile.inference`): evidences for all N and K with a
  Poisson prior on N; amplitude positivity by exact reweighting.
* **Validation** (`hiprofile simulate`, `hiprofile validate`): simulation-based
  calibration, detection/confusion statistics, misspecified profiles,
  baselines and RFI, injection into real spectra; always compared with the
  classical W50/W20.
* **Survey runs**: process pool (`--jobs`), SLURM arrays (`--chunk i/n`),
  atomic per-spectrum files, `--resume`, per-spectrum seeds.

The original Cython code is kept in `legacy/` as frozen reference; the new
kernel reproduces it to 1e-11 (golden files in `tests/reference/`).

Documentation: `docs/` (`mkdocs serve`): [model](docs/model.md),
[inference](docs/inference.md), [outputs and flags](docs/outputs.md),
[configuration](docs/configuration.md), [tutorial](docs/tutorial.md),
[validation](docs/validation.md), [legacy code](docs/legacy.md).

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest              # fast tests incl. golden files
pytest -m slow      # nested samplers, full mode
ruff check hiprofile tests && mypy hiprofile
```

## Citation

See `CITATION.cff`; please cite Stewart, Blyth & de Blok (2014) for the model.
