# hiprofile

`hiprofile` fits global HI line profiles with the six-parameter model of
Stewart, Blyth & de Blok (2014, A&A 567, A61) and Bayesian model comparison.
A single call fits a whole list of spectra in parallel:

```bash
hiprofile fit spectra.txt --config survey.yaml --jobs 32 --out results/
```

and writes, per spectrum, an HDF5 file with weighted posterior samples (including
W50, W20, flux and systemic velocity), evidences for N = 0, 1, 2, ... components
and Chebyshev baseline orders K, the detection probability, quality metrics and
flags, and optionally plots. Over all spectra it builds a catalogue
(FITS/Parquet/CSV) and an HTML report.

## Installation

```bash
uv pip install "hiprofile[all]"      # JAX, dynesty, UltraNest, matplotlib, pyarrow
uv pip install hiprofile             # minimal: compiled kernel, SMC and nautilus
```

Extras: `[jax]`, `[gpu]` (CUDA JAX), `[samplers]`, `[plots]`, `[parquet]`, `[dev]`.

## Commands

| command | purpose |
|---|---|
| `hiprofile fit` | fit a spectrum list (list file, glob, catalogue with paths or cube positions) |
| `hiprofile simulate` | generate test spectra with known truth |
| `hiprofile validate` | run the validation stages (SBC, detection, misspecification, injection) |
| `hiprofile report` | build catalogue and HTML report from existing results |
| `hiprofile plot` | plots for single or flagged spectra from stored samples |

## Package layout

```
hiprofile/
  model/        Fourier kernel, ring and disk basis functions, spectrometer response
  likelihood/   marginalisation, noise models, baseline priors, RFI pre-stage
  samplers/     smc (default), nautilus, dynesty, ultranest, quick (MAP + Laplace + IS)
  inference/    fit per spectrum: N and K, evidences, amplitudes, derived quantities
  simulate/     simulator, templates, injection
  validate/     SBC, confusion matrix, coverage
  diagnostics/  metrics, flags
  plots/        spectrum, corner, evidences, sampler, HTML report
  io/           loading spectra, result files, catalogue, cube extraction
  cli.py        hiprofile fit | simulate | validate | report | plot
legacy/         original Cython code, kept as frozen reference
tests/          unit tests, tests/reference/ with golden files
```

## Citation

Please cite Stewart, Blyth & de Blok (2014), A&A 567, A61 (arXiv:1405.1838)
for the model; see `CITATION.cff`.
