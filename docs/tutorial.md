# Tutorial: from simulation to catalogue

## 1. Simulate test spectra

```bash
hiprofile simulate --config examples/simulate.yaml --n 200 --out sims --seed 1
```

This writes `sims/spectra/*.fits`, a list file `sims/spectra.txt` and the truth
table `sims/truth.csv` (true N, component parameters, W50/W20, S/N, RFI
channels).

## 2. Quick look

```bash
hiprofile fit sims/spectra.txt --mode quick --jobs 8 --out quick
```

`quick` uses MAP + Laplace + importance sampling: seconds per spectrum, good
start values, but no guarantee for multimodal posteriors.

## 3. Full fit

```bash
hiprofile fit sims/spectra.txt --config examples/survey.yaml --jobs 8 --out results
```

Inputs can also be a glob (`"data/*.fits"`) or a catalogue table with a `path`
column (other columns are kept as `meta_*`), or with `cube`, `ra`, `dec`
columns to extract aperture spectra from a cube. Column names and units are
detected automatically; override them with `--flux-column`, `--velocity-unit`,
`--frequency-column`, `--convention` etc. Frequency axes are converted with
the radio convention by default, which keeps a uniform grid uniform.

On a cluster:

```bash
sbatch examples/slurm_array.sh      # --chunk $SLURM_ARRAY_TASK_ID/100 --resume
hiprofile report results            # catalogue + report once all chunks are done
```

## 4. Inspect

```bash
open results/report.html
hiprofile plot results sim000042         # plots for one spectrum
hiprofile plot results --flagged         # all flagged spectra
```

```python
from hiprofile.io import read_result
res = read_result("results/spectra/sim000042.h5")
post = res.posteriors[1]
print(res.p_detect, post.summary()["W50_1"])
```

## 5. Validate

```bash
hiprofile validate --stage sbc --n 1000 --jobs 32 --out validation
hiprofile validate --stage detection --stage misspec --n 500 --jobs 32 --out validation
hiprofile validate --stage injection --spectra signal_free.txt --n 500 --out validation
```

Each stage writes `validation/validate_<stage>/summary.md`, `summary.json` and
plots; see [Validation](validation.md).

## From Python

See `examples/quickstart.py`:

```python
from hiprofile import Spectrum
from hiprofile.config import Config
from hiprofile.inference.fit import fit_spectrum

res = fit_spectrum(Spectrum(velocity, flux, id="g1"), Config())
```
