# Outputs and flags

## Directory layout of `hiprofile fit --out results/`

```
results/
  config.yaml          validated configuration of the run
  hiprofile.log        log, including failed spectra
  spectra/<id>.h5      one result file per spectrum (written atomically)
  plots/<id>_*.png     spectrum, corner, evidence and sampler plots
  catalog.fits         one row per spectrum (or .parquet / .csv)
  report.html          distributions, flag statistics, flagged spectra
```

`--resume` skips spectra whose result file exists. Seeds derive from the global
seed and the spectrum id, so a resumed, re-ordered or re-chunked run gives
identical results. A failing spectrum is stored with `status = failed` and the
error message; the run continues.

## Result file

```
/                         attrs: id, status, error, version, seed, config (YAML),
                          p_detect, best_n, best_k, runtime, and JSON-encoded
                          p_n, metrics, flags, classic, meta, catalog_row
/spectrum                 velocity, flux, sigma, mask (incl. RFI), rfi_q
/model                    band (2.5/16/50/84/97.5 % posterior predictive),
                          components_median, baseline_median
/evidence                 n, k, log_z, log_z_err, log_z_gauss, method, backend,
                          ess_fraction, negative_fraction, runtime, n_like,
                          log_z_check, log_z_check_err
/posterior/N<n>           theta, u, weights, beta, log_l; attrs names, k
/posterior/N<n>/derived   S_i, f_i, W50_i, W20_i, v_sys_i, peak_i (per component)
/posterior/N<n>/trace     sampler diagnostics
```

Read it with `hiprofile.io.read_result(path)`.

`theta` columns per profile component: `v_ctr`, `delta_v`, `delta_v_rand`,
`alpha`; `beta` holds `a_ring`, `a_disk` per component, then the Chebyshev
coefficients. W50/W20 are measured on the intrinsic profile (with random
motions, without spectrometer response) as the distance between the outermost
crossings of 50 %/20 % of the peak; `v_sys` is the midpoint of the W50
crossings.

## Catalogue columns

| column | meaning |
|---|---|
| `id`, `status` | spectrum id, `ok` or `failed` |
| `p_detect`, `p_n<N>` | P(N ≥ 1 \| y), P(N \| y) |
| `best_n`, `best_k` | most probable N, best K for it |
| `logz_n<N>_k<K>` | evidences |
| `<par>`, `<par>_p16`, `<par>_p84` | posterior median and 68 % interval (for N = max(best_n, 1)) |
| `classic_W50`, `classic_W20`, `classic_S`, `classic_v_sys` | classical measurement on the data |
| `m_<metric>` | quality metrics (below) |
| `flag_<name>`, `flagged` | flags |
| `meta_<key>` | metadata from the input (e.g. catalogue columns) |

## Metrics and flags

| metric | flag | flagged when |
|---|---|---|
| `chi2_red`, `ppp` (posterior predictive p) | `ppp` | p < `diagnostics.ppp_min` (0.01) |
| `wiggle_J`, `wiggle_snr` (Stewart et al. Eq. 8) | `wiggle` | J > `wiggle_max` and J² significant at 3σ |
| `residual_acf` (lag 1–3, in units of 1/√n) | `residual_acf` | any \|value\| > `acf_sigma` (4) |
| `edge_<par>`, `edge_mass` (outermost of 20 bins at f, α, Δv, Δv_rand edges) | `prior_edge` | > `edge_mass` (0.2) |
| `multimodal_weight`, `multimodal_separation` (2-means in the unit cube) | `multimodal` | weight > 0.1 and separation > 4 |
| `ess` (Kish) | `low_ess` | < `min_ess` (500) |
| `logz_check_sigma` (check mode) | `logz_mismatch` | > `logz_sigma` (3) |
| `negative_fraction` | `negative_amplitudes` | > `negative_fraction` (0.5) |
| `rfi_fraction` | `rfi` | > `rfi.flag_fraction` (0.05) |
| – | `failed` | the fit raised an error |

Profile-specific flags are cleared when the most probable model is N = 0.

## Plots

Per spectrum: data with posterior-predictive band, components, baseline and
normalised residuals (with the RFI outlier probability); a corner plot of the
nonlinear parameters plus W50, W20, S; evidences for all N and K; sampler
diagnostics. By default (`output.plots: sample`) only flagged spectra and a
deterministic random fraction are plotted; `hiprofile plot results [ids]
[--flagged]` draws the rest from the stored samples.
