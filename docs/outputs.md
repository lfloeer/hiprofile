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
/evidence                 n, k, log_z, log_z_err, log_z_gauss, method (analytic | marginal |
                          quick | quick-gated), backend,
                          ess_fraction, negative_fraction, runtime, n_like,
                          log_z_check, log_z_check_err
/posterior/N<n>           theta, u, weights, beta, log_l, k_samples; attrs names, k,
                          k_values, k_probability (posterior marginalised over K;
                          beta zero-padded to K_max; k = most probable order)
/posterior/N<n>/derived   S_i, f_i, W50_i, W20_i, v_sys_i, peak_i (per component)
/posterior/N<n>/trace     sampler diagnostics
```

Read it with `hiprofile.io.read_result(path)`.

`theta` columns per profile component: `v_ctr`, `delta_v`, `delta_v_rand`,
`alpha`, `f`; `beta` holds the integrated flux `S` of each component, then
the Chebyshev coefficients up to `k_max` (zero beyond the order drawn for the
sample). W50/W20 are measured on the intrinsic profile (with random
motions, without spectrometer response) as the distance between the outermost
crossings of 50 %/20 % of the peak; `v_sys` is the midpoint of the W50
crossings.

## Catalogue columns

| column | meaning |
|---|---|
| `id`, `status` | spectrum id, `ok` or `failed` |
| `p_detect`, `p_n<N>` | P(N ≥ 1 \| y), P(N \| y) |
| `best_n`, `best_k` | most probable N, best K for it |
| `logz_n<N>_k<K>` | evidences; `p_k<K>` the posterior probability of each baseline order |
| `<par>`, `<par>_p16`, `<par>_p84` | posterior median and 68 % interval (for N = max(best_n, 1)) |
| `classic_W50`, `classic_W20`, `classic_S`, `classic_v_sys` | naive classical measurement on the baseline-subtracted data: outermost channels above 50 % / 20 % of the peak channel, interpolated |
| `classic_W50_springob`, `classic_W20_springob`, `classic_v_sys_springob` | Springob et al. (2005) W_F50: straight-line fits to each flank between 85 % and 15 % of (horn − rms), read at 50 % (and, as an extension, 20 %); Hanning correction Δs = 2 dv λ(S/N) applied when `model.response` contains `hanning`; `classic_W50_raw_springob` before the correction, `classic_snr_springob` = (horn − rms) / rms |
| `m_<metric>` | quality metrics (below) |
| `flag_<name>`, `flagged` | flags |
| `meta_<key>` | metadata from the input (e.g. catalogue columns) |

## Metrics and flags

| metric | flag | flagged when |
|---|---|---|
| `chi2_red`, `ppp` (posterior predictive p) | `ppp` | p < `diagnostics.ppp_min` (0.01) |
| `wiggle_J`, `wiggle_snr` (Stewart et al. Eq. 8) | `wiggle` | J > `wiggle_max` and J² significant at 3σ |
| `residual_lowfreq_lnp`, `residual_lowfreq_ratio` (excess power of the whitened residuals at periods longer than 1/`structure_fmax` = 10 channels: ln p against the noise model, observed/expected power) | `residual_structure` | p < `structure_p` (10⁻³): baseline ripple, RFI or a profile the model does not describe |
| `residual_acf` (lag 1–3, in units of 1/√n; informational) | – | – (it also reacts to a slightly wrong white-noise fraction near the Nyquist frequency, which carries no line information; on real ALFALFA spectra that fired in a quarter of the fits) |
| `edge_<par>`, `edge_band_<i>`, `edge_mass` (posterior mass next to a bound of α, Δv, Δv_rand (f is bounded by definition and not checked), or of a centre pressed against the band edge by `priors.contain`: within half the posterior's 16–84 % width, at most 1/20 of the range) | `prior_edge` | > `edge_mass` (0.2): the bound cuts the posterior |
| `multimodal_weight`, `multimodal_separation` (2-means in the unit cube) | `multimodal` | weight > 0.1 and separation > 4 |
| `ess` (Kish) | `low_ess` | < `min_ess` (500) |
| `logz_check_sigma` (check mode) | `logz_mismatch` | > `logz_sigma` (3) |
| `negative_fraction` | `negative_amplitudes` | > `negative_fraction` (0.5) |
| `constraint_<par>` (1 − posterior/prior 16–84 % width) | `prior_dominated` | `constraint_delta_v` < `min_constraint` (0.3): the width is not measured |
| `w50_min_channels` (narrowest component) | `unresolved` | < `min_w50_channels` (2): an RFI-like component narrower than the channel response |
| `rfi_fraction` | `rfi` | > `rfi.flag_fraction` (0.05) |
| `sampler_budget` | `sampler_budget` | a sampler stopped at its time / call limit (nautilus: 600 s or 10⁶ calls; options in `sampler.options`) |
| – | `failed` | the fit raised an error |

Profile-specific flags are cleared when the most probable model is N = 0.

## Plots

Per spectrum: data with posterior-predictive band, components, baseline and
normalised residuals (with the RFI outlier probability); a corner plot of the
nonlinear parameters plus W50, W20, S; evidences for all N and K; sampler
diagnostics. By default (`output.plots: sample`) only flagged spectra and a
deterministic random fraction are plotted; `hiprofile plot results [ids]
[--flagged]` draws the rest from the stored samples.
