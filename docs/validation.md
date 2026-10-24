# Validation

Everything on this page comes from one run of `benchmarks/validate_all.sh`
on a 4-core machine with the default configuration (full mode, SMC sampler,
N ≤ 2, K ≤ 3, RFI pre-stage, f ~ Beta(1, 3)), 5 km/s channels unless stated
otherwise. Each `summary.json` records the commit: 61a2b05 for the SBC and
the correlated-noise study, 0f993e5 for detection, misspecification, bias
and ALFALFA. The two differ only in the automatic flux-prior scale and the
RFI run limit, which the SBC and the correlated-noise study do not use (both
fix the flux scale and fit without the RFI pre-stage).

| stage | command | spectra |
|---|---|---|
| SBC | `hiprofile validate --stage sbc --n 1000 --seed 11` | 1000 |
| detection | `hiprofile validate --stage detection --n 200 --seed 21` | 200, fitted twice |
| misspecification | `hiprofile validate --stage misspec --n 100 --seed 31` | 5 × 100 |
| conditional bias | `python benchmarks/bias_study.py --n 20 --priors default\|flat` | 2 × 500 |
| correlated noise | `python benchmarks/correlated_noise.py --n 300 --n-noise 150` | 450, fitted four times |
| real data | `python benchmarks/alfalfa_study.py` | 10 ALFALFA sources |

## Stage 1: simulation-based calibration

Every simulation is drawn from the fit priors: the five shape parameters,
the flux (half-normal, scale 4 Jy km/s) and a Chebyshev baseline of order
K ~ U{0, …, 3} with coefficients from the baseline prior; white noise with
σ = 0.01 Jy on 512 channels. The fit uses the same priors with the order
marginalised. The rank of the truth among 99 posterior draws must then be
uniform, and the posterior intervals must cover the truth at their nominal
rate.

| parameter | rank p-value | 68 % coverage | 95 % coverage |
|---|---|---|---|
| v_ctr | 0.91 | 0.69 | 0.96 |
| Δv | 0.86 | 0.68 | 0.94 |
| Δv_rand | 0.39 | 0.69 | 0.95 |
| α | 0.48 | 0.66 | 0.95 |
| S | 0.58 | 0.67 | 0.95 |
| f | 0.70 | 0.70 | 0.94 |
| W50 | 0.59 | 0.70 | 0.95 |
| W20 | 0.23 | 0.67 | 0.95 |

![SBC: rank histograms, interval coverage and baseline-order calibration](img/sbc.png)

All rank histograms are flat (p > 0.2), and the coverages lie inside the
±2σ bands for 1000 simulations (0.65–0.71 and 0.94–0.96). The faint draws
of the flux prior that are not detected take part like every other
simulation.

**Baseline order.** The most probable K is the true one in 93 % of the
simulations and P(K_true | y) averages 0.88; the reliability curve (right
panel) follows the diagonal. The confusion matrix (rows: true K, columns:
most probable K) is one-sided:

```
  266      2      0      0
   14    229      0      0
    5     13    222      0
    2      6     26    215
```

A true K = 3 whose highest coefficients happen to be small is read as K = 2,
the right answer for terms the data cannot see. This calibration holds when
the baseline prior matches the baselines; Stage 3 shows what happens when
it does not.

**Flags.** 11.4 % of the fits carry a flag: `prior_edge` 5.2 % (Δv_rand at
its 4 km/s bound for faint, narrow lines), `prior_dominated` 3.4 %,
`wiggle` 2.1 %, `multimodal` 1.3 %, `ppp` 1.1 %.

The same simulations measured against the truth (detected, N = 1):

| quantity | Bayes bias | Bayes scatter | classical bias | classical scatter |
|---|---|---|---|---|
| W50 [km/s] | 0.3 | 10.0 | 0.2 | 18.2 |
| W20 [km/s] | 0.5 | 11.9 | 26.2 | 35.5 |
| S [Jy km/s] | −0.01 | 0.35 | 0.03 | 0.40 |
| v_sys [km/s] | −0.2 | 4.6 | 0.1 | 8.2 |

![Line widths: Bayesian fit against the classical measurement](img/bayes_vs_classic.png)

The posterior medians are unbiased, and their scatter is half (W50, v_sys)
to a third (W20) of that of the classical measurement on the same data. The
classical W20 is biased high at low S/N because noise peaks raise the 20 %
crossing points.

Example posterior (corner plot of a low-S/N spectrum with a bimodal centre):

![corner plot](img/corner_example.png)

## Stage 2: detection and confusion

200 spectra with N = 0, 1, 2 (probabilities 0.4, 0.4, 0.2), integrated S/N
2–40 per component, a first-order baseline and single-channel RFI spikes
(2–3 per spectrum), fitted with and without the RFI pre-stage and the
automatic flux-prior scale.

![Detection: completeness, reliability and confusion matrix](img/detection.png)

| | with RFI pre-stage | without |
|---|---|---|
| false positives, P(detect) > 0.5 / 0.9 | 0 of 77 / 0 | 83 % / 83 % |
| completeness S/N 3–5 / 5–8 / ≥ 8 | 0.47 / 0.94 / 1.00 | 0.84 / 1.00 / 1.00 |
| flagged | 4.5 % | 87 % |

Confusion matrix with the pre-stage (rows: true N, columns: most probable N):

```
   77      0      0
   19     64      0
    2     19     19
```

* **N = 0 and N = 1** are separated cleanly: no false positive at any
  threshold, and no single line split into two. Without the pre-stage the
  RFI spikes are fitted as lines (83 % false positives, 60 % flagged
  `unresolved`); its higher "completeness" is the same effect.
* **The completeness** at S/N 3–5 is that of the SBC, which has no RFI, so
  the pre-stage costs none.
* **N = 2.** Of the 18 pairs with both components at S/N > 6 and separated
  by more than their width, 15 are fitted as N = 2. Pairs with a second
  component below S/N 6 (21 of 40) are mostly fitted as N = 1: a second line
  at S/N 3 is not detectable on its own either.
* Widths of the detected single lines: W50 bias 0.8 km/s, scatter 10.1
  (classical 11.8), W20 bias 1.1 (classical 24.8).

## Stage 3: misspecification

Five sets of 100 one-component spectra at S/N 5–80: model profiles
(`clean`); profiles the model does not generate (`template`: tilted-ring
galaxies with a rising rotation curve, inclinations 30–85°, lopsidedness up
to 30 % and central HI holes, integrated on a fine grid); model profiles on
a sinusoidal baseline that no Chebyshev polynomial up to K = 3 describes;
model profiles with RFI and no pre-stage; and Chebyshev baselines of order
0–3 whose coefficients are 50 times smaller than the fit's automatic prior
scale.

![Flag rates on misspecified spectra](img/misspec_flags.png)

| set | flagged | W50 bias / scatter [km/s] | 68 % coverage of W50 |
|---|---|---|---|
| clean | 6 % | 0.8 / 4.7 | 0.74 |
| template | 5 % | −0.5 / 6.6 | 0.68 |
| sine baseline | 99 % | −0.1 / 21.8 | 0.34 |
| RFI unmasked | 100 % | −90 / 111 | 0.18 |
| baseline prior 50 × too wide | 10 % | −0.2 / 6.0 | 0.67 |

* The model describes the galaxy-like templates well enough that the widths
  stay unbiased and calibrated.
* When the model is wrong the posteriors are overconfident, as any
  posterior of a wrong model is, but the flags catch it: `ppp` and
  `residual_acf` mark almost every sine-baseline fit; `ppp`, `prior_edge`,
  `unresolved` and `wiggle` mark the RFI fits.
* **A wrong baseline prior** leaves the line parameters unbiased and
  calibrated, but not the baseline order: the most probable K is right in
  41 % of the spectra, nearly always too low (22 of 28 true K = 3 are fitted
  as K = 0). Coefficients of 0.2 σ per channel are barely visible, and with
  a prior 50 times too wide the Occam factor wins. The `p_k` columns of the
  catalogue are calibrated only with a fitting `priors.baseline_scale`, for
  example from signal-free neighbour spectra
  (`hiprofile.likelihood.baseline.baseline_prior_from_spectra`).

## Conditional bias: default against flat priors

The SBC shows that the posteriors are calibrated *on average over the
prior*. For a given galaxy the posterior median is still pulled towards
typical values when the data are weak. The bias study fixes the true
profile on a grid of widths (Δv = 20–400 km/s, Δv_rand = 12 km/s, α = 0.1,
f = 0.25) and integrated S/N (3–40), simulates 20 spectra per cell and fits
them with the default priors and with flat priors (log-uniform Δv and
Δv_rand, uniform α and f).

![Conditional bias of W50 for fixed true profiles](img/bias_study.png)

* From S/N 8 upwards the default priors are unbiased to within 11 km/s,
  within 5 km/s from S/N 15, while the classical W50 of wide profiles is
  biased by up to +37 km/s.
* At S/N 5 the default priors pull a 430 km/s line down by 22 km/s and a
  48 km/s line up by 17 km/s; the flat priors pull the wide line down by
  137 km/s and a 230 km/s line by 42 km/s. A flat prior is therefore *not*
  the unbiased choice: the log-uniform width prior puts most of its mass
  at small widths, and the faint wings of a wide line are lost in the noise
  and the baseline whatever the prior.
* These cells carry the `prior_dominated` flag in up to 40 % of the fits:
  the width of a line at S/N ≤ 5 is a statement of the prior, not a
  measurement.

No choice of prior removes the conditional bias at low S/N, because there
is no information on the width there; the default priors keep it smallest
for the population they were calibrated on.

## Correlated noise

The SBC simulations with the noise smoothed by a Gaussian kernel of 0.8
channels that the fit does not know (ρ₁ = 0.67, ρ₂ = 0.21, ρ₃ = 0.03, as
after regridding; the line itself is not smoothed), plus 150 line-free
spectra, fitted with four noise models.

![Coverage and false positives for four noise models](img/correlated_noise.png)

| `noise.correlation` | ρ₁ used | false positives P > 0.5 / 0.9 | 68 % coverage | 95 % coverage |
|---|---|---|---|---|
| `white` (correlation ignored) | 0 | 85 % / 53 % | 0.38–0.59 | 0.66–0.90 |
| `hanning+white` (close guess) | 0.65 | 2.0 % / 0 % | 0.65–0.69 | 0.93–0.96 |
| `estimate` | 0.66 (0.63–0.70 per spectrum) | 4.0 % / 0.7 % | 0.63–0.67 | 0.90–0.94 |
| true correlation | 0.67 | 1.3 % / 0 % | 0.66–0.71 | 0.94–0.97 |

* Ignoring the correlation is not an option: most line-free spectra become
  detections and every interval is far too narrow.
* A close guess written into `model.response` is as good as the truth.
  `auto`, which takes the correlation from the smoothing in the response,
  stays the default.
* `estimate` measures ρ₁ to ±0.035 per spectrum but leaves the posteriors
  slightly overconfident (coverages 2–5 points low). Use it only when the
  smoothing is unknown.

## Channel widths

The runs above use 5 km/s channels. Surveys range from 1 to 15 km/s; the
pipeline was checked at both ends with simulated dwarf (W50 ≈ 25 km/s) and
spiral (W50 ≈ 260 km/s) profiles at S/N 10 with RFI spikes and a baseline:

| channels | dwarf | spiral | time per fit |
|---|---|---|---|
| 1 km/s, 1000 channels | W50 25 (22–29) for 28 | mean W50 error −3 km/s over 4 seeds | 13–19 s |
| 5 km/s, 512 channels | 26 (23–30) for 26 | +4 km/s | 8–10 s |
| 15 km/s, 170 channels | 26 (22–30) for 25, flagged `unresolved` | 268 (258–277) for 260 | 5–11 s |

Everything physical is in km/s (model, priors, derived widths), and the
channel response integrates the profile over channels of any width. The
longest run of outlier channels the RFI pre-stage may mask was once three
channels; at 15 km/s that masked 25 km/s dwarfs (3 of 3 lost), so it is now
`rfi.max_width` = 20 km/s (at least one channel). A line narrower than one
channel cannot be told from a spike at that resolution; the `unresolved`
flag (W50 below two channels) marks it. The cost of a fit scales with the
number of channels times the supersampling that the narrowest allowed
Δv_rand needs (7 at 15 km/s, 1 at 1 km/s).

## Real data: low-S/N ALFALFA sources

Ten sources from the ALFALFA α.40 catalogue (Haynes et al. 2011), drawn at
random: six with HIcode 1 (S/N 6.5–9) and four with HIcode 2 (S/N 5–6.5,
confirmed by an optical redshift). The spectra come from NED (baseline
subtracted, Hanning smoothed, 5.2 km/s channels) and are fitted in a
±1000 km/s window with the default configuration and
`model.response: boxcar+hanning`; widths and fluxes are converted to the cz
scale of the catalogue.

![hiprofile fits of ten low-S/N ALFALFA spectra against the catalogue](img/alfalfa_study.png)

| AGC | code | S/N | W50 ALFALFA | W50 hiprofile (16–84 %) | S ALFALFA | S hiprofile | note |
|---|---|---|---|---|---|---|---|
| 238702 | 1 | 7.1 | 139 ± 8 | 160 (139–186) | 0.92 ± 0.07 | 1.04 (0.90–1.19) | second component = window-edge artefact |
| 258145 | 1 | 6.9 | 210 ± 59 | 215 (192–240) | 0.85 ± 0.07 | 0.77 (0.63–0.90) | second component on a 3 mJy bump 700 km/s away |
| 102780 | 1 | 7.6 | 190 ± 5 | 186 (179–193) | 1.04 ± 0.07 | 1.07 (0.94–1.20) | |
| 219208 | 1 | 7.4 | 369 ± 18 | 386 (358–413) | 1.45 ± 0.11 | 1.26 (1.01–1.47) | |
| 258284 | 1 | 8.5 | 476 ± 32 | 509 (478–533) | 2.12 ± 0.13 | 1.96 (1.71–2.25) | |
| 257950 | 1 | 7.3 | 49 ± 6 | 60 (49–80) | 0.51 ± 0.05 | 0.61 (0.51–0.72) | |
| 232843 | 2 | 5.2 | 111 ± 11 | 389 (74–426) | 0.62 ± 0.08 | 1.24 (0.98–1.47) | bimodal: one horn or the full profile |
| 732650 | 2 | 6.2 | 63 ± 41 | 209 (199–219) | 0.38 ± 0.03 | 1.21 (1.11–1.32) | neighbour AGC 231950 in the beam |
| 260144 | 2 | 5.8 | 136 ± 50 | 322 (206–407) | 0.66 ± 0.06 | 1.02 (0.79–1.29) | neighbour AGC 260955 in the beam |
| 203516 | 2 | 6.0 | 81 ± 8 | 74 (62–88) | 0.60 ± 0.07 | 0.64 (0.53–0.73) | |

W50 in km/s, S in Jy km/s. The systemic velocities agree within 1.6σ of
the posterior errors (3–81 km/s) for all sources but the confused
AGC 732650.

* **HIcode 1:** all six widths and fluxes agree with the catalogue within
  the combined errors. In two spectra hiprofile prefers N = 2, but the
  catalogue source is one of the components and is measured correctly; the
  extra component sits on the rising edge of the extracted window or on a
  broad 3 mJy bump that is not in the catalogue.
* **HIcode 2:** one of four agrees. Two are confused: a catalogued
  neighbour 2.9′ and 3.8′ away (inside the 3.5′ Arecibo beam) overlaps in
  velocity, and the single-spectrum fit measures the blend, which the
  ALFALFA team separated with the maps. In the third (AGC 232843, S/N 5.2)
  the catalogue measured the bright horn only; the posterior is bimodal
  between that and a 390 km/s profile with the weaker emission on the far
  side, and its 16–84 % range covers both.
* **Solid-body fraction:** the f medians are 0.07–0.26, apart from the two
  blends of two galaxies (0.46, 0.52), whose summed profiles are genuinely
  dome-shaped.
* **Noise:** the automatic Hanning + white model gives ρ₁ = 0.64–0.66. With
  an exact Hanning model every fit was flagged with χ²_red up to 10⁴: real
  smoothed spectra have a little power at the Nyquist frequency, where the
  exact kernel has none, and the whitening amplified it by orders of
  magnitude; the estimated white part (0.5–4 % here) removes that.
