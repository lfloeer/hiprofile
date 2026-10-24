# Validation

All stages run with `hiprofile validate` (set-up of each stage in
`hiprofile/validate/stages.py`); the figures come from
`python docs/make_figures.py validation <run directory>` and
`python docs/make_figures.py bias <bias-study directory>`. The results below
use the default configuration (full mode, SMC sampler, N ≤ 2, K ≤ 3, RFI
pre-stage) on a 4-core machine: `--seed 11` for the 1000 SBC simulations,
`--seed 21` for the 200 detection simulations, `--seed 31` for the 4 × 100
misspecified spectra and `benchmarks/bias_study.py --n 20` (seed 5) for the
bias study.

## Stage 1: simulation-based calibration

`hiprofile validate --stage sbc --n 1000 --seed 11 --jobs 4`

Every simulation is drawn from the fit priors: the five shape parameters,
the flux (half-normal with the default `amplitude_scale`), and a Chebyshev
baseline of order K ~ U{0, …, 3} with coefficients from the baseline prior;
white noise with σ = 0.01 Jy on 512 channels of 5 km/s. The solid-body
fraction comes from its default prior Beta(1, 3). The fit uses the same
priors with the order marginalised (`k_min = 0`, `k_max = 3`). The rank of
the truth among 99 posterior draws must then be uniform, and the posterior
intervals must cover the truth at their nominal rate.

| parameter | rank p-value | 68 % coverage | 95 % coverage |
|---|---|---|---|
| v_ctr | 0.77 | 0.69 | 0.96 |
| Δv | 0.52 | 0.68 | 0.94 |
| Δv_rand | 0.43 | 0.69 | 0.94 |
| α | 0.22 | 0.66 | 0.95 |
| S | 0.63 | 0.67 | 0.95 |
| f | 0.31 | 0.70 | 0.95 |
| W50 | 0.88 | 0.70 | 0.95 |
| W20 | 0.99 | 0.67 | 0.94 |

![SBC: rank histograms, interval coverage and baseline-order calibration](img/sbc.png)

All rank histograms are flat (no U shape, slope or peak; all p > 0.2), and
the 68 % and 95 % coverages lie inside the ±2σ bands for 1000 simulations
(0.65–0.71 and 0.94–0.96). The simulations that are not detected (faint
draws of the half-normal flux prior, about a quarter) take part in the
rank statistics like every other simulation.

**Baseline order.** The most probable K is the true one in 93 % of the
simulations and P(K_true | y) averages 0.88; the reliability curve (right
panel) follows the diagonal, so the order probabilities in the catalogue
mean what they say. The confusion matrix (rows: true K, columns: most
probable K) is one-sided:

```
  266      2      0      0
   14    229      0      0
    5     13    222      0
    2      6     26    215
```

A true K = 3 whose highest coefficients happen to be small is read as K = 2
(26 of 249), which is the right answer for terms the data cannot see; the
reverse, a spurious high order, almost never happens.

**Flags.** 11.1 % of the fits carry a flag: `prior_edge` 5.7 % (Δv_rand at
its 4 km/s bound for faint, narrow lines), `prior_dominated` 3.1 %,
`wiggle` 2.1 %, `multimodal` 1.1 %, `ppp` 1.0 %. The simulations
include the faint end of the flux prior, where the width is not measurable.

The same simulations measured against the truth (detected ones, N = 1):

| quantity | Bayes bias | Bayes scatter | classical bias | classical scatter |
|---|---|---|---|---|
| W50 [km/s] | 0.4 | 9.8 | 0.2 | 18.6 |
| W20 [km/s] | 0.4 | 11.6 | 26.0 | 35.1 |
| S [Jy km/s] | 0.00 | 0.36 | 0.03 | 0.40 |
| v_sys [km/s] | −0.2 | 4.6 | 0.1 | 8.2 |

![Line widths: Bayesian fit against the classical measurement](img/bayes_vs_classic.png)

The posterior medians are unbiased, and their scatter is half (W50, v_sys)
to a third (W20) of that of the classical measurement on the same data. The
classical W20 is biased high at low S/N because noise peaks raise the 20 %
crossing points.

Example posterior (corner plot of a low-S/N SBC spectrum with a bimodal
centre):

![corner plot](img/corner_example.png)

## Stage 2: detection and confusion

`hiprofile validate --stage detection --n 200 --seed 21 --jobs 4`

(Stages 2 and 3 ran before the solid-body prior changed from uniform to
Beta(1, 3); they test detection and flags, which hardly depend on it.)

200 spectra with N = 0, 1, 2 (probabilities 0.4, 0.4, 0.2), integrated
S/N 2–40 per component, a first-order baseline and single-channel RFI
spikes (on average 0.005 per channel, i.e. 2–3 per spectrum), fitted twice:
with and without the RFI pre-stage.

![Detection: completeness, reliability and confusion matrix](img/detection.png)

| | with RFI pre-stage | without |
|---|---|---|
| false positives, P(detect) > 0.5 | 1 of 77 (1.3 %) | 83 % |
| false positives, P(detect) > 0.9 | 0 | 83 % |
| completeness S/N 3–5 / 5–8 / ≥ 8 | 0.58 / 0.94 / 1.00 | 0.89 / 1.00 / 1.00 |
| flagged | 8 % | 87 % |

Confusion matrix with the pre-stage (rows: true N, columns: most probable N):

```
   76      1      0
   17     66      0
    2     19     19
```

* **N = 0 and N = 1** are separated cleanly. The completeness with the
  pre-stage equals that of the SBC, which has no RFI at all (0.45 at
  S/N 3–5, 0.93 at 5–8, 1.00 above 8), so the pre-stage costs no
  completeness. Without it, RFI spikes are fitted as lines: 83 % false
  positives, 64 % of the fits flagged `unresolved`.
* **The one false positive** (P(detect) = 0.58) is a 520 km/s wide, shallow
  "line" that competes with the baseline; it carries the `prior_dominated`
  and `wiggle` flags.
* **N = 2.** Of the 19 pairs with both components at S/N > 6 and separated
  by more than their width, 15 are fitted as N = 2; the four others have
  P(N = 2) between 0 and 0.43 with evidences within 1–3 nats of N = 1, so
  the data do not decide them. Pairs with a second component below S/N 6
  (20 of 40) are mostly fitted as N = 1, as they should be: a second line at
  S/N 3 is not detectable on its own either.

An earlier version missed bright pairs: the one-component RFI pre-stage
masked the second line as a run of outliers, and neither the quick stage nor
the SMC found the narrow two-line mode. The pre-stage now keeps runs longer
than three channels, and both samplers are warm-started from residual
features (see [Inference](inference.md#model-comparison)).

## Stage 3: misspecification

`hiprofile validate --stage misspec --n 100 --seed 31 --jobs 4`

Four sets of 100 one-component spectra at S/N 5–80: model profiles
(`clean`), profiles not generated by the model (`template`: tilted-ring
galaxies with a rising rotation curve, inclinations of 30–85°, lopsidedness
up to 30 % and central HI holes, integrated on a fine grid), model profiles on a sinusoidal baseline
that no Chebyshev polynomial up to K = 3 describes (`sine_baseline`), and
model profiles with RFI fitted without the pre-stage (`rfi_unmasked`).

![Flag rates on misspecified spectra](img/misspec_flags.png)

| set | flagged | W50 bias / scatter [km/s] | 68 % coverage of W50 |
|---|---|---|---|
| clean | 9 % | 0.9 / 4.8 | 0.73 |
| template | 13 % | 0.5 / 6.8 | 0.72 |
| sine baseline | 99 % | 0.5 / 27.9 | 0.27 |
| RFI unmasked | 100 % | −90 / 115 | 0.14 |

The model describes these galaxy-like profiles well enough that the widths stay
unbiased and calibrated. When the model is wrong (a sine baseline, unmasked
RFI) the posteriors are overconfident, as any posterior of a wrong model is,
but the flags catch it: `ppp` and `residual_acf` mark almost every
sine-baseline fit, `ppp`, `prior_edge`, `unresolved` and `wiggle` mark the
RFI fits. Flagged fits should not enter a science sample without a look at
the plot.

## Conditional bias: default against flat priors

`python benchmarks/bias_study.py --out DIR --n 20 --priors default|flat`

The SBC shows that the posteriors are calibrated *on average over the
prior*. For a given galaxy the posterior median is still pulled towards
typical values when the data are weak. The bias study fixes the true profile
on a grid of widths (Δv = 20–400 km/s, Δv_rand = 12 km/s, α = 0.1,
f = 0.25) and integrated S/N (3–40), simulates 20 spectra per cell and fits
them with the default (THINGS) priors, f ~ Beta(1, 3) included, and with
flat priors (log-uniform Δv and Δv_rand, uniform α and f).

![Conditional bias of W50 for fixed true profiles](img/bias_study.png)

* From S/N 8 upwards both prior sets are unbiased to within about 10 km/s
  (default priors: |bias| ≤ 10 km/s in every cell, ≤ 5 km/s from S/N 15),
  while the classical W50 of wide profiles is biased by up to +37 km/s.
* At S/N 5 the default priors pull a 430 km/s line down by 20 km/s and a
  48 km/s line up by 18 km/s; the flat priors pull the wide line down by
  130 km/s. A flat prior is therefore *not* the unbiased choice: the
  log-uniform width prior puts most of its mass at small widths, and the
  faint wings of a wide line are lost in the noise and the baseline whatever
  the prior. With the THINGS priors the shrinkage is as small or smaller
  everywhere except for the narrowest lines. The Beta(1, 3) prior on f
  halved the pull on the 430 km/s line (−45 km/s with a uniform f), because
  a dome-shaped solid-body profile no longer competes with the faint
  double horn.
* These cells carry the `prior_dominated` flag in up to half of the fits
  (constraint on Δv below 0.3): the width of a line at S/N ≤ 5 is a
  statement of the prior, not a measurement.

No choice of prior removes the conditional bias at low S/N, because there is
no information on the width there; the default priors minimise it for the
galaxy population they were calibrated on and are kept.

## Real data: low-S/N ALFALFA sources

`python benchmarks/alfalfa_study.py --out DIR --figure docs/img/alfalfa_study.png`

Ten sources from the ALFALFA α.40 catalogue (Haynes et al. 2011), drawn at
random: six with HIcode 1 (S/N 6.5–9) and four with HIcode 2 (S/N 5–6.5,
detections confirmed by an optical redshift). The spectra come from NED
(baseline subtracted, Hanning smoothed, 5.2 km/s channels) and are fitted
in a ±1000 km/s window with the default configuration and
`model.response: boxcar+hanning`; widths and fluxes are converted to the
cz scale of the catalogue.

![hiprofile fits of ten low-S/N ALFALFA spectra against the catalogue](img/alfalfa_study.png)

| AGC | code | S/N | W50 ALFALFA | W50 hiprofile (16–84 %) | S ALFALFA | S hiprofile | note |
|---|---|---|---|---|---|---|---|
| 238702 | 1 | 7.1 | 139 ± 8 | 158 (139–184) | 0.92 ± 0.07 | 1.04 (0.90–1.19) | second component = window-edge artefact |
| 258145 | 1 | 6.9 | 210 ± 59 | 211 (183–269) | 0.85 ± 0.07 | 0.73 (0.56–1.11) | |
| 102780 | 1 | 7.6 | 190 ± 5 | 187 (179–194) | 1.04 ± 0.07 | 1.07 (0.94–1.20) | |
| 219208 | 1 | 7.4 | 369 ± 18 | 388 (360–413) | 1.45 ± 0.11 | 1.25 (1.04–1.49) | |
| 258284 | 1 | 8.5 | 476 ± 32 | 508 (476–535) | 2.12 ± 0.13 | 1.97 (1.69–2.25) | |
| 257950 | 1 | 7.3 | 49 ± 6 | 62 (49–86) | 0.51 ± 0.05 | 0.62 (0.52–0.75) | |
| 232843 | 2 | 5.2 | 111 ± 11 | 399 (74–427) | 0.62 ± 0.08 | 1.27 (1.01–1.48) | bimodal: one horn or the full profile |
| 732650 | 2 | 6.2 | 63 ± 41 | 209 (199–220) | 0.38 ± 0.03 | 1.20 (1.10–1.31) | neighbour AGC 231950 in the beam |
| 260144 | 2 | 5.8 | 136 ± 50 | 321 (215–397) | 0.66 ± 0.06 | 1.02 (0.79–1.28) | neighbour AGC 260955 in the beam |
| 203516 | 2 | 6.0 | 81 ± 8 | 75 (62–88) | 0.60 ± 0.07 | 0.63 (0.53–0.74) | |

W50 in km/s, S in Jy km/s. The systemic velocities agree within 1.7σ of the
posterior errors (3–81 km/s) for all sources but the confused AGC 732650.

The fits use the default solid-body prior f ~ Beta(1, 3). With the earlier
uniform prior the f medians of six sources lay at 0.45–0.76, including
0.76 for a 350 km/s wide line; with Beta(1, 3) they are 0.08–0.27, apart
from the two blends of two galaxies (0.46, 0.52), whose summed profiles are
genuinely dome-shaped. Widths and fluxes moved by less than their errors.

* **HIcode 1:** all six widths and fluxes agree with the catalogue within
  the combined errors. In AGC 238702 hiprofile prefers N = 2, but the
  catalogue source is one of the components and is measured correctly; the
  extra component sits on the rising edge of the extracted window. With the
  earlier uniform f prior AGC 258145 also got a second component, on a broad
  3 mJy bump 700 km/s away; with Beta(1, 3) it is fitted as N = 1.
* **HIcode 2:** one of four agrees. Two of the three others are confused: a
  catalogued neighbour 2.9′ and 3.8′ away (inside the 3.5′ Arecibo beam)
  overlaps in velocity, and the single-spectrum fit measures the blend,
  which the ALFALFA team separated using the maps. In the third
  (AGC 232843, S/N 5.2) the catalogue measured the bright horn only; the
  posterior is bimodal between that and a 360 km/s profile that includes
  the weaker emission on the far side, and its 16–84 % range (78–427 km/s)
  covers both.

The first run of this study found every fit flagged with χ²_red up to 10⁴:
real Hanning-smoothed spectra are not exactly Hanning, and the exact
Hanning covariance has zero variance at the Nyquist frequency, where the
data have a little. Whitening amplified that component by orders of
magnitude. Since then a `hanning` response uses `hanning+white` noise, with
the white fraction estimated per spectrum from fourth and first differences
(0.5–4 % in these spectra, half of them at the 0.5 % floor); see
[Configuration](configuration.md).

## Correlated noise

`python benchmarks/correlated_noise.py --out DIR --n 300 --n-noise 150`

The SBC simulations (parameters from the priors, baseline order 0–3,
σ = 0.01 Jy given to the fit) with the noise smoothed by a Gaussian kernel
of 0.8 channels that the fit does not know (ρ₁ = 0.67, ρ₂ = 0.21,
ρ₃ = 0.03, as after regridding), plus 150 line-free spectra, fitted with four
noise models.

![Coverage and false positives for four noise models](img/correlated_noise.png)

| `noise.correlation` | ρ₁ used | false positives P > 0.5 / 0.9 | 68 % coverage | 95 % coverage | worst rank p |
|---|---|---|---|---|---|
| `white` (correlation ignored) | 0 | 85 % / 54 % | 0.39–0.60 | 0.69–0.89 | 10⁻⁵³ |
| `hanning+white` (plausible guess) | 0.65 | 2.0 % / 0 % | 0.65–0.71 | 0.94–0.97 | 0.15 |
| `estimate` | 0.66 (0.63–0.70 per spectrum) | 4.7 % / 1.3 % | 0.62–0.66 | 0.89–0.93 | 0.0009 |
| true correlation | 0.67 | 1.3 % / 0 % | 0.67–0.71 | 0.94–0.96 | 0.13 |

* Ignoring the correlation is not an option: 85 % of the line-free spectra
  become detections, and every interval is far too narrow.
* A plausible guess that is close (Hanning for a Gaussian of 0.8 channels)
  is as good as the truth. `auto`, which takes the correlation from the
  smoothing in `model.response`, therefore stays the default.
* `estimate` recovers ρ₁ to ±0.035 per spectrum and ρ₂ to about 0.02 low,
  which leaves the posteriors slightly overconfident: coverages 2–5 points
  below nominal and 4.7 % false positives at P > 0.5 (1.3 % at P > 0.9).
  It is the right choice when the smoothing is unknown (not documented, or
  changed by regridding), and the wrong one when a known kernel can be
  written into `model.response`.
