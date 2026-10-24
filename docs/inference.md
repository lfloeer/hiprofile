# Inference

## Marginal likelihood

For fixed $\theta$ the model is $m = M\beta$ with the design matrix
$M = [R, D, \dots, T_0, \dots, T_K]$. With the prior $\beta \sim \mathcal N(\mu, \Lambda)$
and the noise covariance $C$,

$$
\mathbf y \mid \theta \sim \mathcal N(M\mu,\; C + M\Lambda M^\top).
$$

With the whitening $L^{-1}$ ($C = LL^\top$) and the Woodbury identity only a
$p\times p$ system is solved, $p$ = number of amplitudes + K + 1:

$$
A = \Lambda^{-1} + \tilde M^\top\tilde M,\quad b = \tilde M^\top \tilde r,\quad
\ln L = -\tfrac12\left[\tilde r^\top\tilde r - b^\top A^{-1} b + \ln|C| + \ln|\Lambda| + \ln|A| + n\ln 2\pi\right]
$$

and $\beta\mid\theta, y \sim \mathcal N(\mu + A^{-1}b, A^{-1})$. The tests compare
this with the dense formula and with a numerical integral over $\beta$.

## Linear and nonlinear parameters

Each profile component has five nonlinear parameters, $\theta = (v_\mathrm{ctr},
\Delta v, \Delta v_\mathrm{rand}, \alpha, f)$, and one linear amplitude, the
integrated flux $S$. The kernel still returns the ring and disk basis
functions separately; the likelihood combines them into the unit-flux column
$(1-f)R + fD$. An earlier version kept $a_\mathrm{ring} = S(1-f)$ and
$a_\mathrm{disk} = Sf$ as two linear amplitudes. That is cheaper by one
sampling dimension but has two drawbacks: for narrow profiles ring and disk
are nearly identical and the two amplitudes become degenerate with large
anti-correlated values, and the implied prior on $f$ (the ratio of two
half-normal amplitudes) depends on $S$, flat for faint and peaked for bright
sources. With $f$ sampled, its prior is explicit and the same for every
source, and the degeneracy becomes a flat direction in $f$ that the prior
handles.

## Amplitude positivity

The amplitude $S$ of each component has a half-normal prior (scale
`priors.amplitude_scale`), i.e. a zero-mean Gaussian truncated to $S \ge 0$.
The truncation enters the likelihood exactly through the orthant probability
of the conditional Gaussian,

$$
p(y \mid \theta, K) = p_\mathrm{gauss}(y \mid \theta, K)\; P(S \ge 0 \mid \theta, y, K)\, 2^{q},
$$

with the normal CDF for one component and the bivariate normal CDF (Owen's T
function) for two. For three or more components the product of the marginal
probabilities is used in the sampling target and the remaining correction is
estimated by Monte Carlo and applied as importance weight
(`hiprofile.inference.amplitudes`). The fraction of Gaussian draws with a
negative amplitude is reported as `negative_fraction`; with the half-normal
prior in the target it is informational.

## Noise

Per-channel $\sigma$ comes from the input or a robust estimate (MAD of first
differences, corrected for the channel correlation; optionally restricted to
`noise.signal_free` ranges). The covariance between channels i and j is
$C_{ij} = \sigma_i\sigma_j\,\rho(|i-j|)$; masked channels are dropped, and
the banded covariance is factorised once (banded Cholesky for NumPy, dense
inverse factor for JAX). `priors.noise_scale` optionally fits a global
scale factor.

Real spectra are smoothed in the channel domain (spectrometer window,
Hanning, regridding to a velocity grid, averaging of Doppler-shifted
scans), which correlates neighbouring channels. `noise.correlation`:

| value | correlation |
|---|---|
| `auto` (default) | the smoothing in `model.response` (`hanning`, `kernel(...)`) plus an estimated white part; white without smoothing |
| `estimate` | measured on the residuals of the robust pre-fit: the noise is modelled as white noise smoothed by an unknown short symmetric kernel $h$ (7 taps), fitted so that $h \star h$ matches the sample correlation of the residuals over pairs of unmasked channels |
| `white`, `hanning`, `hanning-decimated`, `[ρ₁, ρ₂, …]` | fixed; a `+white` suffix adds an estimated white part |

Two details keep the noise model safe on real data. Its power spectrum is
kept above 0.5 % (`+white`) or 1 % (`estimate`) of the mean (a white floor); an exact Hanning model has zero
power at the Nyquist frequency, real spectra do not, and whitening with it
amplified that component by orders of magnitude (χ²_red up to 10⁴ on
ALFALFA spectra). And the estimate is positive definite by construction,
so masks and gaps cannot break the Cholesky factorisation. The fitted
correlation is stored as the metrics `noise_rho` and `noise_rho1`.
`estimate` costs one extra pre-fit iteration when the RFI pre-stage is off;
[Validation](validation.md#correlated-noise) compares the options on noise
smoothed by a kernel the fit does not know.

## Baseline

Chebyshev polynomials of order K = `k_min` … `k_max` with a Gaussian prior
(`baseline_scale`, or mean/covariance from signal-free neighbour spectra via
`hiprofile.likelihood.baseline.baseline_prior_from_spectra`). K is
compared by evidence with a flat prior.

## RFI

1. **Pre-stage** (`rfi.prefit`): EM fit of profile + baseline with the outlier
   mixture of Hogg, Bovy & Lang (2010),
   $L_j = (1-P_b)\mathcal N(y_j\mid m_j,\sigma_j) + P_b B(y_j)$, with a broad
   background $B$ that can be restricted to positive residuals. Output: outlier
   probabilities $q_j$ (stored and plotted).
2. **Main fit**: channels with $q_j$ above `rfi.threshold` are masked
   (`action: mask`) or their $\sigma$ inflated by $1/\sqrt{1-q_j}$
   (`action: inflate`). A single channel between two outliers is masked
   with them. Many outliers set the `rfi` flag.
3. **Extension**: `model.n_rfi` adds narrow Gaussian components (width
   `priors.rfi_sigma_channels`, 0.3–1 channel by default) to every model.

Two guards keep the pre-stage from eating signal. Its line model is not
allowed to be narrower than `rfi.min_sigma_channels` (2) channels, so a
bright spike cannot be absorbed as a "line" and hide its own centre from the
mask. And a run of more than `rfi.max_width_channels` (3) outlier channels
(gaps of up to two channels included) is not RFI but signal the
one-component pre-stage cannot explain, typically a second line, and stays
in the fit; broader RFI needs a larger value or `model.n_rfi`.
`hiprofile validate --stage detection` measures the completeness with and
without the pre-stage.

## Model comparison

For each spectrum the models N = 0 … `n_max` are evaluated; the baseline
order K = `k_min` … `k_max` is marginalised *inside* the likelihood,

$$
p(y \mid \theta) = \sum_K P(K)\, p(y \mid \theta, K), \qquad P(K) \text{ flat}.
$$

Because the Chebyshev columns are nested, $A_K$ is the leading principal block
of $A_{K_\max}$: one Cholesky factorisation gives $\ln|A_K|$ and
$b_K^\top A_K^{-1} b_K$ for every K as partial sums, so marginalising over K
costs about as much as a single order. Consequences:

* one sampler run per N (no K scan, no reweighting, no re-runs);
* the posterior is exactly marginalised over K; every sample carries
  $P(K \mid \theta, y)$, and the order stored with a sample (`k_samples`) is
  drawn from it;
* the per-(N, K) evidences follow from the run without extra sampling,
  $Z_{N,K} = Z_N\, \mathbb E_\mathrm{post}[P(K\mid\theta,y)] / P(K)$;
* an informative baseline prior from neighbour spectra (mean or covariance)
  fixes K = `k_max`, because the leading-block trick needs independent
  zero-mean coefficient priors.

N = 0 has no nonlinear parameters and its evidence is analytic for every K.
For N ≥ 1 a quick fit (MAP + Laplace + importance sampling) runs first; if
its evidence lies more than `sampler.gate_margin` (5) nats below a model with
fewer components, the sampler is skipped and the quick evidence kept
(`method = quick-gated`). Otherwise the sampler runs once on the K-marginal
likelihood (`method = marginal`).

A second, fainter line is a needle in the prior volume: random starts and
prior-drawn particles rarely place a component on it. Two warm starts take
care of that without touching the evidence. For N ≥ 2 the quick stage also
optimises the N − 1 solution with a new component on each of the strongest
features of the N − 1 residual (matched filters of 2 and 6 channels,
`sampler.n_seeds`). The SMC then draws `sampler.init_fraction` (25 %) of its
particles from the quick stage's importance proposal and the rest from the
prior; the initial importance weights prior/q₀ of this defensive mixture
keep ln Z exact and are bounded by 1/(1 − 0.25). Every second MCMC move of
the SMC is an independence proposal from the same mixture (exact
Metropolis–Hastings), so particles can jump into the narrow mode of a faint
second line at every temperature instead of having to random-walk there.
On pairs whose second line has S/N ≈ 6 (ln Z of N = 2 within a nat of N = 1)
this brought the SMC evidence from 2–5 nats below nautilus to within a nat.

Model probabilities use a Poisson prior on N (mean `poisson_rate`) and the
flat prior on K; the detection probability is $P(N\ge 1\mid y)$. Several
components are compared through evidences, not trans-dimensional sampling.

## Samplers

All backends implement `run(loglike, prior_transform, ndim) -> SamplerResult`
(samples, weights, ln Z ± error, runtime, number of likelihood calls, statistics).

| backend | role |
|---|---|
| `smc` | **default**: adaptive tempered SMC (1000 particles, random-walk moves), one batched likelihood call per step; with `--gpu` (JAX likelihood) the batch runs as one XLA call |
| `nautilus` | most precise ln Z, fewest likelihood calls, but ~10x slower here because its neural-network training dominates for a cheap likelihood; default `check_backend` |
| `dynesty` | second opinion, well-documented diagnostics |
| `ultranest` | robustness check, conservative ln Z errors |
| `quick` | MAP + Laplace + adaptive importance sampling; quick look and start values, may miss modes |

Modes: `--mode quick` (quick backend everywhere), `--mode full` (sampling of
all N and K), `--mode check` (additionally runs `check_backend` at K* and
flags $|\Delta\ln Z| > 3\sigma$).

Every nested sampler has a time / call budget (nautilus 600 s or 10⁶ calls,
dynesty and UltraNest 10⁶ calls); a fit that hits it gets the `sampler_budget` flag.

### Accuracy and speed

Measured on a 512-channel spectrum (S/N 15, first-order baseline), eight seeds per setting:

| backend | N = 1: ln Z bias / scatter / time | N = 2: ln Z bias / scatter / time |
|---|---|---|
| nautilus (reference) | – / 0.02 / 10 s | – / 0.03 / 40–46 s |
| smc, 1000 particles, 10 moves | −0.04 / 0.10 / 0.9 s | +0.25 / 0.23 / 3.2 s |
| smc, 1000 particles, 20 moves | 0.00 / 0.11 / 1.8 s | +0.16 / 0.11 / 6.6 s |
| quick | −0.03 / 0.02 / 0.3 s | −0.1 / 0.1 / 0.7 s (low ESS) |

dynesty and UltraNest lie 0.05–0.15 above nautilus for N = 2, so part of the SMC
offset is nautilus' own. The SMC error from the incremental-weight variance
underestimates the run-to-run scatter by about a factor 2 and is multiplied by
2. Offsets of a few tenths in ln Z change model probabilities by tens of
per cent at most; use `--backend nautilus` or `--mode check` where precise
evidences matter.

### Compiled kernel and likelihood

With numba installed (a default dependency), the ring/disk kernel and the
Woodbury algebra run as compiled loops (`hiprofile.model.fast`,
`hiprofile.likelihood.fast`): table-based Bessel functions (Estrin scheme),
recurrences for the phase factor and the Gaussian, direct scattering into the
folded spectrum, banded whitening and a small Cholesky solve per parameter
vector. They agree with the NumPy reference to 1e-12 (tests/test_fast.py).
One evaluation of the marginal likelihood (one component, 512 channels, K = 2)
takes about 19 µs in batches and 36 µs as a single call; the NumPy path takes
about 100 µs and 300 µs.

Complete fits with the default configuration (N ≤ 2, K = 0…3, RFI pre-stage), one process:

| spectrum | before (NumPy kernel, nautilus) | now (numba, smc) |
|---|---|---|
| pure noise | 15–55 s | 3.4 s |
| one component | 100–220 s | 4.9 s |
| two components | 100–250 s | 9 s |

`benchmarks/bench_kernel.py` and `benchmarks/bench_fit.py` reproduce the numbers.
