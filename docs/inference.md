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

## Amplitude positivity

The amplitudes have zero-mean Gaussian priors that are truncated to
$a \ge 0$ (half-normal, scale `priors.amplitude_scale`). The sampler works with the
untruncated marginal likelihood; afterwards every posterior sample draws
amplitudes from $\beta\mid\theta, y$, negative draws are rejected and the
weights multiplied by the acceptance probability. The evidence of the
truncated model is

$$
Z_\mathrm{trunc} = Z_\mathrm{gauss}\; \mathbb E_\mathrm{post}\!\left[P(a\ge 0 \mid \theta, y)\right] \big/ 2^{-n_\mathrm{amp}}.
$$

A large rejected fraction (`negative_fraction`) is flagged. For narrow profiles
ring and disk are nearly degenerate and the fraction is naturally high; the
evidence correction stays exact.

## Noise

Per-channel $\sigma$ comes from the input or a robust estimate (MAD of first
differences, corrected for the channel correlation; optionally restricted to
`noise.signal_free` ranges). Correlations: `white`, `hanning`
($\rho = 1, 2/3, 1/6$), `hanning-decimated` ($\rho = 1, 1/6$) or explicit. Masked
channels are dropped; the banded covariance is factorised once (banded
Cholesky for NumPy, dense inverse factor for JAX). `priors.noise_scale`
optionally fits a global scale factor.

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
   (`action: inflate`). Many outliers set the `rfi` flag.
3. **Extension**: `model.n_rfi` adds narrow Gaussian components (width
   `priors.rfi_sigma_channels`, 0.3–1 channel by default) to every model.

The pre-stage can explain narrow weak lines or single horns as outliers;
`hiprofile validate --stage detection` measures the completeness with and
without it.

## Model comparison

For each spectrum the models N = 0 … `n_max` and K = `k_min` … `k_max` are
evaluated. N = 0 has no nonlinear parameters and its evidence is analytic.
For N ≥ 1 the pipeline

1. runs quick fits (MAP + Laplace + importance sampling) for all K, warm-started
   from the previous K, and picks K*;
2. runs the sampler at K* and importance-reweights its samples to the other K
   (`sampler.baseline_evidence: reweight`); a K whose reweighted effective sample
   size falls below `reweight_min_ess` and whose evidence is competitive is
   re-run (`rerun` samples every K);
3. applies the positivity correction to every evidence;
4. builds the posterior for this N as the mixture of the per-K posteriors
   weighted with P(K | N, y), so that the baseline uncertainty propagates into
   all parameters (`k_samples` records the order of each sample);
5. skips nested sampling for an N whose quick evidence lies more than
   `sampler.gate_margin` (5) nats below a model with fewer components and keeps
   the quick evidence (`method = quick-gated`); such a model has a relative
   probability below e⁻⁵.

Model probabilities use a Poisson prior on N (mean `poisson_rate`) and a flat
prior on K; the detection probability is $P(N\ge 1\mid y)$. Several components
are compared through evidences, not trans-dimensional sampling.

## Samplers

All backends implement `run(loglike, prior_transform, ndim) -> SamplerResult`
(samples, weights, ln Z ± error, runtime, number of likelihood calls, statistics).

| backend | role |
|---|---|
| `nautilus` | default: fewest likelihood calls in low dimension |
| `dynesty` | second opinion, well-documented diagnostics |
| `ultranest` | robustness check, conservative ln Z errors |
| `smc` | adaptive tempered SMC, one batched call per step; with `--gpu` (JAX likelihood) the batch runs as one XLA call on CPU or GPU |
| `quick` | MAP + Laplace + adaptive importance sampling; quick look and start values, may miss modes |

Modes: `--mode quick` (quick backend everywhere), `--mode full` (nested
sampling, all N and K), `--mode check` (additionally runs `check_backend` at
K* and flags $|\Delta\ln Z| > 3\sigma$).

On a cheap likelihood (~100 µs per evaluation) nautilus spends most of its
time training its neural network; `n_networks = 1` is set by default. In
measurements on a 400-channel spectrum (N = 1) all backends agreed in ln Z
within 0.4 (nautilus 12.9k calls / 14 s, UltraNest 26k / 12 s, dynesty 43k / 28 s,
SMC 91k / 9 s, quick 6.9k / 1.4 s).
