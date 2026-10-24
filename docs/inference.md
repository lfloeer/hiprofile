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
