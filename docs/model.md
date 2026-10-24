# Model

## Line profile

One component is the profile of Stewart et al. (2014): a rotating outer ring
(double horn) and a solid-body inner disk (half ellipse), multiplied by an
asymmetry term and convolved with a Gaussian for random motions. With
$u = 2 (v - v_\mathrm{ctr}) / \Delta v$:

$$
s(v) = \left[ \frac{2 S (1-f)}{\pi \Delta v} \frac{1}{\sqrt{1-u^2}}
       + \frac{4 S f}{\pi \Delta v} \sqrt{1-u^2} \right] (1 + \alpha u)
       \ast \mathcal{N}(0, \Delta v_\mathrm{rand}^2)
$$

The six parameters are the centre $v_\mathrm{ctr}$, the width $\Delta v$, the
dispersion $\Delta v_\mathrm{rand}$, the asymmetry $\alpha \in [-1, 1]$, the
integrated flux $S$ and the solid-body fraction $f \in [0, 1]$.

## Linear and nonlinear parameters

The kernel does not return a finished curve but two unit-flux basis functions,
ring $R(v;\theta)$ and disk $D(v;\theta)$, already including asymmetry, Gaussian
and channel response. With $\theta = (v_\mathrm{ctr}, \Delta v, \Delta v_\mathrm{rand}, \alpha, f)$
the model is linear in the flux and the baseline:

$$
m(v) = S\,\big[(1-f) R(v;\theta) + f D(v;\theta)\big] + \sum_{k=0}^{K} c_k T_k(v).
$$

$S$ and the $c_k$ are marginalised analytically, and so is the baseline order
$K$ (see [Inference](inference.md)); the sampler sees the five nonlinear
parameters per component.

## Fourier-space evaluation

With $x = \Delta v\,\omega/2$:

$$
\hat R(\omega) = [J_0(x) - i\alpha J_1(x)]\,P(\omega),\qquad
\hat D(\omega) = \left[\frac{2 J_1(x)}{x} - 2 i\alpha \frac{J_2(x)}{x}\right] P(\omega)
$$

$$
P(\omega) = e^{-i\omega v_\mathrm{ctr}}\, e^{-\Delta v_\mathrm{rand}^2\omega^2/2}\, H(\omega)
$$

$H$ is the spectrometer response (below). The inverse transform runs on a
periodic fine grid (spacing $\Delta v_\mathrm{chan}/s$), zero-padded so that a
profile inside the band never wraps around, and is sampled at the channel
centres. Sampling every $s$-th point equals an inverse FFT of the spectrum folded
modulo the coarse grid length, so the NumPy kernel only evaluates the Bessel
functions where the Gaussian factor is above $10^{-13}$ and only runs a short FFT.

`J2/x` and `J1/x` are evaluated without cancellation for $\Delta v \to 0$. JAX has
no `j0`/`j1`; `hiprofile.model.bessel` implements them (power series below
$x = 12.5$, Hankel expansion above, accurate to $3\times10^{-12}$, finite
gradients everywhere). NumPy uses SciPy's Cephes routines.

The default supersampling makes the truncation of the Gaussian at the fine
Nyquist frequency smaller than $10^{-7}$ for the smallest allowed
$\Delta v_\mathrm{rand}$.

## Spectrometer response

`model.response` is a `+`-joined list:

| term | $H(\omega)$ | meaning |
|---|---|---|
| `none` | 1 | point sampling |
| `boxcar` | $\mathrm{sinc}(\omega\Delta v_\mathrm{chan}/2)$ | mean over the channel (default) |
| `hanning` | $\tfrac12 + \tfrac12\cos(\omega\Delta v_\mathrm{chan})$ | 3-tap Hanning smoothing, exact |
| `tukey(r)` | Tukey taper up to $\pi/\Delta v_\mathrm{chan}$ | lag window of a correlator |
| `kernel(w₀, w₁, …, w_q)` | $\sum_{|j| \le q} w_{|j|}\cos(j\,\omega\Delta v_\mathrm{chan})$ | any symmetric channel smoothing (normalised to sum 1), exact; `hanning` = `kernel(0.5, 0.25)` |

Example: `boxcar+hanning`. With `noise.correlation: auto` the channel
smoothing (`hanning`, `kernel`) also sets the noise correlation, e.g.
$\rho = (1, 2/3, 1/6)$ for Hanning, plus an estimated white part.

## Sampling parametrisation

* `direct` (default): $\Delta v$ uniform, $\log\Delta v_\mathrm{rand}$ uniform,
  $\alpha$ uniform in $[-1, 1]$.
* `effective`: $W = \Delta v + 2.35\,\Delta v_\mathrm{rand}$ uniform and
  $\varphi = \Delta v / W$ uniform, with
  $\Delta v_\mathrm{rand} = \sqrt{((1-\varphi) W / 2.35)^2 + \sigma_\mathrm{min}^2}$; this
  aligns the degeneracy ridge between $\Delta v$ and $\Delta v_\mathrm{rand}$ with
  a cube axis. The two parametrisations imply different priors.

Several profile components are ordered, $v_{\mathrm{ctr},1} < v_{\mathrm{ctr},2} < \dots$,
by mapping the unit cube to uniform order statistics, which removes label
switching without changing the evidence.

## Legacy compatibility

`hiprofile.model.legacy_profile` reproduces the legacy Cython `LineModel`
(point samples on an unpadded grid averaged in blocks, which shifts the profile
by $\Delta v_\mathrm{chan}(s-1)/(2s)$ and wraps it at the band edges). It shares
the Fourier coefficients with the new kernel and is only used to compare with
the golden files; it agrees to $10^{-11}$ relative to the peak.
