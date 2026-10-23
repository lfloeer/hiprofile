"""Quality metrics of a fit. Each returns plain floats so they fit into the catalogue."""

from __future__ import annotations

import numpy as np

from hiprofile.likelihood.noise import NoiseModel


def kish_ess(weights: np.ndarray) -> float:
    w = np.asarray(weights, float)
    s = w.sum()
    return float(s**2 / np.sum(w**2)) if s > 0 else 0.0


def chi2_reduced(resid_white: np.ndarray, n_par: int) -> float:
    nu = max(resid_white.size - n_par, 1)
    return float(np.sum(resid_white**2) / nu)


def posterior_predictive_p(
    flux_good: np.ndarray,
    models_good: np.ndarray,
    noise: NoiseModel,
    rng: np.random.Generator,
) -> float:
    """p = P(chi2(y_rep) >= chi2(y)) over posterior draws (models: (m, n_good)).

    y_rep = m + L eps with the full (possibly correlated) noise covariance.
    """
    if models_good.shape[0] == 0:
        return float("nan")
    t_obs = np.sum(noise.whiten(flux_good[None] - models_good) ** 2, axis=1)
    # In whitened space the replicated residual is just eps
    t_rep = np.sum(rng.standard_normal(models_good.shape) ** 2, axis=1)
    return float(np.mean(t_rep >= t_obs))


def wiggle_fraction(
    flux: np.ndarray, line_model: np.ndarray, sigma: np.ndarray, width: float, dv_chan: float
) -> tuple[float, float]:
    """Wiggle fraction J (Stewart et al. 2014, Eq. 8), baseline already subtracted.

    J^2 = sum_j y_j [(y_j - s_j)^2 - sigma_j^2] / (S <s>^2), S = sum_j s_j and
    <s> = S dv_chan / width the mean model flux across the line. J is the
    flux-weighted fractional rms of residuals beyond the noise; negative J^2
    (noise-dominated) returns 0.

    Returns (J, significance of J^2), the latter J^2 over its standard
    deviation under pure noise, sqrt(2 sum y_j^2 sigma_j^4) / (S <s>^2), so that
    noise-dominated J values of faint lines are not flagged.
    """
    s_tot = float(np.sum(line_model))
    if s_tot <= 0 or width <= 0:
        return float("nan"), float("nan")
    mean_s = s_tot * dv_chan / width
    norm = s_tot * mean_s**2
    j2 = float(np.sum(flux * ((flux - line_model) ** 2 - sigma**2)) / norm)
    j2_err = float(np.sqrt(2.0 * np.sum(flux**2 * sigma**4)) / norm)
    return (float(np.sqrt(j2)) if j2 > 0 else 0.0), j2 / max(j2_err, 1e-300)


def residual_acf(resid_white: np.ndarray, lags: tuple[int, ...] = (1, 2, 3)) -> np.ndarray:
    """Autocorrelation of whitened residuals, in units of its standard error 1/sqrt(n)."""
    r = resid_white - resid_white.mean()
    n = r.size
    denom = float(np.sum(r**2))
    if denom <= 0:
        return np.zeros(len(lags))
    return np.array([np.sum(r[:-lag] * r[lag:]) / denom * np.sqrt(n) for lag in lags])


def edge_mass(values: np.ndarray, weights: np.ndarray, low: float, high: float, n_bins: int = 20) -> float:
    """Largest posterior mass in the outermost bin at a finite prior edge."""
    w = weights / weights.sum()
    width = (high - low) / n_bins if np.isfinite(low) and np.isfinite(high) else None
    out = 0.0
    if width is None:
        return 0.0
    out = max(out, float(np.sum(w[values <= low + width])))
    out = max(out, float(np.sum(w[values >= high - width])))
    return out


def multimodality(
    u: np.ndarray, weights: np.ndarray, rng: np.random.Generator, n_iter: int = 30
) -> tuple[float, float]:
    """Weighted 2-means in unit-cube coordinates.

    Returns (weight of the smaller cluster, separation), with the separation the
    largest per-dimension distance of the centroids in units of the pooled
    within-cluster standard deviation.
    """
    if u.shape[0] < 10:
        return 0.0, 0.0
    w = weights / weights.sum()
    i0 = rng.choice(u.shape[0], p=w)
    d0 = np.sum((u - u[i0]) ** 2, axis=1)
    i1 = int(np.argmax(d0 * (w > 1e-12)))
    cents = np.stack([u[i0], u[i1]])
    lab = np.zeros(u.shape[0], int)
    for _ in range(n_iter):
        d = np.sum((u[:, None, :] - cents[None]) ** 2, axis=2)
        lab = np.argmin(d, axis=1)
        for k in range(2):
            m = lab == k
            if w[m].sum() > 0:
                cents[k] = np.sum(w[m, None] * u[m], axis=0) / w[m].sum()
    wk = np.array([w[lab == k].sum() for k in range(2)])
    if wk.min() <= 0:
        return 0.0, 0.0
    var = np.zeros(u.shape[1])
    for k in range(2):
        m = lab == k
        var += np.sum(w[m, None] * (u[m] - cents[k]) ** 2, axis=0)
    sep = float(np.max(np.abs(cents[0] - cents[1]) / np.sqrt(var + 1e-12)))
    return float(wk.min()), sep
