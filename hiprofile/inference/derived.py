"""Derived quantities per posterior sample: S, f, W50, W20, v_sys, peak flux.

Widths are measured on the intrinsic profile (with random motions, without the
spectrometer response) on a fine grid: the distance between the outermost
crossings of 50 % (20 %) of the peak, linearly interpolated, as for the data
in Stewart et al. (2014). v_sys is the midpoint of the W50 crossings.
"""

from __future__ import annotations

import numpy as np

from hiprofile.model.kernel import SpectralGrid, profile_basis

DERIVED = ("S", "f", "W50", "W20", "v_sys", "peak")


def crossing_widths(v: np.ndarray, y: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Outermost crossings of ``level * max(y)`` per row, returns (low, high) velocities."""
    peak = y.max(axis=1, keepdims=True)
    thr = level * peak
    above = y >= thr
    n = y.shape[1]
    i_lo = np.argmax(above, axis=1)
    i_hi = n - 1 - np.argmax(above[:, ::-1], axis=1)
    rows = np.arange(y.shape[0])
    t = thr[:, 0]

    def interp(i_in: np.ndarray, i_out: np.ndarray) -> np.ndarray:
        valid = (i_out >= 0) & (i_out < n)
        i_out = np.clip(i_out, 0, n - 1)
        y_in, y_out = y[rows, i_in], y[rows, i_out]
        denom = np.where(y_in != y_out, y_in - y_out, 1.0)
        frac = np.clip((y_in - t) / denom, 0.0, 1.0)
        res = v[i_in] + frac * (v[i_out] - v[i_in])
        return np.where(valid, res, v[i_in])

    return interp(i_lo, i_lo - 1), interp(i_hi, i_hi + 1)


def fine_grid(grid: SpectralGrid, max_spacing: float = 1.0) -> SpectralGrid:
    sub = max(1, int(np.ceil(grid.dv / max_spacing)))
    n = grid.n_chan * sub
    return SpectralGrid(
        v0=grid.v0 - 0.5 * grid.dv * (1 - 1 / sub),
        dv=grid.dv / sub,
        n_chan=n,
        supersample=1,
        n_total=grid.n_total * sub,
        response="none",
    )


def profile_derived(
    grid: SpectralGrid, theta: np.ndarray, amps: np.ndarray, chunk: int = 256
) -> dict[str, np.ndarray]:
    """Derived quantities of one profile component.

    theta : (n, 4) (v_ctr, dv, dv_rand, alpha); amps : (n, 2) (a_ring, a_disk)
    """
    fg = fine_grid(grid)
    v = fg.velocity
    n = theta.shape[0]
    out = {k: np.full(n, np.nan) for k in DERIVED}
    s = amps.sum(axis=1)
    out["S"] = s
    out["f"] = np.where(s > 0, amps[:, 1] / np.where(s > 0, s, 1.0), np.nan)
    ok = np.isfinite(s) & (s > 0)
    idx = np.flatnonzero(ok)
    for start in range(0, idx.size, chunk):
        sel = idx[start : start + chunk]
        basis = profile_basis(fg, theta[sel])
        y = amps[sel, 0, None] * basis[:, 0] + amps[sel, 1, None] * basis[:, 1]
        lo50, hi50 = crossing_widths(v, y, 0.5)
        lo20, hi20 = crossing_widths(v, y, 0.2)
        out["W50"][sel] = hi50 - lo50
        out["W20"][sel] = hi20 - lo20
        out["v_sys"][sel] = 0.5 * (lo50 + hi50)
        out["peak"][sel] = y.max(axis=1)
    return out


def classic_widths(
    velocity: np.ndarray,
    flux: np.ndarray,
    window: tuple[float, float] | None = None,
    good: np.ndarray | None = None,
) -> dict[str, float]:
    """Classical W50/W20 and integrated flux measured directly on the data.

    Peak within ``window`` (default: whole band), outermost channels above
    50 % / 20 % of the peak, linear interpolation to the neighbouring channel.
    """
    v = np.asarray(velocity, float)
    y = np.asarray(flux, float).copy()
    if good is not None:
        y = np.where(good, y, 0.0)
    sel = np.ones(v.size, bool) if window is None else (v >= window[0]) & (v <= window[1])
    if not sel.any():
        return {"W50": np.nan, "W20": np.nan, "v_sys": np.nan, "S": np.nan, "peak": np.nan}
    ys = np.where(sel, y, -np.inf)
    lo50, hi50 = crossing_widths(v, np.where(sel, y, -1e300)[None], 0.5)
    lo20, hi20 = crossing_widths(v, np.where(sel, y, -1e300)[None], 0.2)
    dv = float(np.median(np.diff(v)))
    in_line = sel & (v >= lo20[0] - dv) & (v <= hi20[0] + dv)
    return {
        "W50": float(hi50[0] - lo50[0]),
        "W20": float(hi20[0] - lo20[0]),
        "v_sys": float(0.5 * (lo50[0] + hi50[0])),
        "S": float(np.sum(y[in_line]) * dv),
        "peak": float(np.max(ys)),
    }


# Springob et al. (2005, ApJS 160, 149), Table 2: lambda(SNR) for Hanning-smoothed
# spectra ("H"), as (lambda1, lambda2, lambda2', lambda3) for dv < 5, 5-11 (linear
# in dv) and > 11 km/s.
_SPRINGOB_H_LO = (0.005, -0.4685, 0.785, 0.395)
_SPRINGOB_H_HI = (0.227, -0.1523, 0.623, 0.533)


def springob_lambda(snr: float, dv: float) -> float:
    """Instrumental and noise broadening lambda (channels) of W_F50, Springob et al. eqs. 5-7."""
    if dv < 5.0:
        l1, l2, l2p, l3 = _SPRINGOB_H_LO
    elif dv > 11.0:
        l1, l2, l2p, l3 = _SPRINGOB_H_HI
    else:  # linear in dv between 5 and 11 km/s
        l1, l2, l2p, l3 = 0.037 * dv - 0.18, 0.0527 * dv - 0.732, -0.027 * dv + 0.92, 0.023 * dv + 0.28
    x = np.log10(max(snr, 1e-3))
    if x < 0.6:
        return float(l1)
    if x > 1.1:
        return float(l3)
    return float(l2 + l2p * x)


def _edge_fit(v: np.ndarray, y: np.ndarray, i_peak: int, step: int, level: float) -> np.ndarray | None:
    """Straight line through one flank: channels from the first below 85 % to the first below
    15 % of ``level`` searching outward from the horn; returns (intercept, slope) or None."""
    n = y.size
    i, i85, i15 = i_peak, None, None
    while 0 <= i < n:
        if i85 is None and y[i] < 0.85 * level:
            i85 = i
        if y[i] < 0.15 * level:
            i15 = i
            break
        i += step
    if i85 is None or i15 is None:
        return None
    lo, hi = sorted((i85, i15))
    if hi - lo < 1:
        lo, hi = max(lo - 1, 0), min(hi + 1, n - 1)
    sel = slice(lo, hi + 1)
    if hi - lo < 1:
        return None
    b, a = np.polyfit(v[sel], y[sel], 1)
    # the flank must fall away from the horn (noise can tilt a short fit the other way)
    return np.array([a, b]) if b * step < 0 else None


def springob_widths(
    velocity: np.ndarray,
    flux: np.ndarray,
    rms: float,
    window: tuple[float, float],
    center: float,
    good: np.ndarray | None = None,
    hanning: bool = True,
    z: float = 0.0,
) -> dict[str, float]:
    """W_F50 of Springob et al. (2005) and the same edge fits at 20 %.

    Per side of the profile (split at ``center``) the horn is the highest channel
    in the window, f_p its flux. Searching outward from the horn, the first
    channels below 85 % and 15 % of f_p - rms bound a straight-line fit to the
    flank; the flank velocity is where the line reaches 50 % (20 %) of
    f_p - rms. W_F50 = V_u - V_l. The instrumental and noise correction
    Delta_s = 2 dv lambda(SNR) with SNR = (f_p - rms) / rms (Table 2, Hanning)
    is subtracted when the spectrum is Hanning smoothed, followed by 1 + z.
    The per-horn levels make W_F50 differ from 50 % of the global peak for
    asymmetric profiles.
    Springob et al. calibrated lambda for W_F50 only; ``W20`` applies the same
    Delta_s and is an extension of the method. NaN when a flank cannot be fitted
    (too few channels, a fit rising away from the horn, or a crossing outside
    the window).
    """
    v = np.asarray(velocity, float)
    y = np.asarray(flux, float).copy()
    if good is not None:
        y = np.where(good, y, np.nan)
    sel = (v >= window[0]) & (v <= window[1]) & np.isfinite(y)
    nan = {"W50": np.nan, "W20": np.nan, "v_sys": np.nan, "snr": np.nan, "W50_raw": np.nan}
    if sel.sum() < 6:
        return nan
    idx = np.flatnonzero(sel)
    vv, yy = v[idx], y[idx]
    left, right = np.flatnonzero(vv < center), np.flatnonzero(vv >= center)
    if left.size < 2 or right.size < 2:
        return nan
    i_l, i_r = int(left[np.argmax(yy[left])]), int(right[np.argmax(yy[right])])
    out: dict[str, float] = {}
    edges = {}
    for side, i_pk, step in (("l", i_l, -1), ("r", i_r, +1)):
        level = float(yy[i_pk]) - rms
        fit = _edge_fit(vv, yy, i_pk, step, level) if level > 0 else None
        if fit is None:
            return nan
        edges[side] = (fit, level)
    dv = float(np.median(np.diff(v)))
    snr = (max(edges["l"][1], edges["r"][1])) / rms
    lam = springob_lambda(snr, dv) if hanning else 0.0
    for frac, key in ((0.5, "W50"), (0.2, "W20")):
        (al, bl), lev_l = edges["l"]
        (ar, br), lev_r = edges["r"]
        v_l = (frac * lev_l - al) / bl
        v_r = (frac * lev_r - ar) / br
        raw = v_r - v_l
        if not (vv[0] <= v_l < v_r <= vv[-1]):  # crossing outside the window: flat, noisy fit
            return nan
        out[key] = float((raw - 2.0 * dv * lam) / (1.0 + z))
        if key == "W50":
            out["W50_raw"] = float(raw)
            out["v_sys"] = float(0.5 * (v_l + v_r))
    out["snr"] = float(snr)
    return out


def classic_measurements(
    velocity: np.ndarray,
    line_flux: np.ndarray,
    rms: float,
    v_sys: float,
    w20: float,
    good: np.ndarray | None = None,
    hanning: bool = True,
) -> dict[str, float]:
    """Both classical measurements on the baseline-subtracted spectrum.

    The window is centred on ``v_sys`` with half-width W20/2 + 25 % + 2 channels,
    as a by-eye window would be chosen. Keys: the ``classic_widths`` ones and the
    ``springob_widths`` ones with suffix ``_springob``.
    """
    dv = float(np.median(np.abs(np.diff(velocity))))
    half = 0.5 * w20 * 1.25 + 2 * dv
    window = (v_sys - half, v_sys + half)
    out = classic_widths(velocity, line_flux, window=window, good=good)
    sp = springob_widths(velocity, line_flux, rms, window, v_sys, good=good, hanning=hanning)
    out.update({f"{k}_springob": v for k, v in sp.items()})
    return out
