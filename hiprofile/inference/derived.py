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
