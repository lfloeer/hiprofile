"""Per-spectrum plots: data + posterior band, corner plot, evidences, sampler diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hiprofile.inference.results import FitResult, weighted_quantile
from hiprofile.plots import style

style.apply()
import matplotlib.pyplot as plt  # noqa: E402


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.+" else "_" for c in name)


def plot_spectrum(res: FitResult, path: str | Path | None = None):
    v, y = res.velocity, res.flux
    show_q = bool(res.rfi_q.size and np.any(res.rfi_q > 0.05))
    if show_q:
        fig, (ax, axr, axq) = plt.subplots(
            3, 1, figsize=(8, 5.8), sharex=True, gridspec_kw={"height_ratios": [3, 1, 0.6], "hspace": 0.08}
        )
    else:
        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}
        )
    good = ~res.mask
    ax.step(v, np.where(good, y, np.nan), where="mid", color=style.INK, lw=0.8, label="data")
    if res.mask.any():
        ax.step(v, np.where(res.mask, y, np.nan), where="mid", color=style.MUTED, lw=0.8, label="masked")
    if res.model_band.size:
        b = res.model_band
        ax.fill_between(v, b[0], b[4], color=style.SERIES[0], alpha=0.15, lw=0, label="95 %")
        ax.fill_between(v, b[1], b[3], color=style.SERIES[0], alpha=0.3, lw=0, label="68 %")
        ax.plot(v, b[2], color=style.SERIES[0], lw=1.5, label="model (median)")
        if res.baseline_median.size:
            ax.plot(v, res.baseline_median, color=style.INK_2, lw=1, ls="--", label="baseline")
        for i, comp in enumerate(res.components_median):
            ax.plot(
                v,
                comp + res.baseline_median,
                color=style.SERIES[1 + i % 6],
                lw=1,
                ls="--" if len(res.components_median) == 1 else "-",
                label=f"component {i + 1}",
            )
        sig = np.where(np.isfinite(res.sigma) & (res.sigma > 0), res.sigma, np.nan)
        r = (y - b[2]) / sig
        axr.step(v, np.where(good, r, np.nan), where="mid", color=style.INK, lw=0.7)
        axr.axhline(0, color=style.MUTED, lw=0.8)
        for lev in (-3, 3):
            axr.axhline(lev, color=style.MUTED, lw=0.6, ls=":")
    if show_q:
        # outlier probability of the RFI pre-stage in its own panel (no second y axis)
        axq.fill_between(v, 0, res.rfi_q, step="mid", color=style.SERIES[6], alpha=0.6, lw=0)
        axq.set_ylim(0, 1)
        axq.set_ylabel("q (RFI)")
        axq.set_xlabel("velocity [km/s]")
    axr.set_ylabel("residual / sigma")
    if not show_q:
        axr.set_xlabel("velocity [km/s]")
    ax.set_ylabel("flux density")
    title = f"{res.spectrum_id}   P(detect) = {res.p_detect:.3f}   N = {res.best_n}, K = {res.best_k}"
    flags = [k for k, val in res.flags.items() if val]
    if flags:
        title += "   flags: " + ", ".join(flags)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper right", ncol=2)
    return _finish(fig, path)


def _finish(fig, path):
    if path is not None:
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return Path(path)
    return fig


def _weighted_hist2d(ax, x, y, w, bins=24):
    from scipy.ndimage import gaussian_filter

    h, xe, ye = np.histogram2d(x, y, bins=bins, weights=w)
    h = gaussian_filter(h.T, 1.0)
    order = np.sort(h.ravel())[::-1]
    c = np.cumsum(order) / order.sum()
    levels = sorted({order[np.searchsorted(c, q)] for q in (0.95, 0.68)})
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    ax.imshow(h, origin="lower", aspect="auto", extent=(xe[0], xe[-1], ye[0], ye[-1]), cmap="Blues")
    if len(levels) == 2 and levels[0] < levels[1]:
        ax.contour(xc, yc, h, levels=levels, colors=[style.SERIES[0]], linewidths=0.8)


def plot_corner(res: FitResult, n: int | None = None, path: str | Path | None = None, max_params: int = 9):
    n = n or max(res.best_n, 1)
    if n not in res.posteriors:
        return None
    post = res.posteriors[n]
    cols = {name: post.theta[:, i] for i, name in enumerate(post.names) if not name.startswith("rfi")}
    for key in ("W50_1", "W20_1", "S_1"):
        if key in post.derived:
            cols[key] = post.derived[key]
    names = list(cols)[:max_params]
    w = post.weights
    ok = np.all([np.isfinite(cols[k]) for k in names], axis=0)
    data = np.array([cols[k][ok] for k in names])
    w = w[ok] / w[ok].sum()
    d = len(names)
    fig, axes = plt.subplots(d, d, figsize=(1.6 * d + 1, 1.6 * d + 1))
    axes = np.atleast_2d(axes)
    for i in range(d):
        lo, hi = weighted_quantile(data[i], w, [0.005, 0.995])
        if hi <= lo:
            lo, hi = lo - 1, hi + 1
        for j in range(d):
            ax = axes[i, j]
            ax.grid(False)
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(
                    data[i],
                    bins=30,
                    range=(lo, hi),
                    weights=w,
                    color=style.SERIES[0],
                    histtype="stepfilled",
                    alpha=0.6,
                )
                q = weighted_quantile(data[i], w, [0.16, 0.5, 0.84])
                for qq in q:
                    ax.axvline(qq, color=style.INK_2, lw=0.7, ls="--")
                ax.set_title(f"{names[i]}\n{q[1]:.4g} (+{q[2] - q[1]:.2g} / -{q[1] - q[0]:.2g})", fontsize=7)
                ax.set_yticks([])
                ax.set_xlim(lo, hi)
            else:
                lo_j, hi_j = weighted_quantile(data[j], w, [0.005, 0.995])
                sel = (data[j] >= lo_j) & (data[j] <= hi_j) & (data[i] >= lo) & (data[i] <= hi)
                if sel.sum() > 10:
                    _weighted_hist2d(ax, data[j][sel], data[i][sel], w[sel])
            if i < d - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(names[j], fontsize=7)
            if j > 0 or i == 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(names[i], fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"{res.spectrum_id}: posterior N = {n}, K = {post.k}", x=0.02, ha="left")
    return _finish(fig, path)


def plot_evidence(res: FitResult, path: str | Path | None = None):
    evs = res.evidences
    if not evs:
        return None
    ns = sorted({e.n for e in evs})
    ks = sorted({e.k for e in evs})
    ref = max(e.log_z for e in evs)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    width = 0.8 / len(ks)
    for i, k in enumerate(ks):
        vals = [next((e.log_z - ref for e in evs if e.n == n and e.k == k), np.nan) for n in ns]
        errs = [next((e.log_z_err for e in evs if e.n == n and e.k == k), 0.0) for n in ns]
        x = np.arange(len(ns)) + (i - (len(ks) - 1) / 2) * width
        ax.bar(
            x,
            np.maximum(vals, -50),
            width=width * 0.9,
            color=style.SERIES[i % 8],
            label=f"K = {k}",
            yerr=errs,
        )
    ax.set_xticks(np.arange(len(ns)), [f"N = {n}\nP = {res.p_n.get(n, np.nan):.3g}" for n in ns])
    ax.set_ylabel("ln Z - max ln Z (clipped at -50)")
    ax.axhline(0, color=style.MUTED, lw=0.8)
    ax.legend(ncol=len(ks))
    ax.set_title(f"{res.spectrum_id}: evidences", loc="left")
    return _finish(fig, path)


def plot_sampler(res: FitResult, n: int | None = None, path: str | Path | None = None):
    n = n or max(res.best_n, 1)
    if n not in res.posteriors:
        return None
    post = res.posteriors[n]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    order = np.argsort(post.log_l)
    axes[0].plot(np.arange(order.size), post.log_l[order], color=style.SERIES[0], lw=1)
    axes[0].set_xlabel("stored samples, sorted by ln L")
    axes[0].set_ylabel("ln L")
    axes[1].plot(np.arange(order.size), np.cumsum(post.weights[order]), color=style.SERIES[1], lw=1.5)
    axes[1].set_xlabel("stored samples, sorted by ln L")
    axes[1].set_ylabel("cumulative posterior weight")
    tr = post.trace
    if "log_z" in tr and "log_x" in tr:
        ax3 = axes[0].twinx()
        ax3.grid(False)
        ax3.plot(
            np.linspace(0, order.size, tr["log_z"].size), tr["log_z"], color=style.INK_2, lw=0.8, ls="--"
        )
        ax3.set_ylabel("ln Z over iterations")
    fig.suptitle(
        f"{res.spectrum_id}: sampler diagnostics (N = {n}, ESS = {res.metrics.get('ess', np.nan):.0f})",
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    return _finish(fig, path)


def plot_all(res: FitResult, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _safe(res.spectrum_id)
    paths = []
    for name, fn in (
        ("spectrum", plot_spectrum),
        ("corner", plot_corner),
        ("evidence", plot_evidence),
        ("sampler", plot_sampler),
    ):
        p = out_dir / f"{base}_{name}.png"
        if fn(res, path=p) is not None:
            paths.append(p)
    return paths
