"""Figures for the README and the documentation.

    python docs/make_figures.py priors            # prior densities vs. THINGS fits
    python docs/make_figures.py gallery           # fits of simulated spectra
    python docs/make_figures.py validation DIR    # SBC / detection / classical comparison from a
                                                  # `hiprofile validate --out DIR` run
    python docs/make_figures.py bias DIR          # conditional bias from benchmarks/bias_study.py

All figures go to docs/img/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from hiprofile.plots import style

style.apply()
import matplotlib.pyplot as plt  # noqa: E402

IMG = Path(__file__).resolve().parent / "img"

# Table 1 of Stewart, Blyth & de Blok (2014): 34 THINGS galaxies
THINGS = {
    "delta_v": [81.24, 0.08, 48.97, 25.70, 97.84, 0.08, 42.27, 45.47, 191.24, 73.00, 94.84, 228.75,
                566.35, 353.80, 133.88, 366.28, 117.12, 117.17, 283.65, 255.37, 415.67, 257.31, 327.58,
                56.34, 133.64, 195.56, 288.68, 357.96, 148.64, 172.90, 138.68, 210.84, 479.34, 157.62],
    "delta_v_rand": [9.04, 12.37, 9.19, 8.83, 13.08, 8.83, 8.33, 11.88, 12.06, 27.65, 9.43, 8.91, 14.62,
                     12.71, 10.80, 23.62, 17.13, 9.60, 10.58, 8.05, 19.17, 11.34, 22.07, 12.40, 20.29,
                     15.47, 11.79, 17.09, 17.36, 31.73, 23.15, 11.09, 15.07, 12.43],
    "f": [0.2600, 0.5029, 0.2449, 0.5369, 0.3410, 0.5001, 0.3803, 0.0080, 0.2731, 0.1929, 0.5250, 0.3580,
          0.1228, 0.1224, 0.5288, 0.0702, 0.0009, 0.1042, 0.1201, 0.0051, 0.0975, 0.2857, 0.1764, 0.0043,
          0.9261, 0.0016, 0.0110, 0.4021, 0.1259, 0.0030, 0.0134, 0.3078, 0.0414, 0.1312],
    "alpha": [-0.0901, -0.0005, -0.0663, -0.6686, -0.0788, 0.0124, 0.1226, -0.1235, 0.0603, -0.8676, 0.0767,
              0.1191, 0.0065, -0.1072, -0.2219, 0.2966, 0.9943, -0.0798, 0.0353, -0.0001, -0.0461, -0.0272,
              0.1190, -0.0458, -0.0222, -0.0500, 0.3199, 0.1099, 0.3317, 0.1650, 0.2260, 0.1004, 0.1150,
              -0.0601],
}  # fmt: skip


def _save(fig, name: str) -> Path:
    IMG.mkdir(parents=True, exist_ok=True)
    path = IMG / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


def priors() -> Path:
    """New default priors (lines) against the 34 THINGS fits (histograms)."""
    from hiprofile.config import Priors
    from hiprofile.inference.priors import build_prior

    pr = build_prior(("profile",), Priors(), np.arange(1000) * 5.0)
    rng = np.random.default_rng(0)
    u = rng.uniform(size=(400000, 5))
    th = pr(u)
    draws = {"delta_v": th[:, 1], "delta_v_rand": th[:, 2], "alpha": th[:, 3], "f": th[:, 4]}
    panels = [
        (
            "delta_v",
            "Δv [km/s]",
            np.linspace(0, 800, 33),
            "log-normal 160 km/s, 0.4 dex",
        ),
        ("delta_v_rand", "Δv_rand [km/s]", np.linspace(4, 40, 37), "log-normal 12.6 km/s, 0.2 dex"),
        ("alpha", "α", np.linspace(-1, 1, 41), "normal(0, 0.25) on [−1, 1]"),
        ("f", "f (solid-body fraction)", np.linspace(0, 1, 21), "uniform on [0, 1]"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
    for ax, (key, label, bins, desc) in zip(axes, panels, strict=True):
        ax.hist(
            THINGS[key], bins=bins, density=True, color=style.SERIES[0], alpha=0.55, label="THINGS fits (34)"
        )
        h, e = np.histogram(draws[key], bins=np.linspace(bins[0], bins[-1], 200), density=True)
        ax.plot(0.5 * (e[1:] + e[:-1]), h, color=style.SERIES[1], lw=2, label="new default prior")
        ax.set_xlabel(label)
        ax.set_yticks([])
        ax.set_title(desc, loc="left", fontsize=8, color=style.INK_2)
    axes[0].legend(loc="upper right")
    fig.suptitle(
        "Default priors vs. the THINGS fits of Stewart et al. (2014)", x=0.01, ha="left", fontsize=11, y=1.06
    )
    return _save(fig, "priors_things.png")


GALLERY = [
    # (title, components [(v_ctr, dv, dv_rand, alpha, S, f)], noise, rfi channels, baseline coefficients)
    ("Spiral galaxy, S/N ≈ 60", [(2400.0, 320.0, 10.0, 0.12, 30.0, 0.1)], 0.01, [], [0.0, 0.0]),
    ("Dwarf galaxy, S/N ≈ 12", [(1800.0, 30.0, 9.0, 0.0, 2.2, 0.5)], 0.01, [], [0.0, 0.0]),
    ("Interacting pair, N = 2", [(2250.0, 180.0, 12.0, -0.2, 8.0, 0.2), (2480.0, 120.0, 10.0, 0.3, 5.0, 0.3)],
     0.01, [], [0.0, 0.0]),
    ("Faint line, RFI and baseline, S/N ≈ 8", [(2000.0, 220.0, 14.0, 0.0, 4.0, 0.2)], 0.01,
     [120, 121, 300, 301, 302, 420], [0.004, 0.006]),
]  # fmt: skip


def _simulate_gallery(rng):
    from hiprofile import Spectrum
    from hiprofile.model import SpectralGrid, chebyshev_basis, profile

    v = 1000.0 + 5.0 * np.arange(512)
    grid = SpectralGrid.from_velocities(v, supersample=8)
    out = []
    for title, comps, sigma, rfi, base in GALLERY:
        y = sum(profile(grid, c) for c in comps) + sigma * rng.standard_normal(v.size)
        y = y + np.asarray(base) @ chebyshev_basis(v.size, len(base) - 1)
        y[rfi] += sigma * rng.uniform(8, 30, len(rfi))
        out.append((title, Spectrum(v, y, sigma=np.full(v.size, sigma), id=title), comps))
    return out


def _panel(ax, axr, res, title, comps):
    v, y = res.velocity, res.flux
    good = ~res.mask
    b = res.model_band
    ax.step(v, np.where(good, y, np.nan), where="mid", color=style.INK, lw=0.6)
    ylo = float(np.min(np.where(good, y, np.inf)))
    yhi = float(max(np.max(np.where(good, y, -np.inf)), np.max(res.model_band[4])))
    pad = 0.08 * (yhi - ylo)
    ax.set_ylim(ylo - pad, yhi + 2.5 * pad)
    if res.mask.any():
        # masked channels as markers at the top edge (their values are off scale)
        ax.plot(
            v[res.mask],
            np.full(res.mask.sum(), yhi + 1.6 * pad),
            "v",
            color=style.SERIES[7],
            ms=5,
            label="masked (RFI pre-stage)",
        )
    ax.fill_between(v, b[0], b[4], color=style.SERIES[0], alpha=0.18, lw=0, label="95 % band")
    ax.fill_between(v, b[1], b[3], color=style.SERIES[0], alpha=0.35, lw=0, label="68 % band")
    ax.plot(v, b[2], color=style.SERIES[0], lw=1.6, label="posterior median")
    if len(res.components_median) > 1:
        for i, comp in enumerate(res.components_median):
            ax.plot(
                v,
                comp + res.baseline_median,
                color=style.SERIES[1 + i],
                lw=1.1,
                ls="--",
                label=f"component {i + 1}",
            )
    if np.any(np.abs(res.baseline_median) > 0):
        ax.plot(v, res.baseline_median, color=style.INK_2, lw=0.9, ls=":", label="baseline")
    post = res.posteriors[max(res.best_n, 1)].summary()
    parts = []
    for i in range(1, max(res.best_n, 1) + 1):
        err = 0.5 * (post[f"W50_{i}_p84"] - post[f"W50_{i}_p16"])
        parts.append(f"{post[f'W50_{i}']:.0f} ± {err:.0f}")
    w50 = "W50 = " + " / ".join(parts) + " km/s"
    ax.set_title(f"{title}\nP(detect) = {res.p_detect:.3f}, N = {res.best_n}, {w50}", loc="left", fontsize=9)
    lo = min(c[0] - c[1] for c in comps) - 250
    hi = max(c[0] + c[1] for c in comps) + 250
    if res.mask.any():  # show the whole band to include the masked channels
        lo, hi = v[0], v[-1]
    for a in (ax, axr):
        a.set_xlim(lo, hi)
    r = (y - b[2]) / res.sigma
    axr.step(v, np.where(good, r, np.nan), where="mid", color=style.INK, lw=0.6)
    axr.axhline(0, color=style.MUTED, lw=0.8)
    axr.set_ylim(-4.5, 4.5)
    axr.set_ylabel("resid. / σ", fontsize=8)
    ax.set_ylabel("flux density [Jy]", fontsize=8)
    ax.tick_params(labelbottom=False)


def gallery() -> Path:
    from hiprofile.config import Config
    from hiprofile.inference.fit import fit_spectrum

    rng = np.random.default_rng(7)
    cases = _simulate_gallery(rng)
    fig = plt.figure(figsize=(12, 7.2))
    outer = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.18)
    handles: dict[str, object] = {}
    for k, (title, spec, comps) in enumerate(cases):
        res = fit_spectrum(spec, Config())
        assert res.status == "ok", res.error
        inner = outer[k // 2, k % 2].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax, axr = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        _panel(ax, axr, res, title, comps)
        if k >= 2:
            axr.set_xlabel("velocity [km/s]")
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(lab, h)
        print(title, "->", res.best_n, f"{res.runtime:.1f}s")
    fig.legend(
        list(handles.values()),
        list(handles.keys()),
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.suptitle(
        "hiprofile fits of simulated spectra (full mode, default priors)", x=0.01, ha="left", fontsize=12
    )
    return _save(fig, "gallery.png")


def _load(run_dir: Path):
    from astropy.table import Table

    from hiprofile.io.catalog import collect_rows

    truth = Table.read(run_dir / "truth.csv")
    rows = {r["id"]: r for r in collect_rows(sorted((run_dir / "spectra").glob("*.h5")))}
    return truth, rows


def _col(rows, ids, key):
    return np.array([float(rows.get(i, {}).get(key, np.nan) or np.nan) for i in ids])


def sbc_figure(stage_dir: Path) -> Path:
    import json

    from scipy import stats

    summ = json.loads((stage_dir / "summary.json").read_text())["runs"]["sbc"]
    names = list(summ["ranks"])
    labels = {
        "v_ctr_1": "v_ctr",
        "delta_v_1": "Δv",
        "delta_v_rand_1": "Δv_rand",
        "alpha_1": "α",
        "S_1": "S",
        "f_1": "f",
        "W50_1": "W50",
        "W20_1": "W20",
    }
    has_k = "k_reliability" in summ
    fig = plt.figure(figsize=(15, 5.2) if has_k else (13, 5.2))
    widths = [1, 1, 1, 1, 0.15, 1.6, 0.15, 1.2] if has_k else [1, 1, 1, 1, 0.15, 1.6]
    gs = fig.add_gridspec(2, len(widths), width_ratios=widths, hspace=0.55, wspace=0.35)
    for k, name in enumerate(names):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        h = np.array(summ["ranks"][name])
        lo, hi = stats.binom.interval(0.99, h.sum(), 1 / h.size)
        ax.axhspan(lo, hi, color=style.GRID, zorder=0)
        ax.bar(np.arange(h.size), h, color=style.SERIES[0], width=0.85)
        ax.axhline(h.sum() / h.size, color=style.INK_2, lw=0.8)
        ax.set_title(
            f"{labels.get(name, name)}   p = {summ['rank_pvalue'][name]:.2f}", loc="left", fontsize=9
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    ax = fig.add_subplot(gs[:, 5])
    y = np.arange(len(names))[::-1]
    c68 = [summ["coverage68"][n] for n in names]
    c95 = [summ["coverage95"][n] for n in names]
    n_sim = summ["n_sbc"]
    for target, vals, color, lab in (
        (0.68, c68, style.SERIES[0], "68 % interval"),
        (0.95, c95, style.SERIES[1], "95 % interval"),
    ):
        err = 2 * np.sqrt(target * (1 - target) / n_sim)
        ax.axvspan(target - err, target + err, color=color, alpha=0.12, lw=0)
        ax.axvline(target, color=color, lw=0.8)
        ax.plot(vals, y, "o", color=color, ms=6, label=lab)
    ax.set_yticks(y, [labels.get(n, n) for n in names])
    ax.set_xlim(0.55, 1.0)
    ax.set_xlabel("coverage")
    ax.legend(loc="lower right")
    ax.set_title("interval coverage (band: ±2σ for N sims)", loc="left", fontsize=9)
    if has_k:
        axk = fig.add_subplot(gs[:, 7])
        rel = np.array(summ["k_reliability"], float)
        ok = rel[:, 1] > 0
        axk.plot([0, 1], [0, 1], color=style.MUTED, lw=0.8)
        axk.plot(rel[ok, 0], rel[ok, 2], "o-", color=style.SERIES[1], ms=5)
        axk.set_xlabel("P(K = k | y)")
        axk.set_ylabel("fraction with true order k")
        axk.set_title(
            f"baseline order: accuracy {summ['k_accuracy']:.0%}, mean P(K_true) {summ['k_mean_p_true']:.2f}",
            loc="left",
            fontsize=9,
        )
    fig.suptitle(
        f"Simulation-based calibration, {n_sim} simulations from the default priors "
        "(rank histograms, grey: 99 % band of uniform ranks)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    return _save(fig, "sbc.png")


def _log_snr_axis(ax, ticks=(2, 3, 5, 10, 20, 50, 100)) -> None:
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())


def classic_figure(run_dir: Path, name: str = "bayes_vs_classic.png") -> Path:
    """Error of posterior median vs. classical W50/W20 against the true S/N."""
    truth, rows = _load(run_dir)
    sel = np.asarray(truth["n_true"]) == 1
    ids = np.asarray(truth["id"])[sel]
    snr = np.asarray(truth["snr_1"])[sel]
    detected = _col(rows, ids, "p_detect") > 0.5
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    edges = np.geomspace(max(snr[detected].min(), 2), snr.max() * 1.01, 9)
    for ax, q in zip(axes, ("W50", "W20"), strict=True):
        t = np.asarray(truth[f"{q}_1"])[sel]
        for key, color, lab in (
            (f"{q}_1", style.SERIES[0], "hiprofile posterior median"),
            (f"classic_{q}", style.SERIES[1], f"classical {q} on the data"),
        ):
            err = _col(rows, ids, key) - t
            ok = detected & np.isfinite(err)
            ax.scatter(snr[ok], err[ok], s=7, color=color, alpha=0.35, edgecolor="none")
            mid, med, q16, q84 = [], [], [], []
            for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                b = ok & (snr >= lo) & (snr < hi)
                if b.sum() >= 5:
                    mid.append(np.sqrt(lo * hi))
                    p = np.percentile(err[b], [16, 50, 84])
                    q16.append(p[0])
                    med.append(p[1])
                    q84.append(p[2])
            ax.plot(mid, med, "-o", color=color, lw=2, ms=4, label=lab)
            ax.fill_between(mid, q16, q84, color=color, alpha=0.15, lw=0)
        ax.axhline(0, color=style.INK_2, lw=0.8)
        _log_snr_axis(ax)
        ax.set_xlabel("true integrated S/N")
        ax.set_ylabel(f"{q} estimate − truth [km/s]")
        lim = np.nanpercentile(np.abs(_col(rows, ids, f"classic_{q}") - t), 95)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"{q}: median and 16–84 % range per S/N bin", loc="left", fontsize=9)
    axes[0].legend(loc="lower right")
    fig.suptitle(
        "Line widths: Bayesian fit vs. classical measurement (detected simulations with N = 1)",
        x=0.01,
        ha="left",
        fontsize=11,
        y=1.03,
    )
    return _save(fig, name)


def detection_figure(stage_dir: Path) -> Path:
    import json

    summ = json.loads((stage_dir / "summary.json").read_text())["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.2, 1, 0.8]})
    for (label, run), color in zip(summ.items(), style.SERIES, strict=False):
        comp = np.array(run["completeness_vs_snr"], float)
        mids = np.where(
            np.isfinite(comp[:, 1]), np.sqrt(np.maximum(comp[:, 0], 1) * comp[:, 1]), comp[:, 0] * 1.5
        )
        nice = label.replace("_", " ")
        axes[0].plot(mids, comp[:, 3], "-o", color=color, lw=2, ms=5, label=nice)
        rel = np.array(run["reliability"], float)
        ok = rel[:, 1] > 0
        axes[1].plot(rel[ok, 0], rel[ok, 2], "-o", color=color, lw=2, ms=5, label=nice)
    _log_snr_axis(axes[0])
    axes[0].set_xlabel("true integrated S/N")
    axes[0].set_ylabel("completeness (P(detect) > 0.5)")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(loc="lower right")
    axes[0].set_title("completeness", loc="left", fontsize=10)
    axes[1].plot([0, 1], [0, 1], color=style.MUTED, lw=0.8)
    axes[1].set_xlabel("P(detect)")
    axes[1].set_ylabel("fraction with a true source")
    axes[1].set_title("reliability of P(detect)", loc="left", fontsize=10)
    run = summ[next(iter(summ))]
    conf = np.array(run["confusion"], float)
    frac = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    axes[2].imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            axes[2].text(
                j,
                i,
                f"{int(conf[i, j])}",
                ha="center",
                va="center",
                color=style.SURFACE if frac[i, j] > 0.6 else style.INK,
                fontsize=10,
            )
    axes[2].set_xticks(range(conf.shape[1]))
    axes[2].set_yticks(range(conf.shape[0]))
    axes[2].set_xlabel("fitted N")
    axes[2].set_ylabel("true N")
    axes[2].grid(False)
    fpr = run["false_positive_rate"]["0.5"]
    axes[2].set_title(f"confusion (with RFI pre-stage)\nfalse positives: {fpr:.1%}", loc="left", fontsize=10)
    fig.suptitle(
        "Detection: simulations with N = 0, 1, 2, Chebyshev baselines and RFI spikes",
        x=0.01,
        ha="left",
        fontsize=11,
        y=1.04,
    )
    return _save(fig, "detection.png")


def misspec_figure(stage_dir: Path) -> Path:
    import json

    summ = json.loads((stage_dir / "summary.json").read_text())["runs"]
    labels = list(summ)
    flags = sorted({f for r in summ.values() for f, v in r["flag_rates"].items() if v > 0.02})
    fig, ax = plt.subplots(figsize=(11, 3.6))
    width = 0.8 / len(labels)
    x = np.arange(len(flags))
    for i, (lab, color) in enumerate(zip(labels, style.SERIES, strict=False)):
        vals = [summ[lab]["flag_rates"].get(f, 0.0) for f in flags]
        ax.bar(
            x + (i - (len(labels) - 1) / 2) * width,
            vals,
            width=width * 0.9,
            color=color,
            label=f"{lab.replace('_', ' ')} (any flag: {summ[lab]['flag_rate']:.0%})",
        )
    ax.set_xticks(x, [f.replace("_", " ") for f in flags])
    ax.set_ylabel("fraction of spectra flagged")
    ax.legend(ncol=2, loc="upper right")
    ax.set_title(
        "Flags on deliberately misspecified spectra vs. clean model spectra", loc="left", fontsize=11
    )
    return _save(fig, "misspec_flags.png")


def bias_figure(root: Path, name: str = "bias_study.png") -> Path:
    """Conditional bias of W50 against S/N for fixed true widths (benchmarks/bias_study.py)."""
    import json

    sets = [
        (p, json.loads((root / f"bias_{p}" / "bias_table.json").read_text())) for p in ("default", "flat")
    ]
    widths = sorted({c["delta_v"] for _, t in sets[0:1] for c in t})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (_, table), title in zip(
        axes[:2], sets, ("default priors (THINGS)", "flat priors (log-uniform widths)"), strict=True
    ):
        for i, dv in enumerate(widths):
            cells = sorted((c for c in table if c["delta_v"] == dv), key=lambda c: c["snr"])
            snr = [c["snr"] for c in cells]
            w50 = cells[0]["delta_v"] + 2.35 * 12.0  # approximate true W50 for the legend
            ax.plot(
                snr,
                [c["bayes_W50_bias"] for c in cells],
                "-o",
                color=style.SERIES[i],
                lw=2,
                ms=4,
                label=f"Δv = {dv:.0f} km/s (W50 ≈ {w50:.0f})",
            )
            ax.fill_between(
                snr,
                [c["bayes_W50_bias"] - c["bayes_W50_scatter"] for c in cells],
                [c["bayes_W50_bias"] + c["bayes_W50_scatter"] for c in cells],
                color=style.SERIES[i],
                alpha=0.08,
                lw=0,
            )
        ax.axhline(0, color=style.INK_2, lw=0.8)
        _log_snr_axis(ax, (3, 5, 8, 15, 40))
        ax.set_xlabel("true integrated S/N")
        ax.set_ylabel("median W50 − truth [km/s]")
        ax.set_title(f"posterior median, {title}", loc="left", fontsize=9)
    ax = axes[2]
    for i, dv in enumerate(widths):
        cells = sorted((c for c in sets[0][1] if c["delta_v"] == dv), key=lambda c: c["snr"])
        ax.plot(
            [c["snr"] for c in cells],
            [c["classic_W50_bias"] for c in cells],
            "-o",
            color=style.SERIES[i],
            lw=2,
            ms=4,
        )
    ax.axhline(0, color=style.INK_2, lw=0.8)
    _log_snr_axis(ax, (3, 5, 8, 15, 40))
    ax.set_xlabel("true integrated S/N")
    ax.set_ylabel("classical W50 − truth [km/s]")
    ax.set_title("classical W50 on the same spectra", loc="left", fontsize=9)
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Conditional bias of W50 for fixed true profiles (median ± robust scatter per cell)",
        x=0.01,
        ha="left",
        fontsize=11,
        y=1.03,
    )
    return _save(fig, name)


def validation(root: Path) -> None:
    root = Path(root)
    if (root / "validate_sbc" / "summary.json").exists():
        sbc_figure(root / "validate_sbc")
        classic_figure(root / "validate_sbc" / "sbc")
    if (root / "validate_detection" / "summary.json").exists():
        detection_figure(root / "validate_detection")
    if (root / "validate_misspec" / "summary.json").exists():
        misspec_figure(root / "validate_misspec")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "priors"
    if what == "priors":
        priors()
    elif what == "gallery":
        gallery()
    elif what == "validation" and len(sys.argv) > 2:
        validation(Path(sys.argv[2]))
    elif what == "bias" and len(sys.argv) > 2:
        bias_figure(Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
