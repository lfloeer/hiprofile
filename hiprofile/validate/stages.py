"""Validation stages (Phase 4), reproducible via ``hiprofile validate``.

1. ``sbc``        simulation-based calibration: parameters from the prior, ranks of
                  the truth among posterior draws, coverage of 68 % / 95 % intervals
2. ``detection``  confusion matrix of N = 0, 1, 2, false-positive rate on pure noise,
                  reliability of P(detect), completeness vs S/N (optionally with and
                  without the RFI pre-stage)
3. ``misspec``    template profiles, sinusoidal baselines under a Chebyshev fit,
                  RFI without mask: bias, scatter, coverage and flag rates
4. ``injection``  model profiles injected into real signal-free spectra:
                  completeness and reliability vs flux and width

Every stage compares bias and scatter with the classical W50/W20 measurement
on the same spectra.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from hiprofile.config import Config
from hiprofile.inference.results import weighted_quantile
from hiprofile.io.catalog import collect_rows, rows_to_table
from hiprofile.io.results import read_result, result_path
from hiprofile.io.spectra import SpectrumSource, list_sources, write_spectrum
from hiprofile.runner import run_fits
from hiprofile.simulate import SimConfig, inject, simulate_many

STAGES = ("sbc", "detection", "misspec", "injection")

# Fixed priors for SBC (the simulation prior must equal the fit prior)
SBC_AMP = 4.0
SBC_BASE = 0.02
SBC_SIGMA = 0.01
SBC_KMAX = 3


@dataclass
class StageRun:
    stage: str
    out_dir: Path
    cfg: Config
    truth: list[dict[str, Any]]


def _write_sims(sims, out_dir: Path) -> tuple[list[SpectrumSource], list[dict[str, Any]]]:
    sdir = out_dir / "sims"
    sdir.mkdir(parents=True, exist_ok=True)
    sources, truth = [], []
    for spec, tr in sims:
        p = sdir / f"{spec.id}.h5"
        write_spectrum(spec, p)
        sources.append(SpectrumSource(str(p), spec.id))
        truth.append(tr.row())
    rows_to_table(truth).write(out_dir / "truth.csv", format="ascii.csv", overwrite=True)
    return sources, truth


def stage_configs(stage: str, base: Config) -> list[tuple[str, SimConfig, Config]]:
    """(label, simulation config, fit config) for each sub-run of a stage."""
    if stage == "sbc":
        priors = base.priors.model_copy(update={"amplitude_scale": SBC_AMP, "baseline_scale": SBC_BASE})
        sim = SimConfig(
            n_components=1,
            profiles="model",
            flux="prior",
            priors=priors,
            noise_sigma=SBC_SIGMA,
            baseline="chebyshev",
            baseline_order=SBC_KMAX,
            baseline_order_min=0,
            baseline_scale=SBC_BASE,
        )
        fit = base.updated(
            model={"n_max": 1, "k_min": 0, "k_max": SBC_KMAX, "n_rfi": 0},
            priors={"amplitude_scale": SBC_AMP, "baseline_scale": SBC_BASE},
            rfi={"prefit": False},
            noise={"sigma": "input"},
        )
        return [("sbc", sim, fit)]
    if stage == "detection":
        sim = SimConfig(
            n_components=[0.4, 0.4, 0.2],
            profiles="model",
            flux="snr",
            snr_range=(2.0, 40.0),
            baseline="chebyshev",
            baseline_order=1,
            baseline_scale=0.005,
            rfi="spikes",
            rfi_rate=0.005,
        )
        fit = base.updated(model={"n_max": 2})
        out = [("with_rfi_prefit", sim, fit)]
        out.append(("without_rfi_prefit", sim, fit.updated(rfi={"prefit": False})))
        return out
    if stage == "misspec":
        tpl = SimConfig(n_components=1, profiles="template", flux="snr", snr_range=(5.0, 80.0))
        sine = SimConfig(
            n_components=1,
            profiles="model",
            flux="snr",
            snr_range=(5.0, 80.0),
            baseline="sine",
            sine_amplitude=1.0,
        )
        rfi = SimConfig(
            n_components=1, profiles="model", flux="snr", snr_range=(5.0, 80.0), rfi="spikes", rfi_rate=0.01
        )
        clean = SimConfig(n_components=1, profiles="model", flux="snr", snr_range=(5.0, 80.0))
        return [
            ("clean", clean, base),
            ("template", tpl, base),
            ("sine_baseline", sine, base),
            ("rfi_unmasked", rfi, base.updated(rfi={"prefit": False})),
        ]
    if stage == "injection":
        sim = SimConfig(n_components=[0.5, 0.5], profiles="model", flux="snr", snr_range=(2.0, 40.0))
        return [("injection", sim, base)]
    raise ValueError(f"unknown stage {stage!r}; choose from {STAGES}")


def run_stage(
    stage: str,
    out_dir: str | Path,
    n: int,
    base: Config,
    jobs: int = 1,
    seed: int = 1,
    spectra: str | None = None,
    resume: bool = False,
    chunk: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Simulate, fit and analyse one stage.

    With `chunk = (i, n)` only the i-th of n slices of the spectra is fitted
    (SLURM arrays) and nothing is analysed; a final call with `resume=True`
    and no chunk collects the results.
    """
    out_dir = Path(out_dir) / f"validate_{stage}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"stage": stage, "n": n, "seed": seed, "runs": {}}
    for i, (label, sim, fit) in enumerate(stage_configs(stage, base)):
        run_dir = out_dir / label
        sim_seed = seed if stage == "detection" else seed + 1000 * i  # detection: same sims, two fits
        if stage == "injection":
            if spectra is None:
                raise ValueError("the injection stage needs --spectra with signal-free survey spectra")
            sims = []
            rng = np.random.default_rng(sim_seed)
            for j, src in enumerate(list_sources(spectra)[:n]):
                sims.append(inject(src.load(), sim, rng, f"inj{j:06d}"))
        else:
            sims = simulate_many(sim, n, sim_seed, prefix=f"{label[:3]}")
        sources, truth = _write_sims(sims, run_dir)
        run_fits(sources, fit, run_dir, jobs=jobs, chunk=chunk, resume=resume)
        (run_dir / "sim_config.yaml").write_text(json.dumps(sim.model_dump(mode="json"), indent=1))
        if chunk is None:
            summary["runs"][label] = analyze(stage, run_dir, truth, fit)
    if chunk is not None:
        summary["chunk"] = list(chunk)
        return summary
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (out_dir / "summary.md").write_text(to_markdown(summary))
    return summary


# ------------------------------------------------------------------------ analysis
def _catalog(run_dir: Path, truth: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    paths = [result_path(run_dir, t["id"]) for t in truth]
    rows = collect_rows([p for p in paths if p.exists()])
    return {r["id"]: r for r in rows}


def _stats(est: np.ndarray, true: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(est) & np.isfinite(true)
    if ok.sum() < 3:
        return {"n": int(ok.sum())}
    d = est[ok] - true[ok]
    rel = d / np.where(true[ok] != 0, np.abs(true[ok]), 1.0)
    return {
        "n": int(ok.sum()),
        "bias": float(np.median(d)),
        "scatter": float(1.4826 * np.median(np.abs(d - np.median(d)))),
        "rel_bias": float(np.median(rel)),
        "rel_scatter": float(1.4826 * np.median(np.abs(rel - np.median(rel)))),
    }


def _coverage(row_lo: np.ndarray, row_hi: np.ndarray, true: np.ndarray) -> float:
    ok = np.isfinite(row_lo) & np.isfinite(row_hi) & np.isfinite(true)
    if not ok.any():
        return float("nan")
    return float(np.mean((true[ok] >= row_lo[ok]) & (true[ok] <= row_hi[ok])))


def _arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(r.get(key, np.nan)) if r.get(key) is not None else np.nan for r in rows])


def compare_quantities(
    cat: dict[str, dict[str, Any]], truth: list[dict[str, Any]], comp_only: bool = True
) -> dict[str, Any]:
    rows = [cat.get(t["id"], {}) for t in truth]
    out: dict[str, Any] = {}
    for q in ("W50", "W20", "S", "v_sys"):
        tv = np.array([float(t.get(f"{q}_1", np.nan)) for t in truth])
        sel = np.array([t["n_true"] == 1 for t in truth]) if comp_only else np.ones(len(truth), bool)
        sel &= np.array([r.get("best_n", 0) >= 1 for r in rows])
        bay = _arr(rows, f"{q}_1")
        cls = _arr(rows, f"classic_{q}")
        lo, hi = _arr(rows, f"{q}_1_p16"), _arr(rows, f"{q}_1_p84")
        out[q] = {
            "bayes": _stats(bay[sel], tv[sel]),
            "classic": _stats(cls[sel], tv[sel]),
            "coverage68": _coverage(lo[sel], hi[sel], tv[sel]),
        }
    return out


def analyze(stage: str, run_dir: Path, truth: list[dict[str, Any]], cfg: Config) -> dict[str, Any]:
    cat = _catalog(run_dir, truth)
    rows = [cat.get(t["id"], {}) for t in truth]
    res: dict[str, Any] = {
        "n_fitted": sum(r.get("status") == "ok" for r in rows),
        "flag_rate": float(np.mean([bool(r.get("flagged", False)) for r in rows])) if rows else float("nan"),
    }
    flag_cols = sorted({k for r in rows for k in r if k.startswith("flag_")})
    res["flag_rates"] = {k[5:]: float(np.mean([bool(r.get(k, False)) for r in rows])) for k in flag_cols}
    if stage == "sbc":
        res.update(sbc_ranks(run_dir, truth))
        res.update(k_calibration(rows, truth))
    if stage in ("detection", "injection"):
        res.update(detection_metrics(rows, truth, cfg))
    if stage in ("misspec", "detection", "injection", "sbc"):
        res["quantities"] = compare_quantities(cat, truth)
    _plots(stage, run_dir, res)
    return res


def sbc_ranks(
    run_dir: Path, truth: list[dict[str, Any]], n_draw: int = 99, n_bins: int = 10
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    names = ["v_ctr_1", "delta_v_1", "delta_v_rand_1", "alpha_1", "S_1", "f_1", "W50_1", "W20_1"]
    ranks: dict[str, list[int]] = {k: [] for k in names}
    cover: dict[str, dict[str, list[bool]]] = {k: {"68": [], "95": []} for k in names}
    for t in truth:
        p = result_path(run_dir, t["id"])
        if not p.exists():
            continue
        r = read_result(p)
        if 1 not in r.posteriors:
            continue
        post = r.posteriors[1]
        cols = {**{nm: post.theta[:, i] for i, nm in enumerate(post.names)}, **post.derived}
        idx = rng.choice(post.weights.size, size=n_draw, p=post.weights)
        for k in names:
            tv = t.get(k)
            if tv is None or not np.isfinite(tv) or k not in cols:
                continue
            draws = cols[k][idx]
            if not np.all(np.isfinite(draws)):
                continue
            # randomised tie-breaking (W50/W20 of near-degenerate profiles can tie)
            less = np.sum(draws < tv)
            ties = np.sum(draws == tv)
            ranks[k].append(int(less + rng.integers(0, ties + 1)))
            # intervals from the full weighted posterior: the 2.5 / 97.5 % quantiles of
            # 99 draws sit inside the true ones and would under-report the coverage
            ok = np.isfinite(cols[k])
            q = weighted_quantile(cols[k][ok], post.weights[ok], [0.025, 0.16, 0.84, 0.975])
            cover[k]["68"].append(q[1] <= tv <= q[2])
            cover[k]["95"].append(q[0] <= tv <= q[3])
    out: dict[str, Any] = {"ranks": {}, "rank_pvalue": {}, "coverage68": {}, "coverage95": {}}
    for k in names:
        rk = np.array(ranks[k])
        if rk.size < 5:
            continue
        h, _ = np.histogram(rk, bins=n_bins, range=(0, n_draw + 1))
        out["ranks"][k] = h.tolist()
        out["rank_pvalue"][k] = float(stats.chisquare(h).pvalue)
        out["coverage68"][k] = float(np.mean(cover[k]["68"]))
        out["coverage95"][k] = float(np.mean(cover[k]["95"]))
    out["n_sbc"] = int(max((len(v) for v in ranks.values()), default=0))
    return out


def k_calibration(rows: list[dict[str, Any]], truth: list[dict[str, Any]]) -> dict[str, Any]:
    """Calibration of P(K | y) against the simulated baseline order.

    Reliability: pairs (P(K = k | y), 1[K_true = k]) over all k, binned in the
    predicted probability; confusion of the most probable K against the truth.
    """
    k_true = np.array([int(t.get("baseline_order", -1)) for t in truth])
    k_cols = sorted({k for r in rows for k in r if k.startswith("p_k")}, key=lambda s: int(s[3:]))
    if not k_cols or np.all(k_true < 0):
        return {}
    k_vals = [int(c[3:]) for c in k_cols]
    p = np.array(
        [[float(r.get(c, np.nan)) if r.get(c) is not None else np.nan for c in k_cols] for r in rows]
    )
    ok = np.all(np.isfinite(p), axis=1)
    p, kt = p[ok], k_true[ok]
    hit = kt[:, None] == np.asarray(k_vals)[None]
    bins = np.linspace(0, 1, 11)
    rel = []
    for lo, hi in itertools.pairwise(bins):
        sel = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        rel.append(
            [float(0.5 * (lo + hi)), int(sel.sum()), float(np.mean(hit[sel])) if sel.any() else float("nan")]
        )
    best = np.asarray(k_vals)[np.argmax(p, axis=1)]
    conf = np.zeros((len(k_vals), len(k_vals)), int)
    for a, b in zip(kt, best, strict=True):
        if a in k_vals:
            conf[k_vals.index(a), k_vals.index(b)] += 1
    expected = float(np.mean(p[hit]))  # mean P(K_true | y)
    return {
        "k_values": k_vals,
        "k_reliability": rel,
        "k_confusion": conf.tolist(),
        "k_mean_p_true": expected,
        "k_accuracy": float(np.mean(best == kt)),
    }


def detection_metrics(rows: list[dict[str, Any]], truth: list[dict[str, Any]], cfg: Config) -> dict[str, Any]:
    n_true = np.array([t["n_true"] for t in truth])
    n_fit = np.array([int(r.get("best_n", -1)) for r in rows])
    pdet = np.array([float(r.get("p_detect", np.nan)) for r in rows])
    n_max = max(int(n_true.max()), cfg.model.n_max)
    conf = np.zeros((n_max + 1, n_max + 1), int)
    for a, b in zip(n_true, n_fit, strict=True):
        if b >= 0:
            conf[a, b] += 1
    noise = n_true == 0
    out: dict[str, Any] = {
        "confusion": conf.tolist(),
        "false_positive_rate": {
            str(th): float(np.mean(pdet[noise] > th)) if noise.any() else float("nan")
            for th in (0.5, 0.9, 0.99)
        },
    }
    bins = np.linspace(0, 1, 11)
    rel = []
    for lo, hi in itertools.pairwise(bins):
        sel = (pdet >= lo) & (pdet < hi if hi < 1 else pdet <= hi)
        rel.append(
            [
                float(0.5 * (lo + hi)),
                int(sel.sum()),
                float(np.mean(n_true[sel] >= 1)) if sel.any() else float("nan"),
            ]
        )
    out["reliability"] = rel
    snr = np.array([float(t.get("snr_1", np.nan)) for t in truth])
    edges = np.array([0, 3, 5, 8, 12, 20, 40, np.inf])
    comp = []
    for lo, hi in itertools.pairwise(edges):
        sel = (n_true >= 1) & (snr >= lo) & (snr < hi)
        comp.append(
            [
                float(lo),
                float(hi),
                int(sel.sum()),
                float(np.mean(pdet[sel] > 0.5)) if sel.any() else float("nan"),
            ]
        )
    out["completeness_vs_snr"] = comp
    return out


def _plots(stage: str, run_dir: Path, res: dict[str, Any]) -> None:
    from hiprofile.plots import style

    style.apply()
    import matplotlib.pyplot as plt

    if stage == "sbc" and res.get("ranks"):
        names = list(res["ranks"])
        fig, axes = plt.subplots(2, (len(names) + 1) // 2, figsize=(2.6 * ((len(names) + 1) // 2), 4.6))
        for ax, k in zip(axes.flat, names, strict=False):
            h = np.array(res["ranks"][k])
            ax.bar(np.arange(h.size), h, color=style.SERIES[0], width=0.85)
            m = h.sum() / h.size
            band = stats.binom.interval(0.99, h.sum(), 1 / h.size)
            ax.axhspan(band[0], band[1], color=style.GRID, zorder=0)
            ax.axhline(m, color=style.INK_2, lw=0.8)
            ax.set_title(f"{k}  p={res['rank_pvalue'][k]:.2f}", loc="left", fontsize=8)
            ax.set_xticks([])
        fig.suptitle("SBC rank histograms (grey: 99 % band for uniform ranks)", x=0.02, ha="left")
        fig.tight_layout()
        fig.savefig(run_dir / "sbc_ranks.png", dpi=100)
        plt.close(fig)
    if "reliability" in res:
        rel = np.array(res["reliability"], float)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        ok = rel[:, 1] > 0
        axes[0].plot([0, 1], [0, 1], color=style.MUTED, lw=0.8)
        axes[0].plot(rel[ok, 0], rel[ok, 2], "o-", color=style.SERIES[0], ms=5)
        axes[0].set_xlabel("P(detect)")
        axes[0].set_ylabel("fraction with a true source")
        axes[0].set_title("reliability", loc="left")
        comp = np.array(res["completeness_vs_snr"], float)
        mids = np.where(np.isfinite(comp[:, 1]), 0.5 * (comp[:, 0] + comp[:, 1]), comp[:, 0] * 1.5)
        axes[1].plot(mids, comp[:, 3], "o-", color=style.SERIES[0], ms=5)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("integrated S/N (true)")
        axes[1].set_ylabel("completeness (P > 0.5)")
        axes[1].set_title("completeness", loc="left")
        fig.tight_layout()
        fig.savefig(run_dir / "detection.png", dpi=100)
        plt.close(fig)


def to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Validation: {summary['stage']}",
        "",
        f"n = {summary['n']} per run, seed = {summary['seed']}",
        "",
    ]
    for label, r in summary["runs"].items():
        lines += [
            f"## {label}",
            "",
            f"fitted: {r.get('n_fitted')}, flagged: {r.get('flag_rate', float('nan')):.1%}",
            "",
        ]
        if r.get("flag_rates"):
            lines.append("flags: " + ", ".join(f"{k} {v:.1%}" for k, v in r["flag_rates"].items() if v > 0))
            lines.append("")
        if "rank_pvalue" in r:
            lines += ["| parameter | rank p-value | 68 % coverage | 95 % coverage |", "|---|---|---|---|"]
            for k, p in r["rank_pvalue"].items():
                lines.append(f"| {k} | {p:.3f} | {r['coverage68'][k]:.3f} | {r['coverage95'][k]:.3f} |")
            lines.append("")
        if "k_confusion" in r:
            lines += [
                f"baseline order: P(K_true | y) averages {r['k_mean_p_true']:.2f}, most probable K correct "
                f"in {r['k_accuracy']:.0%} (rows: true K, columns: most probable K, K = {r['k_values']}):",
                "",
                "```",
            ]
            lines += ["  ".join(f"{v:5d}" for v in row) for row in r["k_confusion"]]
            lines += ["```", ""]
        if "confusion" in r:
            lines += ["confusion matrix (rows: true N, columns: fitted N):", "", "```"]
            lines += ["  ".join(f"{v:5d}" for v in row) for row in r["confusion"]]
            lines += ["```", "", f"false-positive rate: {r['false_positive_rate']}", ""]
            lines += ["| S/N | n | completeness |", "|---|---|---|"]
            for lo, hi, n, c in r["completeness_vs_snr"]:
                lines.append(f"| {lo:g}-{hi:g} | {n} | {c:.2f} |")
            lines.append("")
        if "quantities" in r:
            lines += [
                "| quantity | bayes bias | bayes scatter | classic bias | classic scatter | 68 % coverage |",
                "|---|---|---|---|---|---|",
            ]
            for q, d in r["quantities"].items():
                b, c = d["bayes"], d["classic"]
                nan = float("nan")
                lines.append(
                    f"| {q} | {b.get('bias', nan):.3g} | {b.get('scatter', nan):.3g} | "
                    f"{c.get('bias', nan):.3g} | {c.get('scatter', nan):.3g} | {d['coverage68']:.2f} |"
                )
            lines.append("")
    return "\n".join(lines)
