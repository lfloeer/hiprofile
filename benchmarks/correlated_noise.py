"""Validation of the noise-correlation models on noise smoothed by a kernel the fit does not know.

    python benchmarks/correlated_noise.py --out DIR [--n 300] [--n-noise 150] [--jobs 4]
                                          [--figure docs/img/correlated_noise.png]

The simulations are those of the SBC stage (all parameters from the priors,
Chebyshev baseline of order 0-3), but the noise is smoothed by a Gaussian
kernel (sigma = 0.8 channels, 7 taps; rho_1 = 0.67, rho_2 = 0.21, rho_3 = 0.03)
as after regridding; the line itself is not smoothed. Plus line-free
spectra for the false-positive rate. Every spectrum is fitted four times:

* ``white``: correlation ignored (the per-channel sigma is right);
* ``hanning``: a plausible but wrong guess (hanning+white);
* ``estimate``: correlation measured on the pre-fit residuals;
* ``oracle``: the true correlation of the kernel.

Reported: SBC rank p-values and 68/95 % coverages, false-positive rates on
the line-free spectra and the estimated rho_1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hiprofile.config import Config
from hiprofile.io.catalog import collect_rows
from hiprofile.io.results import result_path
from hiprofile.likelihood.noise import kernel_correlation
from hiprofile.runner import run_fits
from hiprofile.simulate import simulate_many
from hiprofile.validate.stages import _write_sims, sbc_ranks, stage_configs

KERNEL = np.exp(-0.5 * (np.arange(-3, 4) / 0.8) ** 2)
KERNEL /= KERNEL.sum()
NAMES = ["v_ctr_1", "delta_v_1", "delta_v_rand_1", "alpha_1", "S_1", "f_1", "W50_1", "W20_1"]


def fit_configs(base: Config) -> dict[str, Config]:
    rho = kernel_correlation(KERNEL)
    return {
        "white": base.updated(noise={"correlation": "white"}),
        "hanning": base.updated(noise={"correlation": "hanning+white"}),
        "estimate": base.updated(noise={"correlation": "estimate"}),
        "oracle": base.updated(noise={"correlation": [float(x) for x in rho[1:]]}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-noise", type=int, default=150)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--figure", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    _, sim, fit = stage_configs("sbc", Config())[0]
    sim = sim.model_copy(update={"noise_kernel": KERNEL.tolist()})
    fit = fit.updated(output={"plots": "none"})
    line_src, line_truth = _write_sims(simulate_many(sim, args.n, args.seed, prefix="lin"), out / "lines")
    noise_sim = sim.model_copy(update={"n_components": 0})
    noise_src, _ = _write_sims(
        simulate_many(noise_sim, args.n_noise, args.seed + 1, prefix="nos"), out / "noise"
    )
    summary: dict[str, dict] = {}
    for label, cfg in fit_configs(fit).items():
        run = out / label
        run_fits([*line_src, *noise_src], cfg, run, jobs=args.jobs, resume=True)
        ranks = sbc_ranks(run, line_truth)
        rows = {r["id"]: r for r in collect_rows([result_path(run, s.id) for s in [*line_src, *noise_src]])}
        p_noise = np.array([float(rows[s.id]["p_detect"]) for s in noise_src])
        rho1 = np.array([float(rows[s.id].get("m_noise_rho1", np.nan)) for s in line_src])
        summary[label] = {
            "rank_pvalue": ranks["rank_pvalue"],
            "coverage68": ranks["coverage68"],
            "coverage95": ranks["coverage95"],
            "fpr_05": float(np.mean(p_noise > 0.5)),
            "fpr_09": float(np.mean(p_noise > 0.9)),
            "rho1_median": float(np.nanmedian(rho1)),
            "rho1_p16_p84": [float(x) for x in np.nanpercentile(rho1, [16, 84])],
            "n_sbc": ranks["n_sbc"],
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    rho = kernel_correlation(KERNEL)
    print(f"true rho_1..3 = {np.round(rho[1:4], 3)}")
    print(
        f"{'config':9s} {'rho_1':>14s} {'FPR>0.5':>8s} {'FPR>0.9':>8s} | "
        "coverage 68 / 95 % (min-max) | worst rank p"
    )
    for label, sm in summary.items():
        c68, c95 = list(sm["coverage68"].values()), list(sm["coverage95"].values())
        pv = sm["rank_pvalue"]
        worst = min(pv, key=pv.get)
        lo, hi = sm["rho1_p16_p84"]
        print(
            f"{label:9s} {sm['rho1_median']:5.2f} [{lo:4.2f},{hi:4.2f}] "
            f"{sm['fpr_05']:8.3f} {sm['fpr_09']:8.3f} | "
            f"{min(c68):.2f}-{max(c68):.2f} / {min(c95):.2f}-{max(c95):.2f} | {worst} {pv[worst]:.3g}"
        )
    if args.figure:
        figure(summary, Path(args.figure), args.n)


def figure(summary: dict[str, dict], path: Path, n: int) -> None:
    from hiprofile.plots import style

    style.apply()
    import matplotlib.pyplot as plt

    labels = {"white": "white (ignored)", "hanning": "Hanning (wrong guess)", "estimate": "estimated",
              "oracle": "true correlation"}  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1.2, 1.2, 0.8]})
    for ax, key, nominal in ((axes[0], "coverage68", 0.68), (axes[1], "coverage95", 0.95)):
        band = 2 * np.sqrt(nominal * (1 - nominal) / n)
        ax.axvspan(nominal - band, nominal + band, color=style.INK_2, alpha=0.12, lw=0)
        ax.axvline(nominal, color=style.INK_2, lw=0.8)
        for j, (label, sm) in enumerate(summary.items()):
            vals = [sm[key].get(p, np.nan) for p in NAMES]
            ax.plot(vals, np.arange(len(NAMES)) + 0.15 * (j - 1.5), "o", ms=5, color=style.SERIES[j],
                    label=labels[label])  # fmt: skip
        ax.set_yticks(np.arange(len(NAMES)), [p.removesuffix("_1") for p in NAMES])
        ax.invert_yaxis()
        ax.set_xlabel(f"{int(nominal * 100)} % interval coverage")
    axes[0].legend(fontsize=8, loc="lower left")
    ax = axes[2]
    x = np.arange(len(summary))
    ax.bar(
        x - 0.2,
        [sm["fpr_05"] for sm in summary.values()],
        0.4,
        color=style.SERIES[1],
        label="P(detect) > 0.5",
    )
    ax.bar(
        x + 0.2,
        [sm["fpr_09"] for sm in summary.values()],
        0.4,
        color=style.SERIES[0],
        label="P(detect) > 0.9",
    )
    ax.set_xticks(x, [labels[k].split(" (")[0] for k in summary], fontsize=8)
    ax.set_ylabel("false-positive rate (line-free spectra)")
    ax.legend(fontsize=8)
    fig.suptitle(
        "Noise smoothed by an unknown kernel (Gaussian, 0.8 channels): which noise model keeps the fit "
        "calibrated?",
        x=0.01,
        ha="left",
        fontsize=11,
        y=1.03,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
