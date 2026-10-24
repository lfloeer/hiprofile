"""Where does the low-S/N width bias come from?

    python benchmarks/w20_bias.py --out DIR [--n 100] [--jobs 4] [--variants default,k0,linear]
                                  [--response boxcar+hanning]

The bias study (bias_study.py) shows median W20 errors at S/N <= 5 from 20
spectra per cell, of which only the detected ones count. This study
separates the candidate causes on the same simulated spectra (Delta v =
20-400 km/s, S/N 3, 5, 8, boxcar channels unless ``--response`` says
otherwise, no baseline in the data):

* small numbers: n spectra per cell (default 100) instead of 20, with a
  bootstrap error of each median;
* selection: medians over all spectra, over the detected (P(detect) > 0.5)
  and over the undetected ones, all from the N = 1 posterior;
* prior: variant ``linear`` replaces the Gamma(2.3, 83 km/s) Delta v prior by
  one uniform in Delta v on [0, 800] km/s, which favours wide lines;
* baseline: variant ``k0`` allows only a constant baseline (K = 0) instead of
  marginalising K = 0-3, so a polynomial cannot take up the faint wings.

The two classical measurements are evaluated on the same spectra, once in
the window the pipeline uses (around the posterior v_sys and W20, so a
prior-pulled posterior narrows it) and once in the same-sized window around
the true v_sys and W20, with a constant baseline from the median outside it
(the best case for an observer who knows where the line is).

Writes per-spectrum results (``spectra.json``) and the cell table
(``cells.json``) per variant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from bias_study import NOISE, simulate

from hiprofile.config import Config
from hiprofile.inference.derived import classic_measurements
from hiprofile.io.catalog import collect_rows
from hiprofile.io.results import result_path
from hiprofile.runner import run_fits

DELTA_V = (20.0, 50.0, 100.0, 200.0, 400.0)
SNR = (3.0, 5.0, 8.0)
VARIANTS = {
    "default": {},
    "k0": {"model": {"k_min": 0, "k_max": 0}},
    "linear": {"priors": {"delta_v": {"kind": "uniform", "low": 0.0, "high": 800.0}}},
}
ESTIMATORS = {  # name: (catalogue key or None for the true window, classical key)
    "bayes": ("{q}_1", None),
    "naive_fitwin": ("classic_{q}", None),
    "springob_fitwin": ("classic_{q}_springob", None),
    "naive_truewin": (None, "{q}"),
    "springob_truewin": (None, "{q}_springob"),
}
KEYS = ("classic_W20", "classic_W50", "classic_W20_springob", "classic_W50_springob",
        "p_detect", "best_n", "best_k", "W20_1", "W20_1_p16", "W20_1_p84", "W50_1", "delta_v_1",
        "delta_v_rand_1", "S_1", "m_constraint_delta_v", "flag_prior_dominated")  # fmt: skip


def median_err(x: np.ndarray, rng: np.random.Generator, n_boot: int = 500) -> tuple[float, float]:
    """Median and its bootstrap standard error (NaN for fewer than 3 values)."""
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan"), float("nan")
    boot = np.median(rng.choice(x, size=(n_boot, x.size)), axis=1)
    return float(np.median(x)), float(np.std(boot))


def true_window(path: Path, t: dict, hanning: bool) -> dict[str, float]:
    """Classical widths in the window around the true v_sys and W20, constant baseline outside it."""
    with h5py.File(path, "r") as f:
        v, y, sigma = f["spectrum/velocity"][()], f["spectrum/flux"][()], f["spectrum/sigma"][()]
    dv = float(np.median(np.diff(v)))
    half = 0.5 * t["W20"] * 1.25 + 2 * dv
    out = np.abs(v - t["v_sys"]) > half
    rms = float(np.nanmedian(sigma))
    return classic_measurements(v, y - np.median(y[out]), rms, t["v_sys"], t["W20"], hanning=hanning)


def cells(spectra: list[dict], rng: np.random.Generator) -> list[dict]:
    out = []
    for dv in DELTA_V:
        for snr in SNR:
            sel = [s for s in spectra if s["delta_v"] == dv and s["snr"] == snr]
            det = np.array([s["p_detect"] > 0.5 for s in sel])
            c: dict = {"delta_v": dv, "snr": snr, "n": len(sel), "completeness": float(det.mean())}
            for q in ("W20", "W50"):
                for est in ESTIMATORS:
                    err = np.array([s[f"{est}_{q}"] - s[f"{q}_true"] for s in sel], float)
                    pre = "" if est == "bayes" else f"{est}_"
                    for name, m in (("all", np.ones_like(det)), ("det", det), ("undet", ~det)):
                        c[f"{pre}{q}_{name}"], c[f"{pre}{q}_{name}_err"] = median_err(err[m], rng)
                    c[f"{pre}{q}_measured"] = float(np.mean(np.isfinite(err)))
            lo = np.array([s["W20_1_p16"] for s in sel], float)
            hi = np.array([s["W20_1_p84"] for s in sel], float)
            tv = np.array([s["W20_true"] for s in sel], float)
            cover = (tv >= lo) & (tv <= hi)
            c["W20_cov68_all"] = float(cover.mean())
            c["W20_cov68_det"] = float(cover[det].mean()) if det.any() else float("nan")
            c["k_gt0"] = float(np.mean([s["best_k"] > 0 for s in sel]))
            c["prior_dominated"] = float(np.mean([bool(s["flag_prior_dominated"]) for s in sel]))
            out.append(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--variants", default="default,k0,linear")
    ap.add_argument("--response", default="boxcar", help="channel response of simulation and fit")
    args = ap.parse_args()
    hanning = "hanning" in args.response.lower()
    root = Path(args.out) if args.response == "boxcar" else Path(args.out) / args.response.replace("+", "_")
    sources, truth = simulate(root, args.n, args.seed, args.response, delta_v=DELTA_V, snrs=SNR)
    for name in args.variants.split(","):
        out = root / name
        upd = VARIANTS[name]
        model = {"n_max": 1, "response": args.response, **upd.get("model", {})}
        cfg = Config().updated(model=model, rfi={"prefit": False},
                               noise={"sigma": "input"}, output={"plots": "none"},
                               priors=upd.get("priors", {}))  # fmt: skip
        run_fits(sources, cfg, out, jobs=args.jobs, resume=True)
        rows = {r["id"]: r for r in collect_rows([result_path(out, t["id"]) for t in truth])}
        spectra = []
        for t in truth:
            r = rows[t["id"]]
            s = {"id": t["id"], "delta_v": t["delta_v"], "snr": t["snr"], "W20_true": t["W20"],
                 "W50_true": t["W50"]}  # fmt: skip
            s.update({k: (np.nan if r.get(k) is None else r.get(k, np.nan)) for k in KEYS})
            tw = true_window(result_path(out, t["id"]), t, hanning)
            for est, (cat_key, tw_key) in ESTIMATORS.items():
                for q in ("W20", "W50"):
                    val = s[cat_key.format(q=q)] if cat_key else tw[tw_key.format(q=q)]
                    s[f"{est}_{q}"] = float(np.nan if val is None else val)
            spectra.append(s)
        (out / "spectra.json").write_text(json.dumps(spectra, default=float))
        table = cells(spectra, np.random.default_rng(1))
        (out / "cells.json").write_text(json.dumps(table, indent=1))
        print(f"\n== {name} (noise {NOISE} Jy, {args.n} spectra per cell)")
        print("   dv  snr compl | W20 error: all, detected, undetected | cov68 all/det | K>0  prior-dom")
        for c in table:
            errs = "  ".join(
                f"{c[f'W20_{m}']:+6.1f} ± {c[f'W20_{m}_err']:4.1f}" for m in ("all", "det", "undet")
            )
            print(
                f"{c['delta_v']:5.0f} {c['snr']:4.0f} {c['completeness']:5.2f} | {errs} | "
                f"{c['W20_cov68_all']:4.2f} / {c['W20_cov68_det']:4.2f} | "
                f"{c['k_gt0']:4.2f} {c['prior_dominated']:5.2f}"
            )
        print(
            "   dv  snr | W20 error over all spectra (fraction measured): " + ", ".join(list(ESTIMATORS)[1:])
        )
        for c in table:
            errs = "  ".join(
                f"{c[f'{e}_W20_all']:+6.1f} ± {c[f'{e}_W20_all_err']:4.1f} ({c[f'{e}_W20_measured']:.2f})"
                for e in list(ESTIMATORS)[1:]
            )
            print(f"{c['delta_v']:5.0f} {c['snr']:4.0f} | {errs}")


if __name__ == "__main__":
    main()
