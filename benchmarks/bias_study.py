"""Bias of the posterior median as a function of the true line width and S/N.

    python benchmarks/bias_study.py --out DIR [--n 20] [--jobs 4] [--priors default|flat]
                                    [--response boxcar+hanning]

Simulates N = 1 spectra on a grid of true W50 (via Delta v) and integrated
S/N, with the other parameters at typical values, fits them with the given
prior set and tabulates the median error of W50, W20, S and v_sys per cell,
for the posterior median and for the two classical measurements (naive
threshold crossing and Springob et al. 2005 edge fits). Unlike the SBC,
the simulated population is *not* drawn from the prior, so shrinkage toward
the prior shows up as bias.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hiprofile import Spectrum
from hiprofile.config import Config
from hiprofile.io.catalog import collect_rows
from hiprofile.io.results import result_path
from hiprofile.io.spectra import SpectrumSource, write_spectrum
from hiprofile.model import SpectralGrid, profile
from hiprofile.runner import run_fits

DELTA_V = (20.0, 50.0, 100.0, 200.0, 400.0)
SNR = (3.0, 5.0, 8.0, 15.0, 40.0)
SIGMA_RAND, ALPHA, F = 12.0, 0.1, 0.25
NOISE = 0.01

PRIOR_SETS = {
    "default": {},
    "flat": {
        "delta_v": {"kind": "loguniform", "low": 5.0, "high": 800.0},
        "delta_v_rand": {"kind": "loguniform", "low": 4.0, "high": 60.0},
        "alpha": {"kind": "uniform", "low": -1.0, "high": 1.0},
        "f": {"kind": "uniform", "low": 0.0, "high": 1.0},
    },
}


def simulate(
    out: Path, n: int, seed: int, response: str = "boxcar"
) -> tuple[list[SpectrumSource], list[dict]]:
    rng = np.random.default_rng(seed)
    v = 1000.0 + 5.0 * np.arange(512)
    grid = SpectralGrid.from_velocities(v, supersample=8, response=response)
    hanning = "hanning" in response.lower()
    sdir = out / "sims"
    sdir.mkdir(parents=True, exist_ok=True)
    sources, truth = [], []
    for dv in DELTA_V:
        for snr in SNR:
            for i in range(n):
                vc = rng.uniform(1800, 2400)
                w_eff = dv + 2.35 * SIGMA_RAND
                s = snr * NOISE * 5.0 * np.sqrt(w_eff / 5.0)
                if hanning:
                    # white noise smoothed like the line: the integrated S/N stays that of the raw
                    # spectrum, the channel rms drops to sqrt(0.375) of it
                    eps = rng.standard_normal(v.size + 2)
                    noise = np.convolve(eps, [0.25, 0.5, 0.25], "valid")
                else:
                    noise = rng.standard_normal(v.size)
                y = profile(grid, [vc, dv, SIGMA_RAND, ALPHA, s, F]) + NOISE * noise
                sid = f"dv{int(dv):03d}_snr{snr:04.1f}_{i:03d}"
                rms = NOISE * (np.sqrt(0.375) if hanning else 1.0)
                spec = Spectrum(v, y, sigma=np.full(v.size, rms), id=sid)
                p = sdir / f"{sid}.h5"
                write_spectrum(spec, p)
                sources.append(SpectrumSource(str(p), sid))
                from hiprofile.inference.derived import profile_derived

                d = profile_derived(
                    grid, np.array([[vc, dv, SIGMA_RAND, ALPHA]]), np.array([[s * (1 - F), s * F]])
                )
                truth.append({"id": sid, "delta_v": dv, "snr": snr, "S": s, "v_ctr": vc,
                              **{k: float(d[k][0]) for k in ("W50", "W20", "v_sys")}})  # fmt: skip
    return sources, truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--priors", default="default", choices=list(PRIOR_SETS))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--response", default="boxcar", help="channel response of simulation and fit")
    args = ap.parse_args()
    suffix = "" if args.response == "boxcar" else "_" + args.response.replace("+", "_")
    out = Path(args.out) / f"bias_{args.priors}{suffix}"
    sources, truth = simulate(out, args.n, args.seed, args.response)
    cfg = Config().updated(model={"n_max": 1, "response": args.response}, rfi={"prefit": False},
                           noise={"sigma": "input"}, output={"plots": "none"},
                           priors=PRIOR_SETS[args.priors])  # fmt: skip
    run_fits(sources, cfg, out, jobs=args.jobs, resume=True)
    rows = {r["id"]: r for r in collect_rows([result_path(out, t["id"]) for t in truth])}
    table = []
    for dv in DELTA_V:
        for snr in SNR:
            sel = [t for t in truth if t["delta_v"] == dv and t["snr"] == snr]
            cell = {"delta_v": dv, "snr": snr, "n": len(sel)}
            det = np.array([rows[t["id"]].get("p_detect", 0.0) > 0.5 for t in sel])
            cell["completeness"] = float(det.mean())
            for q in ("W50", "W20", "S", "v_sys"):
                tv = np.array([t[q] for t in sel])
                estimators = [("bayes", f"{q}_1"), ("classic", f"classic_{q}")]
                if q != "S":
                    estimators.append(("springob", f"classic_{q}_springob"))
                for label, key in estimators:
                    est = np.array([float(rows[t["id"]].get(key, np.nan) or np.nan) for t in sel])
                    ok = det & np.isfinite(est)
                    cell[f"{label}_{q}_bias"] = (
                        float(np.median(est[ok] - tv[ok])) if ok.sum() >= 3 else float("nan")
                    )
                    cell[f"{label}_{q}_scatter"] = (
                        float(1.4826 * np.median(np.abs(est[ok] - tv[ok] - np.median(est[ok] - tv[ok]))))
                        if ok.sum() >= 3
                        else float("nan")
                    )
                lo = np.array([float(rows[t["id"]].get(f"{q}_1_p16", np.nan) or np.nan) for t in sel])
                hi = np.array([float(rows[t["id"]].get(f"{q}_1_p84", np.nan) or np.nan) for t in sel])
                ok = det & np.isfinite(lo)
                cell[f"{q}_coverage68"] = (
                    float(np.mean((tv[ok] >= lo[ok]) & (tv[ok] <= hi[ok]))) if ok.any() else float("nan")
                )
            cons = np.array([float(rows[t["id"]].get("m_constraint_delta_v", np.nan) or np.nan) for t in sel])
            cell["constraint_delta_v"] = (
                float(np.nanmedian(cons)) if np.isfinite(cons).any() else float("nan")
            )
            cell["prior_dominated"] = float(
                np.mean([bool(rows[t["id"]].get("flag_prior_dominated", False)) for t in sel])
            )
            table.append(cell)
    (out / "bias_table.json").write_text(json.dumps(table, indent=1))
    head = ("dv", "snr", "compl", "W50 bias bayes/naive/Springob", "W20 bias", "cov68 W50", "prior-dom")
    print(" ".join(f"{h:>{w}}" for h, w in zip(head, (5, 5, 6, 30, 24, 10, 9), strict=True)))
    for c in table:
        w50 = " / ".join(f"{c[f'{e}_W50_bias']:+8.1f}" for e in ("bayes", "classic", "springob"))
        w20 = " / ".join(f"{c[f'{e}_W20_bias']:+6.1f}" for e in ("bayes", "classic", "springob"))
        print(
            f"{c['delta_v']:5.0f} {c['snr']:5.1f} {c['completeness']:6.2f} {w50:>30} {w20:>24} "
            f"{c['W50_coverage68']:10.2f} {c['prior_dominated']:9.2f}"
        )


if __name__ == "__main__":
    main()
