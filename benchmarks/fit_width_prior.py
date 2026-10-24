"""Calibrate the Delta v prior on a catalogue of measured W50.

    python benchmarks/fit_width_prior.py --out DIR [--catalog FILE --w50-column W50 --cz-column Vhelio]
                                         [--figure docs/img/width_prior.png]

Without ``--catalog`` the ALFALFA alpha.100 catalogue (Haynes et al. 2018) is
downloaded and its HIcode 1 detections are used. W50 is a combination of the
rotational width Delta v and the random motions Delta v_rand (and depends on
alpha and f), so Delta v is not histogrammed directly: profiles are drawn
with Delta v from a candidate prior and Delta v_rand, alpha, f from the
default priors, their W50 is computed with the model, and the candidate's
parameters are fitted so that the quantiles of log10 W50 match the
catalogue's. Gamma and log-normal candidates are fitted; the result is the
prior of the *detected* population of that survey, selection included,
which is what a fit of a spectrum from that survey needs.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy import optimize, stats

from hiprofile.config import Priors
from hiprofile.inference.derived import profile_derived
from hiprofile.inference.priors import build_prior
from hiprofile.model import SpectralGrid

ALFALFA = "https://egg.astro.cornell.edu/alfalfa/data/a100files/a100.code12.table2.190808.csv"
C = 299792.458
QUANTILES = np.array([2.5, 5, 10, 16, 25, 35, 50, 65, 75, 84, 90, 95, 97.5])
HIGH = 800.0


def load_catalog(path: Path, w_col: str, cz_col: str | None, code_col: str | None) -> np.ndarray:
    """log10 of the rest-frame W50 (observed W50 / (1 + z) when a velocity column is given)."""
    out = []
    for r in csv.DictReader(path.open()):
        if code_col and r.get(code_col, "").strip() not in ("1",):
            continue
        try:
            w = float(r[w_col])
            cz = float(r[cz_col]) if cz_col else 0.0
        except (KeyError, ValueError):
            continue
        if w > 0:
            out.append(np.log10(w / (1.0 + cz / C)))
    return np.array(out)


class ForwardModel:
    """log10 W50 of profiles with a candidate Delta v and the default priors for the rest."""

    def __init__(self, n: int = 4000, seed: int = 1) -> None:
        v = -1200.0 + 2.0 * np.arange(1200)
        self.grid = SpectralGrid.from_velocities(v)
        rng = np.random.default_rng(seed)
        self.base = build_prior(("profile",), Priors(), v).transform(rng.uniform(size=(n, 5)))
        self.u = rng.uniform(size=n)  # common random numbers: smooth objective

    def delta_v(self, dist) -> np.ndarray:
        return dist.ppf(self.u * dist.cdf(HIGH))  # truncated at HIGH

    def log_w50(self, dist) -> np.ndarray:
        th = self.base[:, :4].copy()
        th[:, 0] = 0.0
        th[:, 1] = self.delta_v(dist)
        f = self.base[:, 4]
        return np.log10(profile_derived(self.grid, th, np.stack([1 - f, f], axis=1))["W50"])


def candidates() -> dict[str, tuple]:
    return {
        "gamma": (lambda p: stats.gamma(p[0], scale=p[1]), [2.5, 70.0]),
        "lognormal": (lambda p: stats.lognorm(p[1] * np.log(10), scale=10 ** p[0]), [2.2, 0.3]),
    }


def fit(obs: np.ndarray, model: ForwardModel) -> dict[str, dict]:
    q_obs = np.percentile(obs, QUANTILES)
    out = {}
    for name, (make, start) in candidates().items():

        def loss(p, make=make):
            if np.any(np.asarray(p) <= 0):
                return 1e9
            return float(np.sum((np.percentile(model.log_w50(make(p)), QUANTILES) - q_obs) ** 2))

        res = optimize.minimize(loss, start, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-8})
        w = model.log_w50(make(res.x))
        dv = model.delta_v(make(res.x))
        out[name] = {
            "params": [float(x) for x in res.x],
            "rms_quantile_misfit_dex": float(np.sqrt(res.fun / QUANTILES.size)),
            "ks_distance": float(stats.ks_2samp(w, obs).statistic),
            "delta_v_median": float(np.median(dv)),
            "delta_v_p16_p84": [float(x) for x in np.percentile(dv, [16, 84])],
        }
    return out


def figure(obs: np.ndarray, model: ForwardModel, result: dict, path: Path, label: str) -> None:
    from hiprofile.plots import style

    style.apply()
    import matplotlib.pyplot as plt

    makes = {k: v[0] for k, v in candidates().items()}
    current = stats.lognorm(0.4 * np.log(10), scale=10**2.2)
    curves = [
        ("previous default: log-normal 2.2 ± 0.4 dex", current),
        (
            "log-normal fitted: {:.2f} ± {:.2f} dex".format(*result["lognormal"]["params"]),
            makes["lognormal"](result["lognormal"]["params"]),
        ),
        (
            "gamma fitted: k = {:.2f}, θ = {:.0f} km/s".format(*result["gamma"]["params"]),
            makes["gamma"](result["gamma"]["params"]),
        ),
    ]
    bins = np.linspace(1.0, 3.1, 43)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.hist(obs, bins=bins, density=True, color=style.INK_2, alpha=0.25, label=f"{label} ({obs.size})")
    for i, (name, dist) in enumerate(curves):
        h, e = np.histogram(model.log_w50(dist), bins=bins, density=True)
        ax.step(0.5 * (e[1:] + e[:-1]), h, where="mid", color=style.SERIES[i], lw=2, label=name)
    ax.set_xlabel("log₁₀ W50 [km/s]")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("W50 predicted by each Δv prior (default Δv_rand, α, f)", loc="left", fontsize=9.5)
    ax = axes[1]
    x = np.linspace(0, HIGH, 801)
    for i, (name, dist) in enumerate(curves):
        ax.plot(x, dist.pdf(x) / dist.cdf(HIGH), color=style.SERIES[i], lw=2, label=name.split(":")[0])
    ax.set_xlabel("Δv [km/s]")
    ax.set_ylabel("prior density")
    ax.legend(fontsize=8)
    ax.set_title("the Δv priors", loc="left", fontsize=9.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--catalog", help="CSV with a W50 column (default: download ALFALFA alpha.100)")
    ap.add_argument("--w50-column", default="W50")
    ap.add_argument("--cz-column", default="Vhelio", help="heliocentric cz [km/s] for the (1 + z) correction")
    ap.add_argument("--code-column", default="HIcode", help="keep rows with value 1; empty string: keep all")
    ap.add_argument("--label", default="ALFALFA α.100, HIcode 1")
    ap.add_argument("--figure", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cat = Path(args.catalog) if args.catalog else out / "a100.csv"
    if not cat.exists():
        subprocess.run(["curl", "-sS", "-m", "300", "--retry", "3", "-o", str(cat), ALFALFA], check=True)
    obs = load_catalog(cat, args.w50_column, args.cz_column or None, args.code_column or None)
    model = ForwardModel()
    result = fit(obs, model)
    result["catalog"] = {
        "n": int(obs.size),
        "log_w50_quantiles": dict(
            zip(QUANTILES.tolist(), np.percentile(obs, QUANTILES).tolist(), strict=True)
        ),
    }
    (out / "width_prior.json").write_text(json.dumps(result, indent=1))
    print(f"{obs.size} W50 values; median {10 ** np.median(obs):.0f} km/s")
    for name in candidates():
        r = result[name]
        print(
            f"{name:10s} params {np.round(r['params'], 3)}  misfit {r['rms_quantile_misfit_dex']:.3f} dex  "
            f"KS {r['ks_distance']:.3f}  Delta v median {r['delta_v_median']:.0f}, "
            f"16-84 % {np.round(r['delta_v_p16_p84'])}"
        )
    if args.figure:
        figure(obs, model, result, Path(args.figure), args.label)


if __name__ == "__main__":
    main()
