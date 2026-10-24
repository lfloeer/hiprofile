"""Mini study: hiprofile fits of low-S/N ALFALFA spectra against the alpha.40 catalogue.

    python benchmarks/alfalfa_study.py --out DIR [--jobs 4] [--figure docs/img/alfalfa_study.png]

Downloads the alpha.40 catalogue (Haynes et al. 2011) from the ALFALFA archive,
draws 6 sources with HIcode 1 (S/N 6.5-9) and 4 with HIcode 2 (S/N 5-6.5),
downloads their spectra from NED (author ASCII, baseline subtracted,
Hanning smoothed), converts them to radio-convention velocities (a uniform
grid; ALFALFA tabulates cz) within +-1000 km/s of the catalogue velocity,
fits them with the default configuration and a boxcar+hanning response, and
compares W50, integrated flux and cz. Widths and fluxes are converted back
to cz units with (1 + z)^2.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

import numpy as np

C = 299792.458
NU0 = 1420.405751768
CATALOG = "https://egg.astro.cornell.edu/alfalfa/data/a40files/a40.datafile1.csv"
NED = "https://ned.ipac.caltech.edu"
NED_QUERY = (
    NED + "/cgi-bin/NEDspectra?objname=AGC+{agc}&extend=no&detail=0&preview=0&numpp=20"
    "&refcode=2011AJ....142..170H&bandpass=ANY&line=ANY"
)
CONFIG = """model:
  response: boxcar+hanning   # ALFALFA spectra are Hanning-smoothed (11 km/s resolution, 5.2 km/s channels)
  n_max: 2
output:
  plots: all
"""


def _get(url: str, out: Path | None = None) -> str:
    cmd = ["curl", "-sS", "-m", "120", "--retry", "3", url]
    if out is not None:
        subprocess.run([*cmd, "-o", str(out)], check=True)
        return ""
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def select(out: Path, seed: int = 2026) -> list[dict[str, str]]:
    cat = out / "a40.datafile1.csv"
    if not cat.exists():
        _get(CATALOG, cat)
    rows = list(csv.DictReader(cat.open()))

    def num(r: dict[str, str], k: str) -> float:
        try:
            return float(r[k])
        except ValueError:
            return np.nan

    rng = np.random.default_rng(seed)
    c1 = [r for r in rows if r["HIcode"] == "1" and 6.5 <= num(r, "SNR") <= 9 and num(r, "Vhelio") > 1500]
    c2 = [r for r in rows if r["HIcode"] == "2" and 5.0 <= num(r, "SNR") < 6.5 and num(r, "Vhelio") > 1500]
    pick = [c1[i] for i in rng.choice(len(c1), 12, replace=False)]
    pick += [c2[i] for i in rng.choice(len(c2), 8, replace=False)]
    got: list[dict[str, str]] = []
    for r in pick:
        if sum(g["HIcode"] == r["HIcode"] for g in got) >= (6 if r["HIcode"] == "1" else 4):
            continue
        raw = out / "raw" / f"AGC{r['AGCNr']}.txt"
        raw.parent.mkdir(parents=True, exist_ok=True)
        if not raw.exists():
            m = re.search(
                r'"(/spc1/2011/2011AJ\.\.\.\.142\.\.170H/[^"]+:S:HI:hgm2011\.txt)"',
                _get(NED_QUERY.format(agc=r["AGCNr"])),
            )
            if not m:
                continue
            _get(NED + m.group(1), raw)
        got.append(r)
    return got


def convert(raw: Path, dest: Path, cz: float, half_window: float = 1000.0) -> None:
    rows = []
    for line in raw.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        try:
            float(p[3])  # subtracted baseline; '*****' outside the baseline-fit region
        except ValueError:
            continue
        rows.append((float(p[1]), float(p[2])))
    a = np.array(rows)
    v = C * (1.0 - a[:, 0] / NU0)  # radio convention: uniform in frequency
    v0 = cz / (1.0 + cz / C)
    sel = np.abs(v - v0) < half_window
    o = np.argsort(v[sel])
    np.savetxt(
        dest,
        np.c_[v[sel][o], a[sel, 1][o] / 1000.0],
        header=f"{raw.stem}: radio heliocentric velocity [km/s], flux density [Jy]; ALFALFA alpha.40 via NED",
        fmt="%.4f %.6f",
    )


def compare(out: Path, sample: list[dict[str, str]]) -> list[dict[str, float | str]]:
    from hiprofile.io.results import read_result

    table: list[dict[str, float | str]] = []
    for r in sample:
        res = read_result(out / "results" / "spectra" / f"AGC{r['AGCNr']}.h5")
        n = max(res.best_n, 1)
        s = res.posteriors[n].summary()
        cz_cat = float(r["Vhelio"])
        v0 = cz_cat / (1.0 + cz_cat / C)
        i = min(range(1, n + 1), key=lambda j: abs(s[f"v_sys_{j}"] - v0))  # component of the catalogue source
        v = s[f"v_sys_{i}"]
        zf = (1.0 / (1.0 - v / C)) ** 2  # radio -> cz widths and fluxes
        row: dict[str, float | str] = {
            "agc": r["AGCNr"],
            "code": r["HIcode"],
            "snr": float(r["SNR"]),
            "best_n": res.best_n,
            "component": i,
            "p_detect": res.p_detect,
            "flags": ",".join(k for k, f in res.flags.items() if f),
            "cz_cat": cz_cat,
            "cz": v / (1.0 - v / C),
            "cz_err": 0.5 * (s[f"v_sys_{i}_p84"] - s[f"v_sys_{i}_p16"]),
            "W50_cat": float(r["W50"]),
            "W50_cat_err": float(r["errW50"]),
            "S_cat": float(r["HIflux"]),
            "S_cat_err": float(r["errflux"]),
            "W50_classic": res.classic.get("W50", np.nan) * zf,
        }
        for key in ("W50", "S"):
            row[key], row[f"{key}_p16"], row[f"{key}_p84"] = (
                s[f"{key}_{i}{x}"] * zf for x in ("", "_p16", "_p84")
            )
        table.append(row)
    return table


def figure(out: Path, table: list[dict[str, float | str]], path: Path) -> None:
    from hiprofile.io.results import read_result
    from hiprofile.plots import style

    style.apply()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.15], hspace=0.55, wspace=0.28)
    for k, row in enumerate(table):
        ax = fig.add_subplot(gs[k // 5, k % 5])
        res = read_result(out / "results" / "spectra" / f"AGC{row['agc']}.h5")
        vr = res.spectrum.velocity if hasattr(res, "spectrum") else None
        if vr is None:
            vr, _ = np.loadtxt(out / "spectra" / f"AGC{row['agc']}.txt", unpack=True)
        flux = np.loadtxt(out / "spectra" / f"AGC{row['agc']}.txt", unpack=True)[1]
        cz = vr / (1.0 - vr / C)
        mjy = 1000.0
        ax.step(cz, flux * mjy, where="mid", color=style.INK, lw=0.6)
        band = res.model_band * mjy
        ax.fill_between(cz, band[1], band[3], color=style.SERIES[0], alpha=0.35, lw=0, step="mid")
        ax.plot(cz, band[2], color=style.SERIES[0], lw=1.4)
        top = float(np.max(band[2]))
        w_cat, c_cat = float(row["W50_cat"]), float(row["cz_cat"])
        ax.plot([c_cat - w_cat / 2, c_cat + w_cat / 2], [0.5 * top] * 2, color=style.SERIES[1], lw=3,
                solid_capstyle="butt", label="ALFALFA W50")  # fmt: skip
        w, c = float(row["W50"]), float(row["cz"])
        ax.plot([c - w / 2, c + w / 2], [0.5 * top - 0.12 * top] * 2, color=style.SERIES[0], lw=3,
                solid_capstyle="butt", label="hiprofile W50")  # fmt: skip
        half = max(1.5 * max(w, w_cat), 400.0)
        ax.set_xlim(c_cat - half, c_cat + half)
        lo = np.percentile(flux * mjy, 1)
        ax.set_ylim(lo * 1.3, max(np.percentile(flux * mjy, 99.7), top) * 1.25)
        ax.set_title(
            f"AGC {row['agc']} (code {row['code']}, S/N {row['snr']:.1f})\nN = {row['best_n']}, "
            f"W50 {row['W50_cat']:.0f} / {row['W50']:.0f} km/s",
            fontsize=8.5,
            loc="left",
        )
        ax.tick_params(labelsize=7)
        if k % 5 == 0:
            ax.set_ylabel("mJy", fontsize=8)
        if k == 0:
            ax.legend(fontsize=7, loc="upper left")
    for j, (key, label, unit) in enumerate((("W50", "W50", "km/s"), ("S", "integrated flux", "Jy km/s"))):
        ax = fig.add_subplot(gs[2, 2 * j : 2 * j + 2])
        for code, mk in (("1", "o"), ("2", "s")):
            sel = [r for r in table if r["code"] == code]
            x = np.array([r[f"{key}_cat"] for r in sel], float)
            xe = np.array([r[f"{key}_cat_err"] for r in sel], float)
            y = np.array([r[key] for r in sel], float)
            ye = np.array(
                [[r[key] - r[f"{key}_p16"] for r in sel], [r[f"{key}_p84"] - r[key] for r in sel]], float
            )
            ax.errorbar(x, y, xerr=xe, yerr=ye, fmt=mk, ms=5, color=style.SERIES[0 if code == "1" else 1],
                        ecolor=style.INK_2, elinewidth=0.8, capsize=0,
                        label=f"HIcode {code} ({'S/N 6.5-9' if code == '1' else 'S/N 5-6.5'})")  # fmt: skip
        lim = max(max(float(r[key]) for r in table), max(float(r[f"{key}_cat"]) for r in table)) * 1.15
        ax.plot([0, lim], [0, lim], color=style.INK_2, lw=0.8)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel(f"ALFALFA {label} [{unit}]")
        ax.set_ylabel(f"hiprofile {label} [{unit}]")
        ax.legend(fontsize=7.5, loc="upper left")
    ax = fig.add_subplot(gs[2, 4])
    ax.axis("off")
    fig.suptitle(
        "Low-S/N ALFALFA alpha.40 sources: posterior median and 68 % band (blue) vs. the catalogue (orange)",
        x=0.01,
        ha="left",
        fontsize=11,
        y=0.98,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--figure", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample = select(out)
    (out / "spectra").mkdir(exist_ok=True)
    lines = []
    for r in sample:
        convert(
            out / "raw" / f"AGC{r['AGCNr']}.txt", out / "spectra" / f"AGC{r['AGCNr']}.txt", float(r["Vhelio"])
        )
        lines.append(f"spectra/AGC{r['AGCNr']}.txt AGC{r['AGCNr']}")
    (out / "spectra.txt").write_text("\n".join(lines) + "\n")
    (out / "alfalfa.yaml").write_text(CONFIG)
    subprocess.run(
        [
            "hiprofile",
            "fit",
            "spectra.txt",
            "--config",
            "alfalfa.yaml",
            "--jobs",
            str(args.jobs),
            "--out",
            "results",
        ],
        cwd=out,
        check=False,
    )
    table = compare(out, sample)
    (out / "comparison.json").write_text(json.dumps(table, indent=1, default=float))
    head = ("AGC", "code", "S/N", "N", "W50 ALFALFA", "W50 hiprofile", "classic", "S ALFALFA", "S hiprofile",
            "cz ALF", "cz hip", "flags")  # fmt: skip
    print(" | ".join(head))
    for r in table:
        print(
            f"{r['agc']:>7} {r['code']:>4} {r['snr']:4.1f} {r['best_n']:>2} | "
            f"{r['W50_cat']:5.0f} ± {r['W50_cat_err']:<3.0f} "
            f"{r['W50']:6.0f} [{r['W50_p16']:4.0f}, {r['W50_p84']:4.0f}] {r['W50_classic']:7.0f} | "
            f"{r['S_cat']:4.2f} ± {r['S_cat_err']:4.2f} "
            f"{r['S']:5.2f} [{r['S_p16']:4.2f}, {r['S_p84']:4.2f}] | "
            f"{r['cz_cat']:6.0f} {r['cz']:6.0f} ± {r['cz_err']:<3.0f} | {r['flags']}"
        )
    if args.figure:
        figure(out, table, Path(args.figure))


if __name__ == "__main__":
    main()
