"""Refit the ALFALFA alpha.40 catalogue (Haynes et al. 2011) and compare.

    python benchmarks/alfalfa_survey.py fit --out DIR [--n 1000 | --n 0] [--seed 2027] [--jobs 4]
    python benchmarks/alfalfa_survey.py compare --out DIR [--catalog FILE.csv] [--figures DIR]

``fit`` draws ``--n`` random sources with HIcode 1 or 2 (``--n 0``: all of
them; code 9, the high-velocity clouds, is left out), downloads their
spectra from NED (Hanning smoothed, baseline subtracted, 5.2 km/s
channels), cuts +-1000 km/s around the catalogue velocity (but nothing
below 350 km/s, where Galactic HI is) and fits them with
``model.response: boxcar+hanning``, N <= 2 and no per-spectrum plots.
Downloads and fits resume where they stopped.

``compare`` writes one row per source: the catalogue values, the posterior
median and 68 % interval of the component nearest the catalogue velocity
(converted to the cz scale of the catalogue), P(detect), N, flags and both
classical widths measured around that component.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from alfalfa_study import CATALOG, NED, NED_QUERY, C, _get, convert

CONFIG = """model:
  response: boxcar+hanning   # ALFALFA spectra are Hanning-smoothed (5.2 km/s channels)
  n_max: 2
output:
  plots: none
"""
NED_DIR = NED + "/spc1/2011/2011AJ....142..170H"
# Galactic HI and the high-velocity clouds reach a few hundred km/s and are
# Jy-bright; channels below this radio velocity are cut from the window.
V_MIN = 350.0


def ned_name(agc: int) -> str:
    """NED file name of an alpha.40 spectrum: UGC numbers below 13000, AGC numbers above."""
    return f"UGC_{agc:05d}" if agc < 13000 else f"AGC_{agc}"


def download(agc: int, raw: Path) -> bool:
    if raw.exists() and raw.stat().st_size > 1000:
        return True
    subdir = f"A{agc:06d}"[:3]  # A + first two digits of the zero-padded AGC number
    url = f"{NED_DIR}/{subdir}/{ned_name(agc)}:S:HI:hgm2011.txt"
    code = subprocess.run(
        ["curl", "-sS", "-m", "120", "--retry", "3", "-o", str(raw), "-w", "%{http_code}", url],
        capture_output=True, text=True, check=False,
    ).stdout  # fmt: skip
    if code == "200" and raw.stat().st_size > 1000:
        return True
    raw.unlink(missing_ok=True)
    # names that do not follow the pattern (e.g. UGC_008015): ask the spectrum search
    queries = [NED_QUERY.format(agc=agc)]
    if agc < 13000:
        queries.append(NED_QUERY.replace("AGC+", "UGC+").format(agc=agc))
    for query in queries:
        try:
            m = re.search(r'"(/spc1/2011/2011AJ\.\.\.\.142\.\.170H/[^"]+:S:HI:hgm2011\.txt)"', _get(query))
            if m:
                _get(NED + m.group(1), raw)
                return raw.exists() and raw.stat().st_size > 1000
        except subprocess.CalledProcessError:
            pass
    return False


def sample(out: Path, n: int, seed: int) -> list[dict[str, str]]:
    cat = out / "a40.datafile1.csv"
    if not cat.exists():
        _get(CATALOG, cat)
    rows = [r for r in csv.DictReader(cat.open()) if r["HIcode"].strip() in ("1", "2")]
    for r in rows:
        r["AGCNr"] = r["AGCNr"].strip()
    if 0 < n < len(rows):
        rows = [rows[i] for i in sorted(np.random.default_rng(seed).choice(len(rows), n, replace=False))]
    return rows


def fit(args: argparse.Namespace) -> None:
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "spectra").mkdir(exist_ok=True)
    rows = sample(out, args.n, args.seed)
    with ThreadPoolExecutor(args.downloads) as ex:
        ok = list(ex.map(lambda r: download(int(r["AGCNr"]), out / "raw" / f"AGC{r['AGCNr']}.txt"), rows))
    missing = [r["AGCNr"] for r, good in zip(rows, ok, strict=True) if not good]
    print(f"{len(rows)} sources, {len(rows) - len(missing)} spectra, {len(missing)} not in NED")
    lines = []
    for r, good in zip(rows, ok, strict=True):
        if not good:
            continue
        dest = out / "spectra" / f"AGC{r['AGCNr']}.txt"
        if not dest.exists():
            convert(out / "raw" / f"AGC{r['AGCNr']}.txt", dest, float(r["Vhelio"]), v_min=V_MIN)
        lines.append(f"spectra/AGC{r['AGCNr']}.txt AGC{r['AGCNr']}")
    (out / "spectra.txt").write_text("\n".join(lines) + "\n")
    (out / "sample.json").write_text(json.dumps({"n": args.n, "seed": args.seed, "missing": missing,
                                                 "agc": [r["AGCNr"] for r in rows]}))  # fmt: skip
    (out / "alfalfa.yaml").write_text(CONFIG)
    subprocess.run(["hiprofile", "fit", "spectra.txt", "--config", "alfalfa.yaml", "--jobs", str(args.jobs),
                    "--out", "results", "--resume"], cwd=out, check=False)  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--out", required=True)
    f.add_argument("--n", type=int, default=1000, help="random sources with HIcode 1 or 2; 0: all")
    f.add_argument("--seed", type=int, default=2027)
    f.add_argument("--jobs", type=int, default=4)
    f.add_argument("--downloads", type=int, default=4, help="parallel NED downloads")
    c = sub.add_parser("compare")
    c.add_argument("--out", required=True)
    c.add_argument("--catalog", default=None, help="CSV with one row per source")
    c.add_argument("--figures", default=None, help="directory for the comparison figures")
    args = ap.parse_args()
    if args.cmd == "fit":
        fit(args)
    else:
        compare(args)


def _num(r: dict[str, str], k: str) -> float:
    try:
        return float(r[k])
    except (KeyError, ValueError):
        return float("nan")


def compare_row(path: Path, r: dict[str, str]) -> dict[str, float | str]:
    """Catalogue values and the fit of the component nearest the catalogue velocity, on the cz scale."""
    from hiprofile.inference.derived import classic_measurements
    from hiprofile.io.results import read_result

    cz_cat = _num(r, "Vhelio")
    row: dict[str, float | str] = {
        "agc": r["AGCNr"].strip(), "code": r["HIcode"].strip(),
        "snr_cat": _num(r, "SNR"), "rms_cat": _num(r, "RMS"),
        "cz_cat": cz_cat, "W50_cat": _num(r, "W50"), "W50_cat_err": _num(r, "errW50"),
        "S_cat": _num(r, "HIflux"), "S_cat_err": _num(r, "errflux"),
    }  # fmt: skip
    res = read_result(path)
    row.update({"status": res.status, "p_detect": res.p_detect, "best_n": res.best_n, "best_k": res.best_k,
                "runtime": res.runtime, "flagged": any(res.flags.values()),
                "flags": ",".join(k for k, f in res.flags.items() if f)})  # fmt: skip
    if res.status != "ok":
        return row
    n = max(res.best_n, 1)
    s = res.posteriors[n].summary()
    v0 = cz_cat / (1.0 + cz_cat / C)  # radio velocity of the catalogue source
    i = min(range(1, n + 1), key=lambda j: abs(s[f"v_sys_{j}"] - v0))
    v = s[f"v_sys_{i}"]
    zf = (1.0 / (1.0 - v / C)) ** 2  # d(cz)/d(v_radio): widths and fluxes on the cz scale
    row["component"] = i
    row["cz"] = v / (1.0 - v / C)
    for q in ("v_sys",):
        lo, hi = s[f"{q}_{i}_p16"], s[f"{q}_{i}_p84"]
        row["cz_p16"], row["cz_p84"] = lo / (1.0 - lo / C), hi / (1.0 - hi / C)
    for q in ("W50", "W20", "S"):
        for x in ("", "_p16", "_p84"):
            row[f"{q}{x}"] = s[f"{q}_{i}{x}"] * zf
    for q in ("delta_v", "delta_v_rand", "alpha", "f"):
        row[q] = s[f"{q}_{i}"]
    row["delta_v"] = float(row["delta_v"]) * zf
    row["constraint_delta_v"] = res.metrics.get("constraint_delta_v", float("nan"))
    rms = float(np.nanmedian(res.sigma))
    row["rms"] = rms * 1000.0  # mJy
    cl = classic_measurements(res.velocity, res.flux - res.baseline_median, rms, v, s[f"W20_{i}"],
                              good=~np.asarray(res.mask, bool), hanning=True)  # fmt: skip
    for k_out, k_in in (("W50_naive", "W50"), ("W20_naive", "W20"), ("S_naive", "S"),
                        ("W50_springob", "W50_springob"), ("W20_springob", "W20_springob")):  # fmt: skip
        row[k_out] = cl[k_in] * zf
    return row


def robust(x: np.ndarray) -> tuple[float, float, int]:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan"), float("nan"), int(x.size)
    m = float(np.median(x))
    return m, float(1.4826 * np.median(np.abs(x - m))), int(x.size)


SNR_BINS = ((5.0, 6.5), (6.5, 8.0), (8.0, 12.0), (12.0, 20.0), (20.0, 50.0), (50.0, np.inf))


def summarise(table: list[dict]) -> dict:
    def col(k: str, rows: list[dict]) -> np.ndarray:
        return np.array([float(r.get(k, np.nan)) for r in rows], float)

    ok = [r for r in table if r["status"] == "ok"]
    out: dict = {"n": len(table), "fitted": len(ok), "failed": len(table) - len(ok)}
    for code in ("1", "2"):
        rows = [r for r in ok if r["code"] == code]
        det = col("p_detect", rows) > 0.5
        out[f"code{code}"] = {
            "n": len(rows),
            "detected": float(det.mean()),
            "best_n": {str(k): int(np.sum(col("best_n", rows) == k)) for k in (0, 1, 2)},
            "flagged": float(np.mean([bool(r["flagged"]) for r in rows])),
        }
    out["by_snr"] = []
    for lo, hi in SNR_BINS:
        rows = [r for r in ok if lo <= float(r["snr_cat"]) < hi]
        det = [r for r in rows if float(r["p_detect"]) > 0.5]
        cell: dict = {"snr": [lo, hi], "n": len(rows), "detected": len(det) / max(len(rows), 1)}
        for est in ("W50", "W50_naive", "W50_springob"):
            cell[f"{est}_rel"] = robust(col(est, det) / col("W50_cat", det) - 1.0)
        cell["S_rel"] = robust(col("S", det) / col("S_cat", det) - 1.0)
        cell["cz_diff"] = robust(col("cz", det) - col("cz_cat", det))
        sig = np.hypot(0.5 * (col("W50_p84", det) - col("W50_p16", det)), col("W50_cat_err", det))
        cell["W50_pull"] = robust((col("W50", det) - col("W50_cat", det)) / sig)
        out["by_snr"].append(cell)
    return out


def compare(args: argparse.Namespace) -> None:
    out = Path(args.out)
    info = json.loads((out / "sample.json").read_text())
    want = set(info["agc"]) - set(info["missing"])
    rows = [r for r in csv.DictReader((out / "a40.datafile1.csv").open()) if r["AGCNr"].strip() in want]
    table = []
    for r in rows:
        p = out / "results" / "spectra" / f"AGC{r['AGCNr'].strip()}.h5"
        if p.exists():
            table.append(compare_row(p, r))
    keys = list(dict.fromkeys(k for row in table for k in row))
    target = Path(args.catalog) if args.catalog else out / "comparison.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        for row in table:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    summary = summarise(table)
    summary["commit"] = (out / "commit.txt").read_text().strip() if (out / "commit.txt").exists() else ""
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {target} ({len(table)} rows)")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_snr"}, indent=1))
    print("S/N        n  det  | W50/cat-1: fit, naive, Springob (median, scatter) | S/cat-1 | dcz | W50 pull")
    for c in summary["by_snr"]:
        ws = "  ".join(
            f"{c[f'{e}_rel'][0]:+.3f} {c[f'{e}_rel'][1]:.3f}" for e in ("W50", "W50_naive", "W50_springob")
        )
        print(f"{c['snr'][0]:4.0f}-{c['snr'][1]:<4.0f} {c['n']:4d} {c['detected']:.2f} | {ws} | "
              f"{c['S_rel'][0]:+.3f} {c['S_rel'][1]:.3f} | {c['cz_diff'][0]:+.1f} {c['cz_diff'][1]:.1f} | "
              f"{c['W50_pull'][0]:+.2f} {c['W50_pull'][1]:.2f}")  # fmt: skip
    if args.figures:
        figures(table, summary, Path(args.figures))
        outlier_figure(out, table, Path(args.figures) / "alfalfa_survey_outliers.png")


def _binned(ax, x: np.ndarray, y: np.ndarray, edges: np.ndarray, color: str, label: str) -> None:
    """Median and 16-84 % range of y in bins of x (bins with at least 5 points)."""
    mid, q = [], []
    for lo, hi in itertools.pairwise(edges):
        m = (x >= lo) & (x < hi) & np.isfinite(y)
        if m.sum() >= 5:
            mid.append(np.sqrt(lo * hi))
            q.append(np.percentile(y[m], [16, 50, 84]))
    if not q:
        return
    qa = np.array(q)
    ax.plot(mid, qa[:, 1], "-o", color=color, lw=2, ms=4, label=label)
    ax.fill_between(mid, qa[:, 0], qa[:, 2], color=color, alpha=0.15, lw=0)


def figures(table: list[dict], summary: dict, outdir: Path) -> None:
    from hiprofile.plots import style

    style.apply()
    import matplotlib.pyplot as plt

    def col(k: str, rows: list[dict]) -> np.ndarray:
        return np.array([float(r.get(k, np.nan)) for r in rows], float)

    ok = [r for r in table if r["status"] == "ok"]
    det = [r for r in ok if float(r["p_detect"]) > 0.5]
    snr = col("snr_cat", det)
    edges = np.geomspace(5.0, max(snr.max(), 60.0) * 1.01, 10)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ax = axes[0, 0]
    for code, color in (("1", style.SERIES[0]), ("2", style.SERIES[1])):
        rows = [r for r in det if r["code"] == code]
        ax.scatter(col("W50_cat", rows), col("W50", rows), s=6, color=color, alpha=0.5, edgecolor="none",
                   label=f"HIcode {code} ({len(rows)})")  # fmt: skip
    ax.plot([10, 1000], [10, 1000], color=style.INK_2, lw=0.8)
    ax.set(xscale="log", yscale="log", xlim=(10, 1000), ylim=(10, 1000),
           xlabel="ALFALFA W50 [km/s]", ylabel="hiprofile W50 [km/s]")  # fmt: skip
    ax.set_title("W50 of the detected sources", loc="left", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[0, 1]
    for key, color, label in (("W50", style.SERIES[0], "hiprofile posterior median"),
                              ("W50_naive", style.SERIES[1], "naive threshold crossing"),
                              ("W50_springob", style.SERIES[2], "Springob et al. (2005)")):  # fmt: skip
        _binned(ax, snr, col(key, det) / col("W50_cat", det) - 1.0, edges, color, label)
    ax.axhline(0, color=style.INK_2, lw=0.8)
    ax.set(xscale="log", ylim=(-0.4, 0.4), xlabel="ALFALFA S/N", ylabel="W50 / W50 ALFALFA − 1")
    ax.set_title("W50 against the catalogue: median and 16–84 %", loc="left", fontsize=9.5)
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[0, 2]
    _binned(
        ax, snr, col("S", det) / col("S_cat", det) - 1.0, edges, style.SERIES[0], "hiprofile posterior median"
    )
    _binned(
        ax, snr, col("S_naive", det) / col("S_cat", det) - 1.0, edges, style.SERIES[1], "sum over the line"
    )
    ax.axhline(0, color=style.INK_2, lw=0.8)
    ax.set(xscale="log", ylim=(-0.5, 0.5), xlabel="ALFALFA S/N", ylabel="S / S ALFALFA − 1")
    ax.set_title("integrated flux against the catalogue", loc="left", fontsize=9.5)
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1, 0]
    bins = np.linspace(-40, 40, 41)
    for code, color in (("1", style.SERIES[0]), ("2", style.SERIES[1])):
        rows = [r for r in det if r["code"] == code]
        d = np.clip(col("cz", rows) - col("cz_cat", rows), -40, 40)
        ax.hist(d, bins=bins, histtype="step", lw=2, color=color, density=True, label=f"HIcode {code}")
    ax.set(xlabel="cz hiprofile − cz ALFALFA [km/s] (clipped at ±40)", ylabel="density")
    ax.set_title("systemic velocity", loc="left", fontsize=9.5)
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    snr_all = col("snr_cat", ok)
    e2 = np.geomspace(5.0, max(snr_all.max(), 60.0) * 1.01, 10)
    for code, color in (("1", style.SERIES[0]), ("2", style.SERIES[1])):
        rows = [r for r in ok if r["code"] == code]
        x, pd, bn = col("snr_cat", rows), col("p_detect", rows), col("best_n", rows)
        mids, fd, f2 = [], [], []
        for lo, hi in itertools.pairwise(e2):
            m = (x >= lo) & (x < hi)
            if m.sum() >= 5:
                mids.append(np.sqrt(lo * hi))
                fd.append(np.mean(pd[m] > 0.5))
                f2.append(np.mean(bn[m] == 2))
        ax.plot(mids, fd, "-o", color=color, lw=2, ms=4, label=f"HIcode {code}: P(detect) > 0.5")
        ax.plot(mids, f2, "--", color=color, lw=1.5, label=f"HIcode {code}: N = 2 preferred")
    ax.set(xscale="log", ylim=(-0.02, 1.02), xlabel="ALFALFA S/N", ylabel="fraction")
    ax.set_title("detections and second components", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="center right")
    ax = axes[1, 2]
    sig = np.hypot(0.5 * (col("W50_p84", det) - col("W50_p16", det)), col("W50_cat_err", det))
    pull = (col("W50", det) - col("W50_cat", det)) / sig
    ax.hist(np.clip(pull, -6, 6), bins=np.linspace(-6, 6, 49), density=True, color=style.SERIES[0], alpha=0.6,
            label="hiprofile − ALFALFA, in combined errors")  # fmt: skip
    g = np.linspace(-6, 6, 200)
    ax.plot(g, np.exp(-0.5 * g**2) / np.sqrt(2 * np.pi), color=style.INK_2, lw=1, label="unit normal")
    ax.set(xlabel="W50 pull (clipped at ±6)", ylabel="density")
    ax.set_title("W50 difference in units of the combined error", loc="left", fontsize=9.5)
    ax.legend(fontsize=8)
    fig.suptitle(f"ALFALFA α.40 refit with hiprofile: {len(ok)} sources, {len(det)} detected",
                 x=0.01, ha="left", fontsize=11, y=1.0)  # fmt: skip
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "alfalfa_survey.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def outlier_figure(out: Path, table: list[dict], path: Path, n: int = 12) -> None:
    """Spectra of the detected sources whose W50 differs most from the catalogue."""
    from hiprofile.io.results import read_result
    from hiprofile.plots import style

    style.apply()
    import matplotlib.pyplot as plt

    det = [r for r in table if r["status"] == "ok" and float(r["p_detect"]) > 0.5]
    det.sort(key=lambda r: -abs(np.log(float(r["W50"]) / float(r["W50_cat"]))))
    fig, axes = plt.subplots(3, 4, figsize=(16, 9.5))
    for ax, row in zip(axes.flat, det[:n], strict=False):
        res = read_result(out / "results" / "spectra" / f"AGC{row['agc']}.h5")
        cz = res.velocity / (1.0 - res.velocity / C)
        mjy = 1000.0
        ax.step(cz, res.flux * mjy, where="mid", color=style.INK, lw=0.6)
        band = res.model_band * mjy
        ax.fill_between(cz, band[1], band[3], color=style.SERIES[0], alpha=0.35, lw=0, step="mid")
        ax.plot(cz, band[2], color=style.SERIES[0], lw=1.4)
        top = float(np.max(band[2])) or 1.0
        marks = (
            ((row["W50_cat"], row["cz_cat"]), style.SERIES[1], 0.5, "ALFALFA W50"),
            ((row["W50"], row["cz"]), style.SERIES[0], 0.38, "hiprofile W50"),
        )
        for (w, c), color, y, lab in marks:
            w, c = float(w), float(c)
            ax.plot(
                [c - w / 2, c + w / 2], [y * top] * 2, color=color, lw=3, solid_capstyle="butt", label=lab
            )
        ax.set_xlim(cz.min(), cz.max())
        head = f"AGC {row['agc']} (code {row['code']}, S/N {float(row['snr_cat']):.1f}), N = {row['best_n']}"
        widths = f"W50 {float(row['W50_cat']):.0f} / {float(row['W50']):.0f} km/s"
        ax.set_title(f"{head}\n{widths}; {row['flags'] or 'no flags'}", fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    for ax in axes[:, 0]:
        ax.set_ylabel("mJy", fontsize=8)
    fig.suptitle(
        "Largest W50 differences: spectrum, posterior median and 68 % band (blue), "
        "catalogue (orange) and hiprofile (blue) W50 at half height",
        x=0.01, ha="left", fontsize=11,
    )  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
