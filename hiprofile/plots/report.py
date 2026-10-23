"""HTML report over a catalogue: distributions, flag statistics, comparison with
classical W50/W20 and a searchable list of flagged spectra with previews."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Any

import numpy as np

from hiprofile import __version__
from hiprofile.plots import style

style.apply()
import matplotlib.pyplot as plt  # noqa: E402

DIST_COLUMNS = (
    "p_detect",
    "S_1",
    "W50_1",
    "W20_1",
    "v_sys_1",
    "delta_v_1",
    "delta_v_rand_1",
    "alpha_1",
    "f_1",
    "m_chi2_red",
    "m_ppp",
    "m_wiggle_J",
    "m_ess",
    "runtime",
)


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _col(table, name: str) -> np.ndarray | None:
    if name not in table.colnames:
        return None
    try:
        a = np.asarray(table[name], float)
    except (TypeError, ValueError):
        return None
    return a


def _distributions(table) -> list[tuple[str, str]]:
    out = []
    for name in DIST_COLUMNS:
        a = _col(table, name)
        if a is None or not np.isfinite(a).any():
            continue
        a = a[np.isfinite(a)]
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        ax.hist(a, bins=min(40, max(5, a.size // 5)), color=style.SERIES[0], histtype="stepfilled", alpha=0.8)
        ax.set_title(name, loc="left")
        ax.set_ylabel("spectra")
        out.append((name, _png(fig)))
    return out


def _flag_stats(table) -> tuple[dict[str, int], str | None]:
    flags = {c[5:]: int(np.sum(np.asarray(table[c], bool))) for c in table.colnames if c.startswith("flag_")}
    if not flags:
        return flags, None
    names = sorted(flags, key=lambda k: -flags[k])
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(names) + 0.8))
    ax.barh(names[::-1], [flags[n] for n in names[::-1]], color=style.SERIES[0], height=0.7)
    for i, n in enumerate(names[::-1]):
        ax.text(flags[n], i, f" {flags[n]}", va="center", color=style.INK_2, fontsize=8)
    ax.set_xlabel("flagged spectra")
    ax.set_title("flag frequencies", loc="left")
    ax.grid(axis="y", visible=False)
    return flags, _png(fig)


def _classic_comparison(table) -> str | None:
    pairs = [("W50_1", "classic_W50"), ("W20_1", "classic_W20")]
    have = [(b, c) for b, c in pairs if _col(table, b) is not None and _col(table, c) is not None]
    if not have:
        return None
    fig, axes = plt.subplots(1, len(have), figsize=(4 * len(have), 3.4))
    axes = np.atleast_1d(axes)
    det = _col(table, "p_detect")
    for ax, (b, c) in zip(axes, have, strict=True):
        x, yv = _col(table, b), _col(table, c)
        assert x is not None and yv is not None
        ok = np.isfinite(x) & np.isfinite(yv)
        if det is not None:
            ok &= det > 0.5
        ax.scatter(x[ok], yv[ok], s=10, color=style.SERIES[0], alpha=0.6, edgecolor="none")
        if ok.any():
            lim = [0, np.nanpercentile(np.r_[x[ok], yv[ok]], 99) * 1.05]
            ax.plot(lim, lim, color=style.MUTED, lw=0.8)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
        ax.set_xlabel(f"{b} (posterior median) [km/s]")
        ax.set_ylabel(f"{c} [km/s]")
        ax.set_title(f"{b} vs classical (P(detect) > 0.5)", loc="left")
    fig.tight_layout()
    return _png(fig)


def _thumb(out_dir: Path, sid: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_.+" else "_" for ch in sid)
    p = out_dir / "plots" / f"{safe}_spectrum.png"
    if p.exists():
        rel = p.relative_to(out_dir)
        return f'<a href="{html.escape(str(rel))}"><img loading="lazy" src="{html.escape(str(rel))}" alt="spectrum {html.escape(sid)}"></a>'
    return '<span class="muted">no plot (run <code>hiprofile plot</code>)</span>'


CSS = """
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8f8d87;--line:#e4e3df;--accent:#2a78d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8d87;--line:#3a3936;--accent:#3987e5}
 img{filter:none;background:#fcfcfb}}
body{background:var(--surface);color:var(--ink);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px;max-width:1200px;margin:auto}
h1{font-size:22px;margin:8px 0} h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.tiles{display:flex;flex-wrap:wrap;gap:12px}.tile{border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:120px}
.tile b{display:block;font-size:22px}.tile span{color:var(--ink2);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:8px}.grid img{width:100%}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
td img{width:320px;max-width:60vw}.muted{color:var(--muted)} input{font:inherit;padding:6px 8px;width:100%;max-width:420px;margin:6px 0;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--ink)}
.scroll{overflow-x:auto} code{font-size:12px}
"""

JS = """
const q=document.getElementById('q');q&&q.addEventListener('input',()=>{const t=q.value.toLowerCase();
document.querySelectorAll('#flagged tbody tr').forEach(r=>{r.style.display=r.textContent.toLowerCase().includes(t)?'':'none'})});
"""


def build_report(out_dir: str | Path, table, max_flagged: int = 500) -> Path:
    out_dir = Path(out_dir)
    n = len(table)
    status = np.asarray(table["status"]).astype(str) if "status" in table.colnames else np.array(["ok"] * n)
    flagged = np.asarray(table["flagged"], bool) if "flagged" in table.colnames else np.zeros(n, bool)
    det = _col(table, "p_detect")
    n_det = int(np.sum(det > 0.5)) if det is not None else 0
    tiles = [
        ("spectra", n),
        ("fitted", int(np.sum(status == "ok"))),
        ("failed", int(np.sum(status != "ok"))),
        ("P(detect) > 0.5", n_det),
        ("flagged", int(flagged.sum())),
    ]
    flags, flag_png = _flag_stats(table)
    parts: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>hiprofile report</title><style>{CSS}</style></head><body>",
        f"<h1>hiprofile report</h1><p class='muted'>{html.escape(str(out_dir))} · hiprofile {__version__}</p>",
        "<div class='tiles'>",
        *[f"<div class='tile'><b>{v}</b><span>{html.escape(k)}</span></div>" for k, v in tiles],
        "</div>",
    ]
    parts.append("<h2>Distributions of catalogue columns</h2><div class='grid'>")
    for name, png in _distributions(table):
        parts.append(f"<img alt='histogram of {html.escape(name)}' src='data:image/png;base64,{png}'>")
    parts.append("</div>")
    if flag_png:
        parts.append("<h2>Flags</h2>")
        parts.append(f"<img alt='flag frequencies' src='data:image/png;base64,{flag_png}'>")
        parts.append(
            "<div class='scroll'><table><thead><tr><th>flag</th><th>spectra</th><th>fraction</th></tr></thead><tbody>"
        )
        for k, v in sorted(flags.items(), key=lambda kv: -kv[1]):
            parts.append(f"<tr><td>{html.escape(k)}</td><td>{v}</td><td>{v / max(n, 1):.1%}</td></tr>")
        parts.append("</tbody></table></div>")
    cmp_png = _classic_comparison(table)
    if cmp_png:
        parts.append("<h2>Comparison with classical W50 / W20</h2>")
        parts.append(
            f"<img alt='posterior widths against classical widths' src='data:image/png;base64,{cmp_png}'>"
        )
    parts.append(
        "<h2>Flagged spectra</h2><input id='q' placeholder='Filter by id or flag…' aria-label='filter flagged spectra'>"
    )
    parts.append(
        "<div class='scroll'><table id='flagged'><thead><tr><th>id</th><th>P(detect)</th><th>N, K</th><th>flags</th><th>preview</th></tr></thead><tbody>"
    )
    flag_cols = [c for c in table.colnames if c.startswith("flag_")]
    idx = np.flatnonzero(flagged)[:max_flagged]
    for i in idx:
        row = table[i]
        sid = str(row["id"])
        fl = ", ".join(c[5:] for c in flag_cols if bool(row[c]))
        pd = f"{float(row['p_detect']):.3f}" if "p_detect" in table.colnames else ""
        nk = f"{row['best_n']}, {row['best_k']}" if "best_n" in table.colnames else ""
        parts.append(
            f"<tr><td>{html.escape(sid)}</td><td>{pd}</td><td>{nk}</td><td>{html.escape(fl)}</td><td>{_thumb(out_dir, sid)}</td></tr>"
        )
    parts.append("</tbody></table></div>")
    if flagged.sum() > max_flagged:
        parts.append(f"<p class='muted'>showing {max_flagged} of {int(flagged.sum())} flagged spectra</p>")
    parts.append(f"<script>{JS}</script></body></html>")
    target = out_dir / "report.html"
    target.write_text("\n".join(parts))
    return target


def report_from_dir(out_dir: str | Path, fmt: str = "fits") -> dict[str, Any]:
    from hiprofile.io.catalog import build_catalog, read_catalog

    cat = build_catalog(out_dir, fmt)
    table = read_catalog(cat)
    rep = build_report(out_dir, table)
    return {"catalog": str(cat), "report": str(rep), "n": len(table)}
