"""Figures for the README and the documentation.

    python docs/make_figures.py priors            # prior densities vs. THINGS fits
    python docs/make_figures.py gallery           # fits of simulated spectra
    python docs/make_figures.py validation DIR    # SBC / detection / classical comparison from a
                                                  # `hiprofile validate --out DIR` run

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
    u = rng.uniform(size=(400000, 4))
    th = pr(u)
    ratio = Priors().disk_scale_ratio
    ar, ad = np.abs(rng.normal(0.0, [1.0, ratio], size=(400000, 2))).T
    draws = {"delta_v": th[:, 1], "delta_v_rand": th[:, 2], "alpha": th[:, 3], "f": ad / (ar + ad)}
    panels = [
        (
            "delta_v",
            "Δv [km/s]",
            np.linspace(0, 800, 33),
            "log-normal 160 km/s, 0.35 dex\n+ 10 % uniform < 60 km/s",
        ),
        ("delta_v_rand", "Δv_rand [km/s]", np.linspace(4, 40, 37), "log-normal 12.6 km/s, 0.2 dex"),
        ("alpha", "α", np.linspace(-1, 1, 41), "normal(0, 0.25) + 5 % uniform on [−1, 1]"),
        ("f", "f (solid-body fraction)", np.linspace(0, 1, 21), "a_disk scale = a_ring / 3"),
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
        "Default priors vs. the THINGS fits of Stewart et al. (2014)", x=0.01, ha="left", fontsize=11, y=1.12
    )
    return _save(fig, "priors_things.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "priors"
    if what == "priors":
        priors()
    elif what == "gallery":
        from make_figures_gallery import gallery  # type: ignore[import-not-found]

        gallery()
    else:
        raise SystemExit(__doc__)
