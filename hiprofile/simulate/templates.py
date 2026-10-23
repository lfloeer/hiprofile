"""Template profiles that the fit model does not describe exactly.

``tilted_ring_profile`` integrates a thin rotating disk numerically: rotation
curve v(r) = v_flat (1 - exp(-r / r_t)), exponential (optionally ring-like)
surface density with an m = 1 lopsidedness, inclination i and isotropic
dispersion. Profiles from files (e.g. THINGS global profiles) can be used as
templates with ``FileTemplate``; they are shifted and stretched in velocity and
scaled in flux.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hiprofile.inference.derived import crossing_widths


@dataclass
class TemplateProfile:
    velocity: np.ndarray  # fine grid [km/s], relative to the centre
    density: np.ndarray  # unit integral over velocity

    def on_grid(self, v: np.ndarray, v_ctr: float, flux: float) -> np.ndarray:
        """Template evaluated (interpolated) on the fine grid ``v``."""
        return flux * np.interp(v - v_ctr, self.velocity, self.density, left=0.0, right=0.0)

    def widths(self) -> tuple[float, float, float]:
        """(W50, W20, midpoint of the W50 crossings relative to the centre)."""
        lo50, hi50 = crossing_widths(self.velocity, self.density[None], 0.5)
        lo20, hi20 = crossing_widths(self.velocity, self.density[None], 0.2)
        return float(hi50[0] - lo50[0]), float(hi20[0] - lo20[0]), float(0.5 * (lo50[0] + hi50[0]))


def tilted_ring_profile(
    rng: np.random.Generator,
    v_flat: float | None = None,
    inclination_deg: float | None = None,
    sigma: float | None = None,
    lopsided: float | None = None,
    hole: float | None = None,
    n_r: int = 400,
    n_phi: int = 720,
    h: float = 0.5,
) -> tuple[TemplateProfile, dict[str, float]]:
    v_flat = rng.uniform(40, 300) if v_flat is None else v_flat
    inc = np.deg2rad(rng.uniform(30, 85) if inclination_deg is None else inclination_deg)
    sigma = rng.uniform(6, 15) if sigma is None else sigma
    lop = rng.uniform(0, 0.3) if lopsided is None else lopsided
    hole = rng.uniform(0, 1.5) if hole is None else hole
    r_t = rng.uniform(0.3, 2.0)
    r = np.linspace(0.01, 6.0, n_r)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    rr, pp = np.meshgrid(r, phi, indexing="ij")
    surf = np.exp(-rr) * (1 - np.exp(-((rr / max(hole, 1e-3)) ** 2)) if hole > 0 else 1.0)
    surf = surf * (1 + lop * np.cos(pp)) * rr  # area element r dr dphi
    vrot = v_flat * (1 - np.exp(-rr / r_t))
    vlos = vrot * np.sin(inc) * np.cos(pp)
    vmax = v_flat * np.sin(inc) + 6 * sigma
    edges = np.arange(-vmax - h / 2, vmax + h, h)
    hist, _ = np.histogram(vlos, bins=edges, weights=surf)
    v = 0.5 * (edges[1:] + edges[:-1])
    kern_v = np.arange(-6 * sigma, 6 * sigma + h, h)
    kern = np.exp(-0.5 * (kern_v / sigma) ** 2)
    dens = np.convolve(hist, kern / kern.sum(), mode="same")
    dens /= dens.sum() * h
    info = {
        "v_flat": v_flat,
        "inclination": float(np.rad2deg(inc)),
        "sigma": sigma,
        "lopsided": lop,
        "hole": hole,
    }
    return TemplateProfile(v, dens), info


@dataclass
class FileTemplate:
    path: str

    def load(self, rng: np.random.Generator, width_scale: float = 1.0) -> TemplateProfile:
        arr = np.loadtxt(Path(self.path), ndmin=2)
        v, y = arr[:, 0], np.clip(arr[:, 1], 0, None)
        centre = np.sum(v * y) / np.sum(y)
        vv = (v - centre) * width_scale
        order = np.argsort(vv)
        vv, y = vv[order], y[order]
        y = y / np.trapezoid(y, vv)
        return TemplateProfile(vv, y)
