"""Priors over the nonlinear parameters as transforms of the unit cube.

Default shapes (see docs/priors.md): log-normal widths and dispersions and a
truncated normal asymmetry, calibrated on the THINGS fits of Stewart et al.
(2014). Every prior is a transform of the unit cube, so the same transform
draws the simulated galaxies and drives the samplers.

Physical parameters per component:

* ``profile``: (v_ctr, delta_v, delta_v_rand, alpha, f)
* ``gauss`` (RFI): (v_ctr, sigma)

plus an optional trailing log10 noise scale.

Sampling parametrisations of the profile widths:

``direct``
    delta_v ~ uniform/log-uniform, log delta_v_rand ~ uniform.
``effective``
    W = delta_v + 2.35 delta_v_rand ~ uniform and phi = delta_v / W ~ uniform,
    with delta_v_rand = sqrt(((1 - phi) W / 2.35)^2 + sigma_min^2). This puts the
    degeneracy ridge between delta_v and delta_v_rand roughly along a cube axis.

With several profile components, the centres are ordered,
v_ctr_1 < v_ctr_2 < ..., by mapping the cube to the order statistics of the
uniform prior; the prior on the ordered set is N! / V^N, which equals the
exchangeable prior with the labels removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import special

from hiprofile.config import Priors, Range

FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
NPROFILE = 5  # v_ctr, delta_v, delta_v_rand, alpha, f


@dataclass(frozen=True)
class Bounds1D:
    """1D prior as a transform of u in [0, 1].

    ``uniform`` and ``loguniform`` on [low, high]; ``normal`` (mean, std) and
    ``lognormal`` (mean, std of log10 x), truncated to [low, high] (infinite
    bounds allowed).
    """

    kind: str
    low: float
    high: float
    mean: float = 0.0
    std: float = 1.0

    def _z(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.kind == "lognormal":
            with np.errstate(divide="ignore", invalid="ignore"):
                lg = np.where(x > 0, np.log10(np.where(x > 0, x, 1.0)), -np.inf)
            return (lg - self.mean) / self.std
        return (x - self.mean) / self.std

    def _trunc(self) -> tuple[float, float]:
        lo = float(special.ndtr(self._z(np.asarray(self.low)))) if np.isfinite(self.low) else 0.0
        hi = float(special.ndtr(self._z(np.asarray(self.high)))) if np.isfinite(self.high) else 1.0
        return lo, hi

    def transform(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if self.kind == "uniform":
            return self.low + (self.high - self.low) * u
        if self.kind == "loguniform":
            return np.exp(np.log(self.low) + (np.log(self.high) - np.log(self.low)) * u)
        c_lo, c_hi = self._trunc()
        z = special.ndtri(np.clip(c_lo + (c_hi - c_lo) * u, 1e-300, 1 - 1e-16))
        x = 10.0 ** (self.mean + self.std * z) if self.kind == "lognormal" else self.mean + self.std * z
        return np.clip(x, self.low, self.high)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.kind == "uniform":
            return np.clip((x - self.low) / (self.high - self.low), 0.0, 1.0)
        if self.kind == "loguniform":
            return np.clip((np.log(x) - np.log(self.low)) / (np.log(self.high) - np.log(self.low)), 0.0, 1.0)
        c_lo, c_hi = self._trunc()
        return np.clip((special.ndtr(self._z(x)) - c_lo) / (c_hi - c_lo), 0.0, 1.0)

    @property
    def edges(self) -> tuple[float, float]:
        return (self.low, self.high)


def _bounds(r: Range, default_low: float, default_high: float) -> Bounds1D:
    if r.kind == "normal":
        low = -np.inf if r.low is None else r.low
        high = np.inf if r.high is None else r.high
    elif r.kind == "lognormal":
        low = 0.0 if r.low is None else r.low
        high = np.inf if r.high is None else r.high
    else:
        low = default_low if r.low is None else r.low
        high = default_high if r.high is None else r.high
    return Bounds1D(
        kind=r.kind,
        low=float(low),
        high=float(high),
        mean=0.0 if r.mean is None else float(r.mean),
        std=1.0 if r.std is None else float(r.std),
    )


def ordered_uniform(u: np.ndarray) -> np.ndarray:
    """Map u in [0,1]^N to sorted uniforms x_1 < ... < x_N (order statistics)."""
    n = u.shape[-1]
    x = np.empty_like(u)
    prev = np.ones(u.shape[:-1])
    for i in range(n - 1, -1, -1):
        prev = prev * u[..., i] ** (1.0 / (i + 1))
        x[..., i] = prev
    return x


def ordered_uniform_inverse(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1]
    u = np.empty_like(x)
    upper = np.ones(x.shape[:-1])
    for i in range(n - 1, -1, -1):
        ratio = np.clip(x[..., i] / upper, 0.0, 1.0)
        u[..., i] = ratio ** (i + 1)
        upper = x[..., i]
    return u


@dataclass
class ModelPrior:
    """Unit-cube transform for one model (list of components)."""

    components: tuple[str, ...]
    v_ctr: Bounds1D
    delta_v: Bounds1D
    delta_v_rand: Bounds1D
    alpha: Bounds1D
    f: Bounds1D
    rfi_v: Bounds1D
    rfi_sigma: Bounds1D
    parametrization: str = "direct"
    noise_scale: Bounds1D | None = None
    names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.parametrization not in ("direct", "effective"):
            raise ValueError(self.parametrization)
        n_prof = self.components.count("profile")
        if n_prof > 1 and self.v_ctr.kind != "uniform":
            raise ValueError("several profile components need a uniform v_ctr prior")
        names = []
        ip = ig = 0
        for c in self.components:
            if c == "profile":
                ip += 1
                names += [f"v_ctr_{ip}", f"delta_v_{ip}", f"delta_v_rand_{ip}", f"alpha_{ip}", f"f_{ip}"]
            else:
                ig += 1
                names += [f"rfi_v_{ig}", f"rfi_sigma_{ig}"]
        if self.noise_scale is not None:
            names.append("log10_noise_scale")
        self.names = names

    @property
    def ndim(self) -> int:
        return len(self.names)

    def _profile_cols(self) -> list[int]:
        cols, i = [], 0
        for c in self.components:
            if c == "profile":
                cols.append(i)
            i += NPROFILE if c == "profile" else 2
        return cols

    def _gauss_cols(self) -> list[int]:
        cols, i = [], 0
        for c in self.components:
            if c == "gauss":
                cols.append(i)
            i += NPROFILE if c == "profile" else 2
        return cols

    # ---------------------------------------------------------------- widths
    def _widths(self, u1: np.ndarray, u2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.parametrization == "direct":
            return self.delta_v.transform(u1), self.delta_v_rand.transform(u2)
        smin = self.delta_v_rand.low
        w_lo = self.delta_v.low + FWHM * smin
        w_hi = self.delta_v.high + FWHM * self.delta_v_rand.high
        w = w_lo + (w_hi - w_lo) * u1
        dv = u2 * w
        sig = np.sqrt(((1.0 - u2) * w / FWHM) ** 2 + smin**2)
        return dv, sig

    def _widths_inverse(self, dv: np.ndarray, sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.parametrization == "direct":
            return self.delta_v.inverse(dv), self.delta_v_rand.inverse(sig)
        smin = self.delta_v_rand.low
        rest = np.sqrt(np.maximum(sig**2 - smin**2, 0.0)) * FWHM
        w = dv + rest
        w_lo = self.delta_v.low + FWHM * smin
        w_hi = self.delta_v.high + FWHM * self.delta_v_rand.high
        return (w - w_lo) / (w_hi - w_lo), np.where(w > 0, dv / np.maximum(w, 1e-300), 0.5)

    # ------------------------------------------------------------- transform
    def transform(self, u: np.ndarray) -> np.ndarray:
        u = np.atleast_2d(np.asarray(u, dtype=float))
        theta = np.empty_like(u)
        pc = self._profile_cols()
        if pc:
            uc = u[:, pc]
            if len(pc) > 1:
                uc = ordered_uniform(uc)
            theta[:, pc] = self.v_ctr.transform(uc)
            for c in pc:
                theta[:, c + 1], theta[:, c + 2] = self._widths(u[:, c + 1], u[:, c + 2])
                theta[:, c + 3] = self.alpha.transform(u[:, c + 3])
                theta[:, c + 4] = self.f.transform(u[:, c + 4])
        gc = self._gauss_cols()
        if gc:
            ug = u[:, gc]
            if len(gc) > 1:
                ug = ordered_uniform(ug)
            theta[:, gc] = self.rfi_v.transform(ug)
            for c in gc:
                theta[:, c + 1] = self.rfi_sigma.transform(u[:, c + 1])
        if self.noise_scale is not None:
            theta[:, -1] = np.log10(self.noise_scale.transform(u[:, -1]))
        return theta

    def inverse(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        u = np.empty_like(theta)
        pc = self._profile_cols()
        if pc:
            uc = self.v_ctr.inverse(theta[:, pc])
            if len(pc) > 1:
                uc = ordered_uniform_inverse(uc)
            u[:, pc] = uc
            for c in pc:
                u[:, c + 1], u[:, c + 2] = self._widths_inverse(theta[:, c + 1], theta[:, c + 2])
                u[:, c + 3] = self.alpha.inverse(theta[:, c + 3])
                u[:, c + 4] = self.f.inverse(theta[:, c + 4])
        gc = self._gauss_cols()
        if gc:
            ug = self.rfi_v.inverse(theta[:, gc])
            if len(gc) > 1:
                ug = ordered_uniform_inverse(ug)
            u[:, gc] = ug
            for c in gc:
                u[:, c + 1] = self.rfi_sigma.inverse(theta[:, c + 1])
        if self.noise_scale is not None:
            u[:, -1] = self.noise_scale.inverse(10.0 ** theta[:, -1])
        return u

    def edge_bounds(self) -> dict[str, tuple[float, float]]:
        """Hard prior edges of each named parameter (for the prior-edge diagnostic)."""
        out: dict[str, tuple[float, float]] = {}
        for name in self.names:
            if name.startswith("delta_v_rand"):
                out[name] = self.delta_v_rand.edges
            elif name.startswith("delta_v"):
                out[name] = self.delta_v.edges
            elif name.startswith("alpha"):
                out[name] = self.alpha.edges
            elif name.startswith("f_"):
                out[name] = self.f.edges
        return out

    def __call__(self, u: np.ndarray) -> np.ndarray:
        return self.transform(u)


def build_prior(
    components: tuple[str, ...],
    priors: Priors,
    velocity: np.ndarray,
    parametrization: str = "direct",
) -> ModelPrior:
    v_lo, v_hi = float(velocity[0]), float(velocity[-1])
    span = v_hi - v_lo
    dv = float(velocity[1] - velocity[0])
    vc = _bounds(priors.v_ctr, v_lo + priors.v_margin * span, v_hi - priors.v_margin * span)
    lo_c, hi_c = priors.rfi_sigma_channels
    noise = None
    if priors.noise_scale is not None:
        noise = Bounds1D("loguniform", *priors.noise_scale)
    return ModelPrior(
        components=components,
        v_ctr=vc,
        delta_v=_bounds(priors.delta_v, 0.0, 600.0),
        delta_v_rand=_bounds(priors.delta_v_rand, 4.0, 60.0),
        alpha=_bounds(priors.alpha, -1.0, 1.0),
        f=_bounds(priors.f, 0.0, 1.0),
        rfi_v=Bounds1D("uniform", v_lo, v_hi),
        rfi_sigma=Bounds1D("loguniform", lo_c * dv, hi_c * dv),
        parametrization=parametrization,
        noise_scale=noise,
    )
