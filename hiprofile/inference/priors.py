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

With several profile components, the centres are ordered,
v_ctr_1 < v_ctr_2 < ..., by mapping the cube to the order statistics of the
uniform prior; the prior on the ordered set is N! / V^N, which equals the
exchangeable prior with the labels removed.

Containment (``priors.contain``, the default for a uniform centre prior): every
profile component lies inside the band minus ``v_margin``, i.e.
v_ctr +- (delta_v / 2 + CONTAIN_SIGMA delta_v_rand) does. The prior is built
sequentially: delta_v_rand and then delta_v keep their shapes and are only
truncated further where no line of that width fits, and the centre is uniform
over the positions where the whole component fits. A fixed window therefore
does not favour narrow lines. With several components the labels are ordered
by this relative position r in [0, 1] instead of by v_ctr.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import special

from hiprofile.config import Priors, Range

NPROFILE = 5  # v_ctr, delta_v, delta_v_rand, alpha, f
CONTAIN_SIGMA = 3.0  # half extent of a component: delta_v / 2 + CONTAIN_SIGMA delta_v_rand


@dataclass(frozen=True)
class Bounds1D:
    """1D prior as a transform of u in [0, 1].

    ``uniform`` and ``loguniform`` on [low, high]; ``normal`` (mean, std) and
    ``lognormal`` (mean, std of log10 x), truncated to [low, high] (infinite
    bounds allowed); ``beta`` (shapes a, b) scaled to [low, high]; ``gamma``
    (shape a, scale) truncated to [low, high].
    """

    kind: str
    low: float
    high: float
    mean: float = 0.0
    std: float = 1.0
    a: float = 1.0
    b: float = 1.0
    scale: float = 1.0

    def _z(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.kind == "lognormal":
            with np.errstate(divide="ignore", invalid="ignore"):
                lg = np.where(x > 0, np.log10(np.where(x > 0, x, 1.0)), -np.inf)
            return (lg - self.mean) / self.std
        return (x - self.mean) / self.std

    def _cdf(self, x: np.ndarray) -> np.ndarray:
        """Untruncated CDF of the normal, log-normal and gamma kinds."""
        if self.kind == "gamma":
            return special.gammainc(self.a, np.maximum(np.asarray(x, float), 0.0) / self.scale)
        return special.ndtr(self._z(x))

    def _trunc(self) -> tuple[float, float]:
        lo = float(self._cdf(np.asarray(self.low))) if np.isfinite(self.low) else 0.0
        hi = float(self._cdf(np.asarray(self.high))) if np.isfinite(self.high) else 1.0
        return lo, hi

    def _cdf_any(self, x: np.ndarray) -> np.ndarray:
        """Untruncated CDF of every kind (uniform, log-uniform and beta on [low, high])."""
        x = np.asarray(x, dtype=float)
        if self.kind == "uniform":
            return np.clip((x - self.low) / (self.high - self.low), 0.0, 1.0)
        if self.kind == "loguniform":
            lx = np.log(np.clip(x, self.low, self.high))
            return (lx - np.log(self.low)) / (np.log(self.high) - np.log(self.low))
        if self.kind == "beta":
            return special.betainc(self.a, self.b, np.clip((x - self.low) / (self.high - self.low), 0.0, 1.0))
        return self._cdf(x)

    def _ppf_any(self, p: np.ndarray) -> np.ndarray:
        if self.kind in ("uniform", "loguniform", "beta"):
            return self.transform(p)
        if self.kind == "gamma":
            return self.scale * special.gammaincinv(self.a, np.clip(p, 0.0, 1.0))
        z = special.ndtri(np.clip(p, 1e-300, 1 - 1e-16))
        return 10.0 ** (self.mean + self.std * z) if self.kind == "lognormal" else self.mean + self.std * z

    def _upper(self, high: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        hi = np.clip(np.asarray(high, dtype=float), self.low, self.high)
        c_lo = 0.0 if self.kind in ("uniform", "loguniform", "beta") else self._trunc()[0]
        return hi, c_lo, self._cdf_any(hi)

    def transform_below(self, u: np.ndarray, high: np.ndarray) -> np.ndarray:
        """Transform with the prior truncated further at a per-sample upper bound ``high``."""
        hi, c_lo, c_hi = self._upper(high)
        x = self._ppf_any(c_lo + (c_hi - c_lo) * np.asarray(u, dtype=float))
        return np.clip(x, self.low, hi)

    def inverse_below(self, x: np.ndarray, high: np.ndarray) -> np.ndarray:
        _, c_lo, c_hi = self._upper(high)
        span = c_hi - c_lo
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (self._cdf_any(x) - c_lo) / span
        return np.clip(np.where(span > 0, u, 0.5), 0.0, 1.0)

    def transform(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if self.kind == "uniform":
            return self.low + (self.high - self.low) * u
        if self.kind == "loguniform":
            return np.exp(np.log(self.low) + (np.log(self.high) - np.log(self.low)) * u)
        if self.kind == "beta":
            return self.low + (self.high - self.low) * special.betaincinv(self.a, self.b, u)
        c_lo, c_hi = self._trunc()
        if self.kind == "gamma":
            p = np.clip(c_lo + (c_hi - c_lo) * u, 0.0, 1.0)
            return np.clip(self.scale * special.gammaincinv(self.a, p), self.low, self.high)
        z = special.ndtri(np.clip(c_lo + (c_hi - c_lo) * u, 1e-300, 1 - 1e-16))
        x = 10.0 ** (self.mean + self.std * z) if self.kind == "lognormal" else self.mean + self.std * z
        return np.clip(x, self.low, self.high)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.kind == "uniform":
            return np.clip((x - self.low) / (self.high - self.low), 0.0, 1.0)
        if self.kind == "loguniform":
            return np.clip((np.log(x) - np.log(self.low)) / (np.log(self.high) - np.log(self.low)), 0.0, 1.0)
        if self.kind == "beta":
            t = np.clip((x - self.low) / (self.high - self.low), 0.0, 1.0)
            return special.betainc(self.a, self.b, t)
        c_lo, c_hi = self._trunc()
        return np.clip((self._cdf(x) - c_lo) / (c_hi - c_lo), 0.0, 1.0)

    @property
    def edges(self) -> tuple[float, float]:
        return (self.low, self.high)


def _bounds(r: Range, default_low: float, default_high: float) -> Bounds1D:
    if r.kind == "normal":
        low = -np.inf if r.low is None else r.low
        high = np.inf if r.high is None else r.high
    elif r.kind in ("lognormal", "gamma"):
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
        a=1.0 if r.a is None else float(r.a),
        b=1.0 if r.b is None else float(r.b),
        scale=1.0 if r.scale is None else float(r.scale),
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
    noise_scale: Bounds1D | None = None
    band: tuple[float, float] | None = None  # every profile component lies inside (containment)
    names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        n_prof = self.components.count("profile")
        if (n_prof > 1 or self.band is not None) and self.v_ctr.kind != "uniform":
            raise ValueError("several profile components and containment need a uniform v_ctr prior")
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

    # ----------------------------------------------------------- containment
    def _half_max(self) -> float:
        """Largest half extent of a component that still fits the band and the centre range."""
        b_lo, b_hi = self.band
        return min(0.5 * (b_hi - b_lo), self.v_ctr.high - b_lo, b_hi - self.v_ctr.low)

    def _centre_range(self, delta_v: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = 0.5 * delta_v + CONTAIN_SIGMA * sigma
        lo = np.maximum(self.v_ctr.low, self.band[0] + h)
        hi = np.maximum(np.minimum(self.v_ctr.high, self.band[1] - h), lo)
        return lo, hi

    def band_position(self, theta: np.ndarray) -> np.ndarray:
        """Relative position r in [0, 1] of each profile centre within its allowed range, (n, N)."""
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        pc = self._profile_cols()
        if self.band is None:
            return self.v_ctr.inverse(theta[:, pc])
        r = np.empty((theta.shape[0], len(pc)))
        for j, c in enumerate(pc):
            lo, hi = self._centre_range(theta[:, c + 1], theta[:, c + 2])
            with np.errstate(divide="ignore", invalid="ignore"):
                r[:, j] = np.where(hi > lo, (theta[:, c] - lo) / (hi - lo), 0.5)
        return np.clip(r, 0.0, 1.0)

    # ------------------------------------------------------------- transform
    def transform(self, u: np.ndarray) -> np.ndarray:
        u = np.atleast_2d(np.asarray(u, dtype=float))
        theta = np.empty_like(u)
        pc = self._profile_cols()
        if pc:
            uc = u[:, pc]
            if len(pc) > 1:
                uc = ordered_uniform(uc)
            if self.band is None:
                theta[:, pc] = self.v_ctr.transform(uc)
            for j, c in enumerate(pc):
                if self.band is None:
                    theta[:, c + 1] = self.delta_v.transform(u[:, c + 1])
                    theta[:, c + 2] = self.delta_v_rand.transform(u[:, c + 2])
                else:
                    h_max = self._half_max()
                    sig = self.delta_v_rand.transform_below(u[:, c + 2], h_max / CONTAIN_SIGMA)
                    width = self.delta_v.transform_below(u[:, c + 1], 2.0 * (h_max - CONTAIN_SIGMA * sig))
                    lo, hi = self._centre_range(width, sig)
                    theta[:, c] = lo + (hi - lo) * uc[:, j]
                    theta[:, c + 1], theta[:, c + 2] = width, sig
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
            if self.band is not None and len(pc) > 1:  # labels in the order of the band positions
                idx = np.array([np.arange(c, c + NPROFILE) for c in pc])
                order = np.argsort(self.band_position(theta), axis=1, kind="stable")
                theta = theta.copy()
                theta[:, idx.ravel()] = np.take_along_axis(theta, idx[order].reshape(len(theta), -1), axis=1)
            uc = self.band_position(theta)
            if len(pc) > 1:
                uc = ordered_uniform_inverse(uc)
            u[:, pc] = uc
            for c in pc:
                if self.band is None:
                    u[:, c + 1] = self.delta_v.inverse(theta[:, c + 1])
                    u[:, c + 2] = self.delta_v_rand.inverse(theta[:, c + 2])
                else:
                    h_max = self._half_max()
                    sig = theta[:, c + 2]
                    u[:, c + 2] = self.delta_v_rand.inverse_below(sig, h_max / CONTAIN_SIGMA)
                    w_max = 2.0 * (h_max - CONTAIN_SIGMA * sig)
                    u[:, c + 1] = self.delta_v.inverse_below(theta[:, c + 1], w_max)
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
        """Hard prior edges of each named parameter (for the prior-edge diagnostic).

        Only truncations the truth could lie beyond count. f is left out: [0, 1]
        is its definition, and f near 0 (no solid-body part) is a valid result.
        """
        out: dict[str, tuple[float, float]] = {}
        for name in self.names:
            if name.startswith("delta_v_rand"):
                out[name] = self.delta_v_rand.edges
            elif name.startswith("delta_v"):
                out[name] = self.delta_v.edges
            elif name.startswith("alpha"):
                out[name] = self.alpha.edges
        return out

    def __call__(self, u: np.ndarray) -> np.ndarray:
        return self.transform(u)


def build_prior(
    components: tuple[str, ...],
    priors: Priors,
    velocity: np.ndarray,
) -> ModelPrior:
    v_lo, v_hi = float(velocity[0]), float(velocity[-1])
    span = v_hi - v_lo
    dv = float(velocity[1] - velocity[0])
    vc = _bounds(priors.v_ctr, v_lo + priors.v_margin * span, v_hi - priors.v_margin * span)
    contain = priors.contain and vc.kind == "uniform" and "profile" in components
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
        noise_scale=noise,
        band=(v_lo + priors.v_margin * span, v_hi - priors.v_margin * span) if contain else None,
    )
