"""Simulated spectra with known truth.

Components come from the fit model itself (parameters drawn from the prior or
from survey-like ranges) or from templates. Baselines are Chebyshev polynomials
or sinusoidal standing waves; RFI is added as narrow spikes or broad bumps;
noise is white or Hanning-correlated. All profiles go through the same channel
response as the fit model (computed exactly in Fourier space for model
profiles, and by integrating a fine grid for templates).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from hiprofile.config import Priors
from hiprofile.inference.derived import fine_grid, profile_derived
from hiprofile.inference.priors import build_prior
from hiprofile.model.kernel import SpectralGrid, chebyshev_basis, profile_basis
from hiprofile.model.response import transfer_function
from hiprofile.simulate.templates import FileTemplate, tilted_ring_profile
from hiprofile.spectrum import Spectrum


class SimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_chan: int = 512
    dv: float = 5.0
    v0: float = 500.0
    response: str = "boxcar"
    supersample: int = 8
    # number of components: fixed value, or probabilities for N = 0, 1, 2, ...
    n_components: int | list[float] = 1
    profiles: Literal["model", "template", "mixed"] = "model"
    template_files: list[str] = Field(default_factory=list)
    priors: Priors = Field(default_factory=lambda: Priors())
    # flux: "prior" draws a_ring, a_disk ~ half-normal(amplitude_scale) (SBC);
    # "snr" draws the integrated S/N log-uniformly in snr_range
    flux: Literal["prior", "snr"] = "snr"
    snr_range: tuple[float, float] = (3.0, 100.0)
    f_range: tuple[float, float] = (0.0, 1.0)
    noise_sigma: float = 0.01
    noise_correlation: Literal["white", "hanning"] = "white"
    baseline: Literal["none", "chebyshev", "sine"] = "none"
    baseline_order: int = 1
    baseline_scale: float = 0.01
    sine_amplitude: float = 0.5  # in units of noise sigma
    sine_period: tuple[float, float] = (100.0, 600.0)  # km/s
    rfi: Literal["none", "spikes", "broad"] = "none"
    rfi_rate: float = 0.01  # spikes per channel
    rfi_amplitude: tuple[float, float] = (5.0, 50.0)  # in units of noise sigma
    write_sigma: bool = True


@dataclass
class Truth:
    id: str
    n: int
    components: list[dict[str, float]] = field(default_factory=list)
    baseline: dict[str, Any] = field(default_factory=dict)
    rfi_channels: list[int] = field(default_factory=list)
    kind: str = "model"

    def row(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "n_true": self.n,
            "kind": self.kind,
            "n_rfi": len(self.rfi_channels),
        }
        for i, c in enumerate(self.components, 1):
            for k, v in c.items():
                out[f"{k}_{i}"] = v
        for k, v in self.baseline.items():
            if np.isscalar(v):
                out[f"baseline_{k}"] = v
        return out


def _draw_n(cfg: SimConfig, rng: np.random.Generator) -> int:
    if isinstance(cfg.n_components, int):
        return cfg.n_components
    p = np.asarray(cfg.n_components, float)
    return int(rng.choice(p.size, p=p / p.sum()))


def _template_channels(grid: SpectralGrid, prof, v_ctr: float, flux: float) -> np.ndarray:
    """Template through the channel response: fine grid -> FFT -> response -> channels."""
    fg = fine_grid(grid, max_spacing=0.25)
    sub = round(grid.dv / fg.dv)
    y = prof.on_grid(fg.velocity, v_ctr, flux)
    n = y.size * 2
    spec = np.fft.rfft(y, n=n)
    omega = 2 * np.pi * np.fft.rfftfreq(n, d=fg.dv)
    spec *= transfer_function(grid.response, omega, grid.dv)
    yy = np.fft.irfft(spec, n=n)[: y.size]
    return yy[sub // 2 :: sub][: grid.n_chan]


def simulate_spectrum(cfg: SimConfig, rng: np.random.Generator, sid: str) -> tuple[Spectrum, Truth]:
    v = cfg.v0 + cfg.dv * np.arange(cfg.n_chan)
    grid = SpectralGrid.from_velocities(v, supersample=cfg.supersample, response=cfg.response)
    n = _draw_n(cfg, rng)
    truth = Truth(id=sid, n=n)
    signal = np.zeros(cfg.n_chan)
    band_noise = cfg.noise_sigma * cfg.dv * np.sqrt(cfg.n_chan)
    kinds = []
    if n:
        prior = build_prior(("profile",) * n, cfg.priors, v)
        theta = prior.transform(rng.uniform(size=(1, prior.ndim)))[0].reshape(n, 4)
        for j in range(n):
            kind = cfg.profiles
            if kind == "mixed":
                kind = "template" if rng.uniform() < 0.5 else "model"
            kinds.append(kind)
            vc, width, sig, alpha = theta[j]
            if cfg.flux == "prior":
                amp_scale = cfg.priors.amplitude_scale
                if amp_scale == "auto":
                    raise ValueError("flux = prior needs a fixed priors.amplitude_scale")
                a = np.abs(rng.normal(0, float(amp_scale), 2))
                s_tot, f = float(a.sum()), float(a[1] / a.sum())
            else:
                snr = np.exp(rng.uniform(*np.log(cfg.snr_range)))
                # integrated S/N over the line: S / (sigma dv sqrt(W / dv))
                w_eff = max(width + 2.35 * sig, cfg.dv)
                s_tot = float(snr * cfg.noise_sigma * cfg.dv * np.sqrt(w_eff / cfg.dv))
                f = float(rng.uniform(*cfg.f_range))
                a = np.array([s_tot * (1 - f), s_tot * f])
            if kind == "model":
                b = profile_basis(grid, theta[j][None])[0]
                signal += a[0] * b[0] + a[1] * b[1]
                d = profile_derived(grid, theta[j][None], a[None])
                comp = {
                    "v_ctr": vc,
                    "delta_v": width,
                    "delta_v_rand": sig,
                    "alpha": alpha,
                    "S": s_tot,
                    "f": f,
                }
                comp.update({k: float(d[k][0]) for k in ("W50", "W20", "v_sys", "peak")})
            else:
                if cfg.template_files:
                    ft = FileTemplate(str(rng.choice(cfg.template_files)))
                    prof = ft.load(rng, width_scale=rng.uniform(0.7, 1.3))
                    info: dict[str, float] = {}
                else:
                    prof, info = tilted_ring_profile(rng, v_flat=0.5 * width + 20, sigma=sig)
                signal += _template_channels(grid, prof, vc, s_tot)
                w50, w20, mid = prof.widths()
                comp = {
                    "v_ctr": vc,
                    "S": s_tot,
                    "W50": w50,
                    "W20": w20,
                    "v_sys": vc + mid,
                    "peak": float(s_tot * prof.density.max()),
                }
                comp.update({f"tpl_{k}": val for k, val in info.items()})
            comp["snr"] = (
                s_tot / band_noise * np.sqrt(cfg.n_chan * cfg.dv / max(comp.get("W20", cfg.dv), cfg.dv))
            )
            truth.components.append(comp)
        truth.kind = "+".join(kinds)
    # baseline
    base = np.zeros(cfg.n_chan)
    if cfg.baseline == "chebyshev":
        c = rng.normal(0, cfg.baseline_scale, cfg.baseline_order + 1)
        base = c @ chebyshev_basis(cfg.n_chan, cfg.baseline_order)
        truth.baseline = {"kind": "chebyshev", "order": cfg.baseline_order, "coef": c.tolist()}
    elif cfg.baseline == "sine":
        period = rng.uniform(*cfg.sine_period)
        phase = rng.uniform(0, 2 * np.pi)
        amp = cfg.sine_amplitude * cfg.noise_sigma
        base = amp * np.sin(2 * np.pi * v / period + phase)
        truth.baseline = {"kind": "sine", "period": period, "amplitude": amp, "phase": phase}
    # noise
    eps = rng.standard_normal(cfg.n_chan + 2)
    if cfg.noise_correlation == "hanning":
        # Hanning-smoothed noise with per-channel std noise_sigma
        raw = np.convolve(eps, [0.25, 0.5, 0.25], mode="valid")
        noise = raw / np.sqrt(0.375) * cfg.noise_sigma
    else:
        noise = eps[: cfg.n_chan] * cfg.noise_sigma
    y = signal + base + noise
    # RFI
    if cfg.rfi == "spikes":
        nr = rng.poisson(cfg.rfi_rate * cfg.n_chan)
        ch = rng.choice(cfg.n_chan, size=nr, replace=False)
        lo_a, hi_a = np.log(cfg.rfi_amplitude)
        y[ch] += np.exp(rng.uniform(lo_a, hi_a, nr)) * cfg.noise_sigma
        truth.rfi_channels = sorted(int(c) for c in ch)
    elif cfg.rfi == "broad":
        nr = max(1, rng.poisson(cfg.rfi_rate * cfg.n_chan / 10))
        for _ in range(nr):
            c0 = int(rng.integers(0, cfg.n_chan))
            wid = int(rng.integers(2, 6))
            amp = np.exp(rng.uniform(*np.log(cfg.rfi_amplitude))) * cfg.noise_sigma
            start, stop = max(0, c0 - wid), min(cfg.n_chan, c0 + wid + 1)
            y[start:stop] += amp * np.hanning(stop - start + 2)[1:-1]
            truth.rfi_channels += list(range(start, stop))
    sigma = np.full(cfg.n_chan, cfg.noise_sigma) if cfg.write_sigma else None
    spec = Spectrum(velocity=v, flux=y, sigma=sigma, id=sid, meta={"n_true": n})
    return spec, truth


def inject(
    spectrum: Spectrum, cfg: SimConfig, rng: np.random.Generator, sid: str | None = None
) -> tuple[Spectrum, Truth]:
    """Add simulated components to a real (signal-free) spectrum."""
    spec = spectrum.normalized()
    sim_cfg = cfg.model_copy(
        update={
            "n_chan": spec.n_chan,
            "dv": spec.channel_width,
            "v0": float(spec.velocity[0]),
            "baseline": "none",
            "rfi": "none",
        }
    )
    clean, truth = simulate_spectrum(sim_cfg, rng, sid or spec.id)
    noise_free = _signal_only(sim_cfg, clean, truth)
    out = Spectrum(
        velocity=spec.velocity,
        flux=spec.flux + noise_free,
        sigma=spec.sigma,
        mask=spec.mask,
        id=sid or f"{spec.id}_inj",
        meta={**spec.meta, "n_true": truth.n, "injected_into": spec.id},
    )
    return out, truth


def _signal_only(cfg: SimConfig, spec: Spectrum, truth: Truth) -> np.ndarray:
    """Noise-free model signal of the simulated components."""
    v = spec.velocity
    grid = SpectralGrid.from_velocities(v, supersample=cfg.supersample, response=cfg.response)
    sig = np.zeros(v.size)
    for c in truth.components:
        if "delta_v" in c:
            th = np.array([[c["v_ctr"], c["delta_v"], c["delta_v_rand"], c["alpha"]]])
            b = profile_basis(grid, th)[0]
            sig += c["S"] * (1 - c["f"]) * b[0] + c["S"] * c["f"] * b[1]
        else:
            raise ValueError("injection supports model profiles only")
    return sig


def simulate_many(cfg: SimConfig, n: int, seed: int, prefix: str = "sim") -> list[tuple[Spectrum, Truth]]:
    ss = np.random.SeedSequence(seed)
    out = []
    for i, child in enumerate(ss.spawn(n)):
        rng = np.random.default_rng(child)
        out.append(simulate_spectrum(cfg, rng, f"{prefix}{i:06d}"))
    return out
