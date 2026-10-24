"""Run configuration (YAML), validated with pydantic.

The full configuration is written into every result file together with the
code version and the seed, so that each fit can be reproduced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Range(_Base):
    """A one-dimensional prior.

    ``uniform`` and ``loguniform`` on [low, high]; ``normal`` (mean, std) and
    ``lognormal`` (mean and std of log10 x) optionally truncated to [low, high];
    ``beta`` (shape parameters a, b) scaled to [low, high]; ``gamma`` (shape a,
    scale) optionally truncated to [low, high].
    """

    kind: Literal["uniform", "loguniform", "normal", "lognormal", "beta", "gamma"] = "uniform"
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    std: float | None = None
    a: float | None = None
    b: float | None = None
    scale: float | None = None

    @model_validator(mode="after")
    def _check(self) -> Range:
        if self.kind in ("normal", "lognormal"):
            if self.mean is None or self.std is None or self.std <= 0:
                raise ValueError(f"{self.kind} prior needs mean and std > 0")
        if self.kind == "beta" and (self.a is None or self.b is None or self.a <= 0 or self.b <= 0):
            raise ValueError("beta prior needs shape parameters a > 0 and b > 0")
        if self.kind == "gamma" and (self.a is None or self.scale is None or self.a <= 0 or self.scale <= 0):
            raise ValueError("gamma prior needs shape a > 0 and scale > 0")
        if self.kind == "gamma" and self.low is not None and self.low < 0:
            raise ValueError("gamma prior needs low >= 0")
        if self.low is not None and self.high is not None and self.high <= self.low:
            raise ValueError("prior needs high > low")
        if self.kind == "loguniform" and self.low is not None and self.low <= 0:
            raise ValueError("log-uniform prior needs low > 0")
        if self.kind == "lognormal" and self.low is not None and self.low < 0:
            raise ValueError("log-normal prior needs low >= 0")
        return self


class Priors(_Base):
    v_ctr: Range = Field(
        default_factory=lambda: Range(kind="uniform"),
        description="Line centre [km/s]. Uniform bounds default to the band minus `v_margin`.",
    )
    v_margin: float = Field(0.05, ge=0, lt=0.5, description="Band fraction excluded at each edge")
    # Defaults from the 34 THINGS fits of Stewart et al. (2014, Table 1 and App. B.2),
    # widened for survey galaxies; see docs/priors.md.
    delta_v: Range = Field(
        default_factory=lambda: Range(kind="gamma", a=2.3, scale=83.0, low=0.0, high=800.0),
        description="Rotational width [km/s]: gamma with shape 2.3 and scale 83 km/s (median 166, "
        "16-84 % 77-305 km/s), truncated at 800 km/s; fitted so that the predicted W50 matches the "
        "25 434 ALFALFA alpha.100 detections (benchmarks/fit_width_prior.py)",
    )
    delta_v_rand: Range = Field(
        default_factory=lambda: Range(kind="lognormal", mean=1.1, std=0.2, low=4.0, high=60.0),
        description="Random motions [km/s]: log-normal around 12.6 km/s (THINGS: 8-32 km/s)",
    )
    alpha: Range = Field(
        default_factory=lambda: Range(kind="normal", mean=0.0, std=0.25, low=-1.0, high=1.0),
        description="Asymmetry: normal(0, 0.25) truncated to [-1, 1] (THINGS: median 0.00, robust std 0.14)",
    )
    f: Range = Field(
        default_factory=lambda: Range(kind="beta", a=1.0, b=3.0, low=0.0, high=1.0),
        description="Solid-body fraction: Beta(1, 3), density 3 (1 - f)^2, median 0.21, P(f > 0.7) = 2.7 %; "
        "fitted to the 34 THINGS galaxies (median 0.15, 3 % above 0.7), which reject a uniform prior",
    )
    amplitude_scale: float | Literal["auto"] = Field(
        "auto",
        description="Std of the half-normal prior on the integrated flux S of each component "
        "[flux * km/s]. Fix it for a survey so that evidences are comparable between spectra.",
    )
    baseline_scale: float | Literal["auto"] = Field(
        "auto", description="Std of the Chebyshev coefficients; auto = 10 x noise rms"
    )
    baseline_mean: list[float] | None = None
    baseline_cov: list[list[float]] | None = None
    rfi_sigma_channels: tuple[float, float] = Field(
        (0.3, 1.0), description="Width range of RFI components in channels"
    )
    noise_scale: tuple[float, float] | None = Field(
        None, description="Fit a global noise scale factor with log-uniform prior in this range"
    )

    @field_validator("alpha", "f")
    @classmethod
    def _bounded(cls, v: Range, info: ValidationInfo) -> Range:
        lo_max, hi_max = (-1.0, 1.0) if info.field_name == "alpha" else (0.0, 1.0)
        if v.kind not in ("uniform", "normal", "beta"):
            raise ValueError(f"{info.field_name} prior must be uniform, (truncated) normal or beta")
        low = lo_max if v.low is None else v.low
        high = hi_max if v.high is None else v.high
        if low < lo_max or high > hi_max:
            raise ValueError(f"{info.field_name} prior must lie within [{lo_max}, {hi_max}]")
        return v.model_copy(update={"low": low, "high": high})


class ModelConfig(_Base):
    n_max: int = Field(2, ge=0, le=5, description="Largest number of profile components")
    k_min: int = Field(0, ge=-1)
    k_max: int = Field(3, ge=-1, le=12, description="Largest Chebyshev baseline order")
    n_rfi: int = Field(0, ge=0, le=3, description="Narrow RFI components added to every model")
    poisson_rate: float = Field(0.5, gt=0, description="Poisson prior mean of N")
    response: str = "boxcar"
    supersample: int | None = Field(None, ge=1)

    @model_validator(mode="after")
    def _k(self) -> ModelConfig:
        if self.k_max < self.k_min:
            raise ValueError("k_max must be >= k_min")
        return self


class NoiseConfig(_Base):
    sigma: Literal["auto", "input", "estimate"] = Field(
        "auto", description="auto: use the input sigma column if present, else estimate"
    )
    correlation: str | list[float] = Field(
        "auto",
        description="auto | estimate | white | hanning | hanning-decimated | [rho_1, rho_2, ...]; a '+white' "
        "suffix (e.g. hanning+white) adds a white component whose fraction is estimated from the data. "
        "auto: the correlation of the channel smoothing in model.response (hanning, kernel) plus an "
        "estimated white part, white without smoothing. estimate: rho_1..rho_m measured on the residuals "
        "of the robust pre-fit (any smoothing, regridding or spectrometer window)",
    )
    signal_free: list[tuple[float, float]] | None = Field(
        None, description="Velocity ranges used to estimate sigma"
    )


class RFIConfig(_Base):
    prefit: bool = True
    background: Literal["positive", "symmetric"] = "positive"
    threshold: float = Field(0.5, gt=0, lt=1)
    action: Literal["mask", "inflate"] = "mask"
    n_iter: int = Field(8, ge=1)
    flag_fraction: float = Field(0.05, gt=0, le=1)
    min_sigma_channels: float = Field(
        2.0,
        gt=0,
        description="Smallest Delta v_rand (in channels) of the line model in the pre-stage, so that a "
        "spike of one or two channels cannot be absorbed as a line",
    )
    close_holes: bool = Field(True, description="Also mask a single channel between two masked ones")
    max_width: float | None = Field(
        20.0,
        gt=0,
        description="Longest run of outlier channels treated as RFI, in km/s (gaps of up to two channels "
        "included; at least one channel): a longer run is signal the one-component pre-stage cannot "
        "explain, typically a second line, and stays in the fit. In velocity so that 1 km/s and 15 km/s "
        "channels behave alike; null: no limit",
    )


class SamplerConfig(_Base):
    mode: Literal["quick", "full", "check"] = "full"
    backend: Literal["nautilus", "dynesty", "ultranest", "smc", "quick"] = Field(
        "smc",
        description="smc: fastest (vectorised tempered SMC); nautilus: most precise ln Z but ~10x slower",
    )
    check_backend: Literal["nautilus", "dynesty", "ultranest", "smc"] = "nautilus"
    n_live: int = Field(400, ge=20)
    gate_margin: float | None = Field(
        5.0,
        description="Skip nested sampling for N when its quick evidence is this many nats below the "
        "best model with fewer components (null: always sample)",
    )
    n_seeds: int = Field(
        3,
        ge=0,
        description="Warm starts of the quick stage for N >= 2: the N-1 solution plus a new component on "
        "each of the strongest residual features (random starts rarely hit a second line)",
    )
    init_fraction: float = Field(
        0.25,
        ge=0,
        lt=1,
        description="Fraction of the SMC particles drawn from the quick-stage posterior instead of the "
        "prior (defensive mixture with importance weights, the evidence stays exact); 0 disables",
    )
    options: dict[str, Any] = Field(default_factory=dict)
    likelihood_backend: Literal["numpy", "jax"] = "numpy"


class DiagnosticsConfig(_Base):
    ppp_min: float = 0.01
    wiggle_max: float = 0.1
    acf_sigma: float = 4.0
    edge_mass: float = 0.2
    min_ess: float = 500.0
    logz_sigma: float = 3.0
    min_w50_channels: float = Field(
        2.0, description="Flag `unresolved` when a component's W50 is narrower than this many channels"
    )
    min_constraint: float = Field(
        0.3,
        description="Flag `prior_dominated` when the posterior 16-84 % width of delta_v is wider than "
        "(1 - min_constraint) times the prior width, i.e. the line width is not measured",
    )
    negative_fraction: float = Field(
        0.5,
        description="Flag when this fraction of amplitude draws is negative. Narrow profiles make "
        "ring and disk nearly degenerate, so 10 % (roadmap value) flags most of them.",
    )
    multimodal_weight: float = 0.1


class OutputConfig(_Base):
    n_posterior: int = Field(2000, ge=100, description="Posterior samples stored per model")
    plots: Literal["none", "flagged", "sample", "all"] = "sample"
    plot_fraction: float = Field(0.02, ge=0, le=1)
    catalog_format: Literal["fits", "parquet", "csv"] = "fits"


class Config(_Base):
    model: ModelConfig = Field(default_factory=ModelConfig)
    priors: Priors = Field(default_factory=Priors)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    rfi: RFIConfig = Field(default_factory=RFIConfig)
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    output: OutputConfig = Field(default_factory=lambda: OutputConfig())
    seed: int = 20141

    @classmethod
    def load(cls, path: str | Path | None) -> Config:
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)

    def dump(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    def updated(self, **sections: dict[str, Any]) -> Config:
        """Return a copy with nested fields overridden, e.g. ``updated(sampler={"mode": "quick"})``."""
        data = self.model_dump(mode="json")
        for key, val in sections.items():
            if isinstance(val, dict):
                data.setdefault(key, {}).update(val)
            else:
                data[key] = val
        return Config.model_validate(data)
