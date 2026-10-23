"""Run configuration (YAML), validated with pydantic.

The full configuration is written into every result file together with the
code version and the seed, so that each fit can be reproduced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Range(_Base):
    """A one-dimensional prior: uniform, log-uniform or normal."""

    kind: Literal["uniform", "loguniform", "normal"] = "uniform"
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    std: float | None = None

    @model_validator(mode="after")
    def _check(self) -> Range:
        if self.kind == "normal":
            if self.mean is None or self.std is None or self.std <= 0:
                raise ValueError("normal prior needs mean and std > 0")
        elif self.low is not None and self.high is not None:
            if self.high <= self.low:
                raise ValueError("prior needs high > low")
            if self.kind == "loguniform" and self.low <= 0:
                raise ValueError("log-uniform prior needs low > 0")
        return self


class Priors(_Base):
    v_ctr: Range = Field(
        default_factory=lambda: Range(kind="uniform"),
        description="Line centre [km/s]. Uniform bounds default to the band minus `v_margin`.",
    )
    v_margin: float = Field(0.05, ge=0, lt=0.5, description="Band fraction excluded at each edge")
    delta_v: Range = Field(default_factory=lambda: Range(kind="uniform", low=0.0, high=600.0))
    delta_v_rand: Range = Field(default_factory=lambda: Range(kind="loguniform", low=4.0, high=60.0))
    alpha: Range = Field(default_factory=lambda: Range(kind="uniform", low=-1.0, high=1.0))
    amplitude_scale: float | Literal["auto"] = Field(
        "auto",
        description="Std of the half-normal prior on a_ring and a_disk [flux * km/s]. "
        "Fix it for a survey so that evidences are comparable between spectra.",
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

    @field_validator("alpha")
    @classmethod
    def _alpha(cls, v: Range) -> Range:
        if v.kind != "uniform" or (v.low or -1) < -1 or (v.high or 1) > 1:
            raise ValueError("alpha prior must be uniform within [-1, 1]")
        return v


class ModelConfig(_Base):
    n_max: int = Field(2, ge=0, le=5, description="Largest number of profile components")
    k_min: int = Field(0, ge=-1)
    k_max: int = Field(3, ge=-1, le=12, description="Largest Chebyshev baseline order")
    n_rfi: int = Field(0, ge=0, le=3, description="Narrow RFI components added to every model")
    poisson_rate: float = Field(0.5, gt=0, description="Poisson prior mean of N")
    parametrization: Literal["direct", "effective"] = "direct"
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
        description="white | hanning | hanning-decimated | [rho_1, rho_2, ...]; "
        "auto: hanning if the response contains hanning",
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


class SamplerConfig(_Base):
    mode: Literal["quick", "full", "check"] = "full"
    backend: Literal["nautilus", "dynesty", "ultranest", "smc", "quick"] = "nautilus"
    check_backend: Literal["nautilus", "dynesty", "ultranest", "smc"] = "dynesty"
    n_live: int = Field(400, ge=20)
    baseline_evidence: Literal["reweight", "rerun"] = Field(
        "reweight",
        description="reweight: sample once per N at the preferred K and importance-reweight "
        "to the other K (rerun when the effective sample size is too small)",
    )
    reweight_min_ess: float = Field(0.2, gt=0, le=1)
    gate_margin: float | None = Field(
        5.0,
        description="Skip nested sampling for N when its quick evidence is this many nats below the "
        "best model with fewer components (null: always sample)",
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
