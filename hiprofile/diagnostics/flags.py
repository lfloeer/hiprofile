"""Turn metrics into boolean flags using the thresholds of the configuration."""

from __future__ import annotations

from typing import Any

import numpy as np

from hiprofile.config import DiagnosticsConfig

FLAG_NAMES = (
    "ppp",
    "wiggle",
    "residual_acf",
    "prior_edge",
    "multimodal",
    "low_ess",
    "logz_mismatch",
    "negative_amplitudes",
    "rfi",
    "failed",
)

MULTIMODAL_SEPARATION = 4.0
WIGGLE_SIGNIFICANCE = 3.0


def compute_flags(metrics: dict[str, Any], cfg: DiagnosticsConfig, rfi_fraction_max: float) -> dict[str, bool]:
    def get(key: str, default: float = np.nan) -> float:
        v = metrics.get(key, default)
        return float(v) if v is not None else default

    acf = np.asarray(metrics.get("residual_acf", [0.0]), float)
    flags = {
        "ppp": get("ppp") < cfg.ppp_min,
        "wiggle": get("wiggle_J") > cfg.wiggle_max and get("wiggle_snr", np.inf) > WIGGLE_SIGNIFICANCE,
        "residual_acf": bool(np.any(np.abs(acf) > cfg.acf_sigma)),
        "prior_edge": get("edge_mass", 0.0) > cfg.edge_mass,
        "multimodal": get("multimodal_weight", 0.0) > cfg.multimodal_weight
        and get("multimodal_separation", 0.0) > MULTIMODAL_SEPARATION,
        "low_ess": get("ess", np.inf) < cfg.min_ess,
        "logz_mismatch": get("logz_check_sigma", 0.0) > cfg.logz_sigma,
        "negative_amplitudes": get("negative_fraction", 0.0) > cfg.negative_fraction,
        "rfi": get("rfi_fraction", 0.0) > rfi_fraction_max,
        "failed": bool(metrics.get("failed", False)),
    }
    return {k: bool(v) for k, v in flags.items()}
