"""Result containers of a fit: evidences, posteriors, the per-spectrum result and seeds."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hiprofile import __version__

log = logging.getLogger(__name__)

# Parameters whose posterior width is compared with the prior width
CONSTRAINT_PARAMS = ("delta_v", "delta_v_rand", "alpha", "f")


@dataclass
class ModelEvidence:
    n: int
    k: int
    log_z: float  # with positivity
    log_z_err: float
    log_z_gauss: float  # evidence of the sampling target before the Monte Carlo correction
    method: str  # analytic | marginal | quick | quick-gated
    backend: str = ""
    ess_fraction: float = 1.0
    negative_fraction: float = 0.0
    runtime: float = 0.0
    n_like: int = 0
    budget_exhausted: bool = False
    log_z_check: float = float("nan")
    log_z_check_err: float = float("nan")


@dataclass
class Posterior:
    n: int
    k: int  # most probable baseline order
    names: list[str]
    theta: np.ndarray  # (m, d)
    u: np.ndarray  # (m, d) unit-cube coordinates
    weights: np.ndarray  # (m,) normalised
    beta: np.ndarray  # (m, p) linear parameters (amplitudes, baseline up to K_max, zero-padded)
    log_l: np.ndarray
    derived: dict[str, np.ndarray]  # "<name>_<i>" -> (m,)
    trace: dict[str, np.ndarray] = field(default_factory=dict)
    k_samples: np.ndarray | None = None  # baseline order drawn for each sample
    k_values: list[int] = field(default_factory=list)
    k_probability: np.ndarray | None = None  # P(K | N, y) over k_values

    @property
    def k_max(self) -> int:
        return int(max(self.k_values)) if self.k_values else self.k

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        cols = {**{nm: self.theta[:, i] for i, nm in enumerate(self.names)}, **self.derived}
        for key, val in cols.items():
            ok = np.isfinite(val)
            if not ok.any():
                continue
            q = weighted_quantile(val[ok], self.weights[ok], [0.16, 0.5, 0.84])
            out[f"{key}_p16"], out[key], out[f"{key}_p84"] = q
        return out


@dataclass
class FitResult:
    spectrum_id: str
    status: str = "ok"
    error: str = ""
    version: str = __version__
    seed: int = 0
    config: str = ""
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    flux: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sigma: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mask: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    rfi_q: np.ndarray = field(default_factory=lambda: np.zeros(0))
    evidences: list[ModelEvidence] = field(default_factory=list)
    p_n: dict[int, float] = field(default_factory=dict)
    p_detect: float = float("nan")
    best_n: int = 0
    best_k: int = 0
    posteriors: dict[int, Posterior] = field(default_factory=dict)
    model_band: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # (5, n_chan)
    components_median: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    baseline_median: np.ndarray = field(default_factory=lambda: np.zeros(0))
    metrics: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)
    classic: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    runtime: float = 0.0

    @property
    def flagged(self) -> bool:
        return any(self.flags.values())

    def catalog_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": self.spectrum_id,
            "status": self.status,
            "best_n": self.best_n,
            "best_k": self.best_k,
            "p_detect": self.p_detect,
            "runtime": self.runtime,
        }
        for n, p in self.p_n.items():
            row[f"p_n{n}"] = p
        for ev in self.evidences:
            row[f"logz_n{ev.n}_k{ev.k}"] = ev.log_z
        post = self.posteriors.get(max(self.best_n, 1)) if self.posteriors else None
        if post is not None:
            for key, val in post.summary().items():
                row[key] = val
            if post.k_probability is not None:
                for k, pk in zip(post.k_values, post.k_probability, strict=True):
                    row[f"p_k{k}"] = float(pk)
        for key, val in self.metrics.items():
            if np.isscalar(val):
                row[f"m_{key}"] = val
        for key, val in self.flags.items():
            row[f"flag_{key}"] = bool(val)
        row["flagged"] = self.flagged
        for key, val in self.classic.items():
            row[f"classic_{key}"] = val
        for key, val in self.meta.items():
            if np.isscalar(val):
                row[f"meta_{key}"] = val
        return row


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: list[float]) -> list[float]:
    order = np.argsort(x)
    x, w = x[order], w[order]
    c = np.cumsum(w)
    c = (c - 0.5 * w) / c[-1]
    return [float(np.interp(qq, c, x)) for qq in q]


def spectrum_seed(global_seed: int, spectrum_id: str) -> int:
    """Seed derived from the global seed and the spectrum id (independent of order)."""
    import hashlib

    h = hashlib.blake2b(f"{global_seed}:{spectrum_id}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") % (2**63)
