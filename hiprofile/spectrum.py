"""In-memory representation of a single 1D spectrum."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Spectrum:
    """A 1D spectrum on a uniform velocity grid.

    Attributes
    ----------
    velocity : (n,) float
        Channel centres in km/s, strictly ascending after :meth:`normalized`.
    flux : (n,) float
        Flux density per channel (e.g. Jy). The integrated flux S of the model is
        in units of ``flux * km/s``.
    sigma : (n,) float or None
        Per-channel noise; estimated from the data when missing.
    mask : (n,) bool or None
        ``True`` marks a channel to *exclude* (blanked, known RFI).
    id : str
        Identifier used for output file names and seeds.
    meta : dict
        Free-form metadata (position, catalogue velocity, file name, ...).
    """

    velocity: np.ndarray
    flux: np.ndarray
    sigma: np.ndarray | None = None
    mask: np.ndarray | None = None
    id: str = "spectrum"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.velocity = np.asarray(self.velocity, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        n = self.velocity.shape[0]
        if self.velocity.ndim != 1 or self.flux.shape != (n,):
            raise ValueError("velocity and flux must be 1D arrays of equal length")
        if n < 8:
            raise ValueError("spectrum needs at least 8 channels")
        if self.sigma is not None:
            self.sigma = np.broadcast_to(np.asarray(self.sigma, dtype=float), (n,)).copy()
        if self.mask is not None:
            self.mask = np.asarray(self.mask, dtype=bool).reshape(n).copy()

    @property
    def n_chan(self) -> int:
        return int(self.velocity.shape[0])

    @property
    def channel_width(self) -> float:
        return float(np.median(np.diff(self.velocity)))

    @property
    def good(self) -> np.ndarray:
        """Boolean array of channels used in the fit."""
        good = np.isfinite(self.flux)
        if self.mask is not None:
            good &= ~self.mask
        if self.sigma is not None:
            good &= np.isfinite(self.sigma) & (self.sigma > 0)
        return good

    def normalized(self, rtol: float = 1e-3) -> Spectrum:
        """Return a copy with ascending velocities, checked for a uniform grid.

        Non-finite flux values are turned into masked channels.
        """
        order = np.argsort(self.velocity)
        v = self.velocity[order]
        dv = np.diff(v)
        if np.any(dv <= 0):
            raise ValueError(f"{self.id}: duplicate velocities")
        if np.max(np.abs(dv - np.median(dv))) > rtol * np.median(dv):
            raise ValueError(
                f"{self.id}: velocity grid is not uniform to rtol={rtol}; "
                "resample the spectrum first"
            )
        flux = self.flux[order]
        mask = np.zeros(v.size, bool) if self.mask is None else self.mask[order]
        mask = mask | ~np.isfinite(flux)
        sigma = None if self.sigma is None else self.sigma[order]
        return Spectrum(
            velocity=v,
            flux=np.where(np.isfinite(flux), flux, 0.0),
            sigma=sigma,
            mask=mask,
            id=self.id,
            meta=dict(self.meta),
        )
