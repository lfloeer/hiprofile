"""hiprofile: Bayesian fitting of global HI line profiles.

The line model follows Stewart, Blyth & de Blok (2014), A&A 567, A61: a rotating
ring plus a solid-body disk, multiplied by an asymmetry term and convolved with a
Gaussian for random motions. The fluxes of ring and disk, and the baseline, enter
linearly and are marginalised analytically.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hiprofile")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout
    __version__ = "0.1.0.dev0"

from hiprofile.spectrum import Spectrum

__all__ = ["Spectrum", "__version__"]
