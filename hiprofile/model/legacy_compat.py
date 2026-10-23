"""Evaluation that reproduces the legacy Cython ``LineModel`` exactly.

The legacy code evaluates the model at ``n_chan * supersample`` points with
spacing ``dv / supersample`` starting at the first channel centre, on a periodic
grid without padding, and averages each block of ``supersample`` points. The
averaged block is therefore centred ``dv (s - 1) / (2 s)`` above the channel
centre, and profiles wrap around the band edges. The new kernel fixes both; this
module only exists to validate the shared Fourier coefficients against the
golden files in ``tests/reference``.

Legacy parameters: (log10 S, v_ctr, dv, dv_rand, f, alpha).
"""

from __future__ import annotations

import numpy as np

from hiprofile._backend import get_xp
from hiprofile.model.kernel import SpectralGrid, profile_fourier


def legacy_profile(velocity: np.ndarray, p, supersample: int = 2, backend: str = "numpy"):
    xp = get_xp(backend)
    v = np.asarray(velocity, dtype=float)
    n = v.size
    dv = float(v[1] - v[0])
    grid = SpectralGrid(v0=float(v[0]), dv=dv, n_chan=n, supersample=supersample, n_total=n, response="none")
    p = xp.asarray(p)
    theta = xp.stack([p[..., 1], p[..., 2], p[..., 3], p[..., 5]], axis=-1)
    coeffs = profile_fourier(grid, theta, backend)
    s = (10.0 ** p[..., 0])[..., None]
    f = p[..., 4][..., None]
    c = s * ((1.0 - f) * coeffs[..., 0, :] + f * coeffs[..., 1, :])
    fine = xp.fft.irfft(c, n=grid.n_fft, axis=-1) / grid.h
    return fine.reshape((*fine.shape[:-1], n, supersample)).mean(axis=-1)
