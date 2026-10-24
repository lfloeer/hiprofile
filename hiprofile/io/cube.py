"""Spectra from data cubes (extension stage): aperture sum around a position.

A catalogue row with ``cube``, ``ra``, ``dec`` (degrees) and optionally
``radius`` (arcsec, default: 1.5 beam FWHM) yields the spectrum summed over a
circular aperture, converted from Jy/beam to Jy with the beam area. This is the
hook for fitting local sub-cubes around candidates later on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hiprofile.io.spectra import HI_REST_FREQUENCY_HZ
from hiprofile.spectrum import Spectrum


def extract_spectrum(cube_path: str, meta: dict[str, Any], sid: str) -> Spectrum:
    import astropy.units as u
    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(cube_path, memmap=True) as hdul:
        hdu = hdul[0]
        data = np.asarray(hdu.data, dtype=float)
        hdr = hdu.header
    while data.ndim > 3:
        data = data[0]
    wcs = WCS(hdr).celestial
    ra, dec = float(meta["ra"]), float(meta["dec"])
    x0, y0 = wcs.world_to_pixel_values(ra, dec)
    pix = abs(hdr.get("CDELT2", hdr.get("CD2_2", 1.0))) * 3600.0
    bmaj = hdr.get("BMAJ", pix / 3600.0) * 3600.0
    bmin = hdr.get("BMIN", pix / 3600.0) * 3600.0
    radius = float(meta.get("radius", 1.5 * bmaj)) / pix
    yy, xx = np.mgrid[: data.shape[1], : data.shape[2]]
    ap = (xx - x0) ** 2 + (yy - y0) ** 2 <= radius**2
    beam_pix = np.pi * bmaj * bmin / (4 * np.log(2)) / pix**2
    flux = np.nansum(data[:, ap], axis=1) / beam_pix
    n = data.shape[0]
    ax = np.arange(n)
    crval, cdelt, crpix = hdr["CRVAL3"], hdr["CDELT3"], hdr["CRPIX3"]
    world = crval + (ax + 1 - crpix) * cdelt
    ctype = str(hdr.get("CTYPE3", "VRAD")).upper()
    unit = u.Unit(hdr.get("CUNIT3", "Hz" if ctype.startswith("FREQ") else "m/s"))
    if ctype.startswith("FREQ"):
        rest = hdr.get("RESTFRQ", hdr.get("RESTFREQ", HI_REST_FREQUENCY_HZ)) * u.Hz
        vel = (world * unit).to(u.km / u.s, equivalencies=u.doppler_radio(rest)).value
    else:
        vel = (world * unit).to(u.km / u.s).value
    n_beams = ap.sum() / beam_pix
    return Spectrum(
        velocity=vel,
        flux=flux,
        id=sid,
        meta={**{k: v for k, v in meta.items() if np.isscalar(v)}, "aperture_beams": float(n_beams)},
    )
