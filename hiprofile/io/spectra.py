"""Loading 1D spectra (FITS table, ASCII, HDF5) and spectrum lists.

The spectral axis is normalised to velocity in km/s with astropy.units. A
frequency axis is converted with the radio convention by default, which keeps
a uniform frequency grid uniform in velocity; the optical convention is only
accepted when the resulting grid is uniform within 0.1 % (narrow bands).
"""

from __future__ import annotations

import glob
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hiprofile.spectrum import Spectrum

HI_REST_FREQUENCY_HZ = 1420405751.768

VELOCITY_NAMES = ("velocity", "vel", "v", "vopt", "vrad", "velo", "v_helio", "vhelio")
FREQUENCY_NAMES = ("frequency", "freq", "f", "nu")
FLUX_NAMES = ("flux", "flux_density", "s", "fnu", "f_nu", "intensity", "spec", "data")
SIGMA_NAMES = ("sigma", "error", "err", "noise", "rms", "uncertainty", "flux_err")
MASK_NAMES = ("mask", "flag", "flags", "bad")


@dataclass
class LoadOptions:
    velocity: str | None = None
    frequency: str | None = None
    flux: str | None = None
    sigma: str | None = None
    mask: str | None = None
    velocity_unit: str = "km/s"
    frequency_unit: str = "Hz"
    convention: str = "radio"
    rest_frequency_hz: float = HI_REST_FREQUENCY_HZ
    extra: dict[str, Any] = field(default_factory=dict)


def _pick(names: list[str], wanted: str | None, candidates: tuple[str, ...]) -> str | None:
    lower = {n.lower(): n for n in names}
    if wanted is not None:
        if wanted in names:
            return wanted
        if wanted.lower() in lower:
            return lower[wanted.lower()]
        raise KeyError(f"column {wanted!r} not found in {names}")
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def _frequency_to_velocity(freq_hz: np.ndarray, opts: LoadOptions) -> np.ndarray:
    import astropy.units as u

    f = freq_hz * u.Hz
    rest = opts.rest_frequency_hz * u.Hz
    if opts.convention == "radio":
        eq = u.doppler_radio(rest)
    elif opts.convention == "optical":
        eq = u.doppler_optical(rest)
    elif opts.convention == "relativistic":
        eq = u.doppler_relativistic(rest)
    else:
        raise ValueError(f"unknown velocity convention {opts.convention!r}")
    return f.to(u.km / u.s, equivalencies=eq).value


def _unit_value(col: Any, default_unit: str, target: str) -> np.ndarray:
    import astropy.units as u

    unit = getattr(col, "unit", None) or u.Unit(default_unit)
    q = np.asarray(col, dtype=float) * u.Unit(unit)
    return q.to(u.Unit(target)).value


def _from_columns(cols: dict[str, Any], opts: LoadOptions, sid: str, meta: dict[str, Any]) -> Spectrum:
    names = list(cols)
    vname = _pick(names, opts.velocity, VELOCITY_NAMES)
    fname = None if vname else _pick(names, opts.frequency, FREQUENCY_NAMES)
    flux = _pick(names, opts.flux, FLUX_NAMES)
    if flux is None:
        raise KeyError(f"{sid}: no flux column among {names}")
    if vname is not None:
        vel = _unit_value(cols[vname], opts.velocity_unit, "km/s")
    elif fname is not None:
        vel = _frequency_to_velocity(_unit_value(cols[fname], opts.frequency_unit, "Hz"), opts)
    else:
        raise KeyError(f"{sid}: no velocity or frequency column among {names}")
    sname = _pick(names, opts.sigma, SIGMA_NAMES)
    mname = _pick(names, opts.mask, MASK_NAMES)
    sigma = np.asarray(cols[sname], float) if sname else None
    mask = np.asarray(cols[mname]).astype(bool) if mname else None
    return Spectrum(
        velocity=vel, flux=np.asarray(cols[flux], float), sigma=sigma, mask=mask, id=sid, meta=meta
    )


def load_spectrum(path: str | Path, opts: LoadOptions | None = None, sid: str | None = None) -> Spectrum:
    """Load a spectrum from a FITS table, HDF5 file or ASCII table."""
    opts = opts or LoadOptions()
    path = Path(path)
    sid = sid or path.name.split(".")[0]
    meta: dict[str, Any] = {"path": str(path), **opts.extra}
    suffixes = "".join(path.suffixes).lower()
    if any(s in suffixes for s in (".fits", ".fit", ".fts")):
        from astropy.table import Table

        t = Table.read(path)
        cols = {n: t[n] for n in t.colnames}
        for key in ("RA", "DEC", "OBJECT", "VSYS"):
            if key in t.meta:
                meta[key.lower()] = t.meta[key]
    elif suffixes.endswith((".h5", ".hdf5", ".hdf")):
        import h5py

        with h5py.File(path, "r") as f:
            cols = {k: f[k][()] for k in f if isinstance(f[k], h5py.Dataset)}
            meta.update({k: v for k, v in f.attrs.items() if np.isscalar(v)})
    else:
        from astropy.io import ascii as aascii

        try:
            t = aascii.read(path, guess=True)
            cols = {n: t[n] for n in t.colnames}
            if all(n.startswith("col") for n in cols):
                raise ValueError
        except Exception:
            arr = np.loadtxt(path, ndmin=2)
            names = ["velocity", "flux", "sigma", "mask"][: arr.shape[1]]
            cols = {n: arr[:, i] for i, n in enumerate(names)}
    return _from_columns(cols, opts, sid, meta)


@dataclass
class SpectrumSource:
    """A lazily loaded spectrum: path plus id and metadata (for the worker pool)."""

    path: str
    id: str
    meta: dict[str, Any] = field(default_factory=dict)

    def load(self, opts: LoadOptions | None = None) -> Spectrum:
        o = opts or LoadOptions()
        o = LoadOptions(**{**o.__dict__, "extra": {**o.extra, **self.meta}})
        if self.meta.get("cube"):
            from hiprofile.io.cube import extract_spectrum

            return extract_spectrum(self.meta["cube"], self.meta, self.id)
        return load_spectrum(self.path, o, self.id)


def _unique_ids(sources: list[SpectrumSource]) -> list[SpectrumSource]:
    seen: dict[str, int] = {}
    for s in sources:
        if s.id in seen:
            seen[s.id] += 1
            s.id = f"{s.id}_{seen[s.id]}"
        else:
            seen[s.id] = 0
    return sources


def list_sources(spec: str) -> list[SpectrumSource]:
    """Resolve a spectrum list.

    * a text file with one path per line (optional second column: id),
    * a glob pattern (``"data/*.fits"``),
    * a catalogue table (FITS/CSV/ECSV) with a ``path`` column, or ``cube``,
      ``ra``, ``dec`` columns for spectra extracted from a cube; other columns are
      kept as metadata,
    * a single spectrum file.
    """
    p = Path(spec)
    if any(ch in spec for ch in "*?[") and not p.exists():
        return _unique_ids([SpectrumSource(f, Path(f).name.split(".")[0]) for f in sorted(glob.glob(spec))])
    if not p.exists():
        raise FileNotFoundError(spec)
    suffix = "".join(p.suffixes).lower()
    if suffix.endswith((".txt", ".lst", ".list")):
        out = []
        for line in p.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            path = parts[0]
            if not Path(path).is_absolute():
                path = str((p.parent / path).resolve()) if not Path(path).exists() else path
            sid = parts[1] if len(parts) > 1 else Path(path).name.split(".")[0]
            out.append(SpectrumSource(path, sid))
        return _unique_ids(out)
    if suffix.endswith((".csv", ".ecsv")) or (suffix.endswith((".fits", ".fit")) and _is_catalog(p)):
        from astropy.table import Table

        t = Table.read(p)
        names = [n.lower() for n in t.colnames]
        out = []
        for i, row in enumerate(t):
            meta = {n.lower(): (row[n].item() if hasattr(row[n], "item") else row[n]) for n in t.colnames}
            sid = str(meta.get("id", meta.get("name", f"row{i}")))
            if "path" in names:
                path = str(meta["path"])
                if not Path(path).is_absolute() and not Path(path).exists():
                    path = str((p.parent / path).resolve())
                out.append(SpectrumSource(path, sid, meta))
            elif "cube" in names:
                out.append(SpectrumSource(str(meta["cube"]), sid, meta))
            else:
                raise KeyError(f"{p}: catalogue needs a 'path' or 'cube' column")
        return _unique_ids(out)
    return [SpectrumSource(str(p), p.name.split(".")[0])]


def _is_catalog(path: Path) -> bool:
    try:
        from astropy.table import Table

        t = Table.read(path)
    except Exception:
        return False
    names = [n.lower() for n in t.colnames]
    return "path" in names or "cube" in names


def iter_spectra(sources: list[SpectrumSource], opts: LoadOptions | None = None) -> Iterator[Spectrum]:
    for s in sources:
        yield s.load(opts)


def write_spectrum(spec: Spectrum, path: str | Path) -> None:
    """Write a spectrum as FITS table, HDF5 or ASCII depending on the suffix."""
    path = Path(path)
    suffix = "".join(path.suffixes).lower()
    cols: dict[str, np.ndarray] = {"velocity": spec.velocity, "flux": spec.flux}
    if spec.sigma is not None:
        cols["sigma"] = spec.sigma
    if spec.mask is not None:
        cols["mask"] = spec.mask.astype(np.int16)
    if ".fits" in suffix:
        import astropy.units as u
        from astropy.table import Table

        t = Table(cols)
        t["velocity"].unit = u.km / u.s
        t.meta.update({k[:8].upper(): v for k, v in spec.meta.items() if np.isscalar(v)})
        t.write(path, overwrite=True)
    elif suffix.endswith((".h5", ".hdf5")):
        import h5py

        with h5py.File(path, "w") as f:
            for k, v in cols.items():
                f[k] = v
            for k, v in spec.meta.items():
                if np.isscalar(v):
                    f.attrs[k] = v
    else:
        header = " ".join(cols)
        np.savetxt(path, np.column_stack(list(cols.values())), header=header, comments="")
