"""Channel integration and spectrometer response in Fourier space.

All responses are multiplicative transfer functions H(omega) applied to the
Fourier transform of the line profile, where omega is the angular frequency
conjugate to velocity (units rad / (km/s)).

The response is given as a string of ``+``-joined terms:

``none``
    Point sampling at the channel centre (no integration).
``boxcar``
    Mean over the channel, H = sinc(omega dv / 2).
``hanning``
    3-tap Hanning smoothing [1/4, 1/2, 1/4] of the channels,
    H = 1/2 + 1/2 cos(omega dv). Exact, because the smoothing kernel is a sum of
    delta functions at multiples of the channel width.
``tukey(r)``
    Tukey lag window with taper fraction r in [0, 1] up to the Nyquist
    frequency omega_N = pi / dv (correlator spectrometers). r = 1 is the Hann
    lag window, r = 0 a sharp cut at omega_N.

Example: ``"boxcar+hanning"`` for an FFT spectrometer with Hanning smoothing.
"""

from __future__ import annotations

import re

import numpy as np

_TERM = re.compile(r"^\s*(none|boxcar|hanning|tukey)\s*(?:\(\s*([0-9.eE+-]+)\s*\))?\s*$")


def parse_response(spec: str) -> list[tuple[str, float | None]]:
    terms: list[tuple[str, float | None]] = []
    for raw in spec.split("+"):
        m = _TERM.match(raw.lower())
        if not m:
            raise ValueError(f"unknown response term {raw!r} in {spec!r}")
        name, arg = m.group(1), m.group(2)
        if name == "tukey":
            r = 0.5 if arg is None else float(arg)
            if not 0.0 <= r <= 1.0:
                raise ValueError("tukey taper fraction must be within [0, 1]")
            terms.append((name, r))
        else:
            if arg is not None:
                raise ValueError(f"response term {name!r} takes no argument")
            terms.append((name, None))
    return terms


def transfer_function(spec: str, omega: np.ndarray, dv: float) -> np.ndarray:
    """Evaluate H(omega) (real, even) for channel width ``dv``."""
    omega = np.abs(np.asarray(omega, dtype=float))
    h = np.ones_like(omega)
    for name, arg in parse_response(spec):
        if name == "none":
            continue
        if name == "boxcar":
            h = h * np.sinc(omega * dv / (2.0 * np.pi))
        elif name == "hanning":
            h = h * (0.5 + 0.5 * np.cos(omega * dv))
        elif name == "tukey":
            assert arg is not None
            w_n = np.pi / dv
            t = omega / w_n
            taper = np.ones_like(t)
            if arg > 0:
                lo = 1.0 - arg
                ramp = (t > lo) & (t < 1.0)
                taper = np.where(ramp, 0.5 * (1.0 + np.cos(np.pi * (t - lo) / arg)), taper)
            taper = np.where(t >= 1.0, 0.0, taper)
            h = h * taper
    return h


def noise_kernel(spec: str) -> np.ndarray:
    """Channel-domain smoothing kernel that the response applies to white noise.

    Only ``hanning`` correlates the noise between channels in this model (the
    boxcar/tukey terms describe how the *signal* is sampled and are assumed to
    leave the per-channel noise white).
    """
    k = np.array([1.0])
    for name, _ in parse_response(spec):
        if name == "hanning":
            k = np.convolve(k, [0.25, 0.5, 0.25])
    return k
