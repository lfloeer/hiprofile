"""Generate the frozen golden files and the speed baseline from the legacy Cython code.

Run from the ``legacy/`` directory after ``python setup.py build_ext --inplace``::

    python make_reference.py            # write ../tests/reference/*
    python make_reference.py --check    # compare against the stored files

The golden files are the reference for the new model kernel (see
``tests/test_golden.py``). The parameter grid covers all six legacy parameters,
including the edges f in {0, 1}, alpha = +-1 and Delta v -> 0, on several
channel widths and supersampling factors.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import timeit
from pathlib import Path

import numpy as np

from lineprofile.fitter import FitGaussian, FitMixture
from lineprofile.model import LineModel

OUT = Path(__file__).resolve().parent.parent / "tests" / "reference"

# (channel width [km/s], number of channels, supersample)
GRIDS = [(2.0, 512, 2), (5.0, 512, 2), (10.24, 256, 2), (5.0, 512, 4)]

DELTA_V = [1e-3, 50.0, 230.0, 500.0]
SIGMA = [2.0, 8.0, 25.0]
FSOLID = [0.0, 0.3, 1.0]
ALPHA = [-1.0, -0.2, 0.0, 1.0]


def parameter_grid(velocities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Full factorial grid over (Delta v, sigma, f, alpha); flux and centre vary."""
    rows = []
    span = velocities[-1] - velocities[0]
    for dv, sig, f, a in itertools.product(DELTA_V, SIGMA, FSOLID, ALPHA):
        log_s = rng.uniform(-0.5, 1.5)
        # Non-integer channel offsets around the band centre
        vc = velocities[0] + span * rng.uniform(0.4, 0.6)
        rows.append([log_s, vc, dv, sig, f, a])
    return np.array(rows)


# Legacy ``LineModel.__dealloc__`` calls ``fftw_cleanup()``, which frees FFTW's
# global state while other plans are still alive (undefined behaviour, segfaults
# in practice). Keep every model alive until the interpreter exits.
_KEEP_ALIVE: list[object] = []


def make_profiles(seed: int = 20141) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for gi, (dv, n, ss) in enumerate(GRIDS):
        v = 1000.0 + np.arange(n) * dv
        params = parameter_grid(v, rng)
        model = LineModel(v, n_disks=1, supersample=ss)
        _KEEP_ALIVE.append(model)
        prof = np.array([model.model(np.ascontiguousarray(p)) for p in params])
        out[f"grid{gi}_velocities"] = v
        out[f"grid{gi}_params"] = params
        out[f"grid{gi}_profiles"] = prof
        out[f"grid{gi}_supersample"] = np.array(ss)
    return out


def make_likelihoods(seed: int = 7) -> dict[str, np.ndarray]:
    """A few legacy log-posterior values (recorded only, not compared to new code)."""
    rng = np.random.default_rng(seed)
    v = np.arange(256) * 10.24
    p = np.array([np.log10(30.0), v.mean(), 230.0, 15.0, 0.2, -0.1])
    _KEEP_ALIVE.append(gen := LineModel(v))
    data = gen.model(p) + rng.normal(0, 0.023, v.size)
    g = FitGaussian(v, data, n_disks=1)
    _KEEP_ALIVE.append(g)
    g.v_center_mean = [v.mean()]
    g.v_center_std = [30.0]
    m = FitMixture(v, data, n_disks=1)
    _KEEP_ALIVE.append(m)
    m.v_center_mean = [v.mean()]
    m.v_center_std = [30.0]
    pg = np.r_[p, np.log10(0.023)]
    pm = np.r_[p, -2.0, np.log10(0.023), 0.0, -1.0]
    return {
        "lnlike_velocities": v,
        "lnlike_data": data,
        "lnlike_gauss_params": pg,
        "lnlike_gauss_value": np.array(g.ln_posterior(pg)),
        "lnlike_mixture_params": pm,
        "lnlike_mixture_value": np.array(m.ln_posterior(pm)),
    }


def benchmark() -> dict[str, object]:
    res: dict[str, object] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "timings_us": {},
    }
    p = np.array([1.0, 0.0, 230.0, 15.0, 0.2, -0.1])
    for n, ss in [(256, 2), (512, 2), (1024, 2), (2048, 2)]:
        v = np.arange(n) * 5.0
        p[1] = v.mean()
        m = LineModel(v, supersample=ss)
        _KEEP_ALIVE.append(m)
        number = 5000
        t = min(timeit.repeat(lambda: m.model(p, False), number=number, repeat=5)) / number
        res["timings_us"][f"n{n}_s{ss}"] = t * 1e6  # type: ignore[index]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare instead of writing")
    ap.add_argument("--rtol", type=float, default=1e-10)
    args = ap.parse_args()

    data = make_profiles()
    data.update(make_likelihoods())

    if args.check:
        ref = np.load(OUT / "golden_profiles.npz")
        bad = []
        for k, val in data.items():
            r = ref[k]
            scale = max(np.max(np.abs(r)), 1e-300)
            if not np.allclose(val, r, rtol=0, atol=args.rtol * scale):
                bad.append(k)
        if bad:
            print("MISMATCH:", ", ".join(bad))
            raise SystemExit(1)
        print(f"legacy code reproduces {len(data)} golden arrays")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "golden_profiles.npz", **data)
    (OUT / "benchmark_legacy.json").write_text(json.dumps(benchmark(), indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
