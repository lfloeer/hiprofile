"""Speed of the model kernel and the marginal likelihood vs. the legacy baseline.

    python benchmarks/bench_kernel.py [--out tests/reference/benchmark_new.json]

"numpy" is the vectorised NumPy kernel, "numba" the compiled kernel used by
default. Settings "legacy-equivalent" use the legacy grid (supersample 2, no padding,
point sampling); "default" uses the accurate default grid (padding, exact
channel integration, supersampling from the smallest dispersion). Batch timings
are per parameter vector for batches of 512 random vectors. The new kernel
returns ring *and* disk basis functions; the legacy code one finished profile.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from hiprofile._backend import get_xp, has_jax
from hiprofile.likelihood.marginal import build_likelihood
from hiprofile.likelihood.noise import NoiseModel
from hiprofile.model import SpectralGrid, fast, kernel, profile_basis

ROOT = Path(__file__).resolve().parent.parent


def timeit(fn, n_rep: int, inner: int = 1) -> float:
    fn()
    best = np.inf
    for _ in range(n_rep):
        t = time.perf_counter()
        for _ in range(inner):
            fn()
        best = min(best, (time.perf_counter() - t) / inner)
    return best


def theta_batch(rng, n, v):
    return np.c_[
        rng.uniform(v[0] + 0.2 * (v[-1] - v[0]), v[-1] - 0.2 * (v[-1] - v[0]), n),
        rng.uniform(20, 500, n),
        np.exp(rng.uniform(np.log(4), np.log(60), n)),
        rng.uniform(-1, 1, n),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "tests" / "reference" / "benchmark_new.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    if has_jax():
        get_xp("jax")  # enables double precision before any jit
    legacy = json.loads((ROOT / "tests" / "reference" / "benchmark_legacy.json").read_text())
    res: dict = {"python": platform.python_version(), "numpy": np.__version__, "results": {}}
    for n in (256, 512, 1024, 2048):
        v = np.arange(n) * 5.0
        grids = {
            "legacy-equivalent": SpectralGrid(
                v0=0.0, dv=5.0, n_chan=n, supersample=2, n_total=n, response="none"
            ),
            "default": SpectralGrid.from_velocities(v),
        }
        tb = theta_batch(rng, 512, v)
        t1 = tb[:1]
        row: dict = {"legacy_single_us": legacy["timings_us"].get(f"n{n}_s2")}
        for name, g in grids.items():
            kernel.USE_COMPILED = False
            row[f"{name}_numpy_single_us"] = timeit(lambda g=g: profile_basis(g, t1), 5, 200) * 1e6
            row[f"{name}_numpy_batch_us"] = timeit(lambda g=g: profile_basis(g, tb), 5) / 512 * 1e6
            kernel.USE_COMPILED = True
            if fast.available():
                row[f"{name}_numba_single_us"] = timeit(lambda g=g: profile_basis(g, t1), 5, 200) * 1e6
                row[f"{name}_numba_batch_us"] = (
                    timeit(lambda g=g: [profile_basis(g, tb[i : i + 64]) for i in range(0, 512, 64)], 5)
                    / 512
                    * 1e6
                )
            if has_jax():
                import jax

                f = jax.jit(lambda t, g=g: profile_basis(g, t, "jax"))
                row[f"{name}_jax_batch_us"] = timeit(lambda f=f: f(tb).block_until_ready(), 5) / 512 * 1e6
        g = grids["default"]
        y = rng.normal(size=n)
        nm = NoiseModel(sigma=np.ones(n), good_index=np.arange(n))
        like = build_likelihood(y, g, nm, ["profile"], 2, 10.0, 1.0)
        row["loglike_batch_us"] = timeit(lambda: like.loglike(tb), 5) / 512 * 1e6
        row["loglike_single_us"] = timeit(lambda: like.loglike(t1), 5, 200) * 1e6
        if has_jax():
            lj = build_likelihood(y, g, nm, ["profile"], 2, 10.0, 1.0, backend="jax")
            row["loglike_jax_batch_us"] = timeit(lambda: np.asarray(lj.loglike(tb)), 5) / 512 * 1e6
        row["default_supersample"] = g.supersample
        row["default_n_fft"] = g.n_fft
        res["results"][str(n)] = row
        print(n, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()})
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
