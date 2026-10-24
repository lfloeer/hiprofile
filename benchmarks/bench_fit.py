"""Wall time of complete fits (one process) for typical spectra.

    python benchmarks/bench_fit.py [--backend smc] [--compare-numpy]

Spectra: pure noise, one component (S/N 15) and two components, 512 channels,
first-order baseline; default configuration (N <= 2, K = 0..3).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from hiprofile.config import Config
from hiprofile.inference.fit import fit_spectrum
from hiprofile.simulate import SimConfig, simulate_spectrum

CASES = {
    "noise": SimConfig(n_components=0, baseline="chebyshev"),
    "one component": SimConfig(n_components=1, snr_range=(15, 15), baseline="chebyshev"),
    "two components": SimConfig(n_components=2, snr_range=(20, 20), baseline="chebyshev"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="smc")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--compare-numpy", action="store_true", help="also time without numba")
    args = ap.parse_args()
    variants = [("compiled", True)] + ([("numpy", False)] if args.compare_numpy else [])
    for mode in ("quick", "full"):
        cfg = Config().updated(sampler={"mode": mode, "backend": args.backend}, output={"plots": "none"})
        for name, sim in CASES.items():
            for label, compiled in variants:
                import hiprofile.likelihood.fast as fl
                import hiprofile.model.kernel as kern

                kern.USE_COMPILED = compiled
                fl.HAVE_NUMBA = compiled
                times = []
                for i in range(args.repeat):
                    spec, _ = simulate_spectrum(sim, np.random.default_rng(i), f"b{i}")
                    t = time.perf_counter()
                    res = fit_spectrum(spec, cfg)
                    times.append(time.perf_counter() - t)
                    assert res.status == "ok", res.error
                med, worst = np.median(times), max(times)
                print(f"{mode:5s} {name:15s} {label:8s} median {med:6.1f} s  (max {worst:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
