"""Add the Springob et al. (2005) widths to existing result files without refitting.

    python benchmarks/add_springob.py DIR [DIR ...]

Recomputes both classical measurements (``classic_measurements``) from the
stored spectrum, mask, noise sigma and posterior-median baseline of every
``*.h5`` below the directories and rewrites the ``classic`` and
``catalog_row`` attributes. The naive measurement must reproduce the stored
one; the largest difference is printed. Afterwards ``hiprofile validate
--resume`` (same stage, n and seed) or ``bias_study.py`` re-analyses the runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from hiprofile.inference.derived import classic_measurements


def update(path: Path) -> float:
    with h5py.File(path, "r+") as f:
        if "catalog_row" not in f.attrs:  # simulated input spectra
            return 0.0
        row = json.loads(f.attrs["catalog_row"])
        if "v_sys_1" not in row or "W20_1" not in row or "spectrum" not in f:
            return 0.0
        old = json.loads(f.attrs["classic"])
        g = f["spectrum"]
        sigma = g["sigma"][()]
        response = yaml.safe_load(f.attrs["config"])["model"]["response"]
        new = classic_measurements(
            g["velocity"][()],
            g["flux"][()] - f["model/baseline_median"][()],
            float(np.nanmedian(sigma)),
            row["v_sys_1"],
            row["W20_1"],
            good=~g["mask"][()],
            hanning="hanning" in response.lower(),
        )
        diff = max(
            (abs(new[k] - old[k]) for k in ("W50", "W20", "v_sys", "S") if old.get(k) is not None),
            default=0.0,
        )
        row.update({f"classic_{k}": (v if np.isfinite(v) else None) for k, v in new.items()})
        f.attrs["classic"] = json.dumps({k: (v if np.isfinite(v) else None) for k, v in new.items()})
        f.attrs["catalog_row"] = json.dumps(row)
    return diff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    for d in args.dirs:
        paths = sorted(Path(d).rglob("*.h5"))
        diffs = [update(p) for p in paths]
        print(f"{d}: {len(paths)} files, largest change of a naive value {max(diffs, default=0.0):.3g}")


if __name__ == "__main__":
    main()
