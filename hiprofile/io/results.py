"""Per-spectrum result files (HDF5), written atomically.

Layout::

    /                       attrs: id, status, version, seed, config (YAML), p_detect,
                            best_n, best_k, runtime, JSON: p_n, metrics, flags,
                            classic, meta, catalog_row, error
    /spectrum               velocity, flux, sigma, mask, rfi_q
    /model                  band (2.5/16/50/84/97.5 %), components_median, baseline_median
    /evidence               columns of the evidence table (n, k, log_z, ...)
    /posterior/N<n>         theta, u, weights, beta, log_l; attrs names, k
    /posterior/N<n>/derived S_1, f_1, W50_1, W20_1, v_sys_1, peak_1, ...
    /posterior/N<n>/trace   sampler diagnostics
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from hiprofile.inference.fit import FitResult, ModelEvidence, Posterior


class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _json(obj: Any) -> str:
    return json.dumps(obj, cls=_Encoder)


def write_result(res: FitResult, path: str | Path) -> Path:
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as f:
        a = f.attrs
        a["id"] = res.spectrum_id
        a["status"] = res.status
        a["error"] = res.error
        a["version"] = res.version
        a["seed"] = res.seed
        a["config"] = res.config
        a["p_detect"] = res.p_detect
        a["best_n"] = res.best_n
        a["best_k"] = res.best_k
        a["runtime"] = res.runtime
        a["p_n"] = _json({str(k): v for k, v in res.p_n.items()})
        a["metrics"] = _json(res.metrics)
        a["flags"] = _json(res.flags)
        a["classic"] = _json(res.classic)
        a["meta"] = _json({k: v for k, v in res.meta.items() if _jsonable(v)})
        a["catalog_row"] = _json({k: v for k, v in res.catalog_row().items() if _jsonable(v)})
        g = f.create_group("spectrum")
        for key in ("velocity", "flux", "sigma", "rfi_q"):
            g.create_dataset(key, data=np.asarray(getattr(res, key), float))
        g.create_dataset("mask", data=np.asarray(res.mask, bool))
        g = f.create_group("model")
        g.create_dataset("band", data=res.model_band)
        g.create_dataset("components_median", data=res.components_median)
        g.create_dataset("baseline_median", data=res.baseline_median)
        g = f.create_group("evidence")
        if res.evidences:
            for fld in dataclasses.fields(ModelEvidence):
                vals = [getattr(e, fld.name) for e in res.evidences]
                if isinstance(vals[0], str):
                    g.create_dataset(fld.name, data=np.array(vals, dtype="S"))
                else:
                    g.create_dataset(fld.name, data=np.array(vals))
        pg = f.create_group("posterior")
        for n, post in res.posteriors.items():
            h = pg.create_group(f"N{n}")
            h.attrs["names"] = _json(post.names)
            h.attrs["k"] = post.k
            for key in ("theta", "u", "weights", "beta", "log_l"):
                h.create_dataset(key, data=getattr(post, key), compression="gzip", shuffle=True)
            d = h.create_group("derived")
            for key, val in post.derived.items():
                d.create_dataset(key, data=val, compression="gzip", shuffle=True)
            t = h.create_group("trace")
            for key, val in post.trace.items():
                t.create_dataset(key, data=np.asarray(val), compression="gzip")
    os.replace(tmp, path)
    return path


def _jsonable(v: Any) -> bool:
    try:
        _json(v)
    except TypeError:
        return False
    return True


def read_catalog_row(path: str | Path) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as f:
        return json.loads(f.attrs["catalog_row"])


def read_result(path: str | Path) -> FitResult:
    import h5py

    with h5py.File(path, "r") as f:
        a = f.attrs
        res = FitResult(
            spectrum_id=str(a["id"]),
            status=str(a["status"]),
            error=str(a["error"]),
            version=str(a["version"]),
            seed=int(a["seed"]),
            config=str(a["config"]),
            p_detect=float(a["p_detect"]),
            best_n=int(a["best_n"]),
            best_k=int(a["best_k"]),
            runtime=float(a["runtime"]),
            p_n={int(k): v for k, v in json.loads(a["p_n"]).items()},
            metrics=json.loads(a["metrics"]),
            flags=json.loads(a["flags"]),
            classic=json.loads(a["classic"]),
            meta=json.loads(a["meta"]),
        )
        g = f["spectrum"]
        res.velocity, res.flux, res.sigma, res.rfi_q = (g[k][()] for k in ("velocity", "flux", "sigma", "rfi_q"))
        res.mask = g["mask"][()]
        g = f["model"]
        res.model_band = g["band"][()]
        res.components_median = g["components_median"][()]
        res.baseline_median = g["baseline_median"][()]
        g = f["evidence"]
        if "n" in g:
            cols = {}
            for fld in dataclasses.fields(ModelEvidence):
                v = g[fld.name][()]
                cols[fld.name] = [x.decode() for x in v] if v.dtype.kind == "S" else v.tolist()
            res.evidences = [
                ModelEvidence(**{k: cols[k][i] for k in cols}) for i in range(len(cols["n"]))
            ]
        for name, h in f["posterior"].items():
            n = int(name[1:])
            res.posteriors[n] = Posterior(
                n=n,
                k=int(h.attrs["k"]),
                names=json.loads(h.attrs["names"]),
                theta=h["theta"][()],
                u=h["u"][()],
                weights=h["weights"][()],
                beta=h["beta"][()],
                log_l=h["log_l"][()],
                derived={k: v[()] for k, v in h["derived"].items()},
                trace={k: v[()] for k, v in h["trace"].items()},
            )
    return res


def result_path(out_dir: str | Path, spectrum_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_.+" else "_" for c in spectrum_id)
    return Path(out_dir) / "spectra" / f"{safe}.h5"
