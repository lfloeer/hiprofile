"""Catalogue table over all result files (FITS, Parquet or CSV)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from hiprofile.io.results import read_catalog_row


def collect_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for p in paths:
        try:
            row = read_catalog_row(p)
        except Exception as exc:
            row = {"id": p.stem, "status": "unreadable", "error": str(exc), "flag_failed": True}
        row["result_file"] = str(p)
        rows.append(row)
    return rows


def rows_to_table(rows: list[dict[str, Any]]):
    from astropy.table import Table

    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    cols = {}
    for k in keys:
        vals = [r.get(k) for r in rows]
        sample = next((v for v in vals if v is not None), None)
        if isinstance(sample, bool):
            cols[k] = np.array([bool(v) if v is not None else False for v in vals])
        elif isinstance(sample, (int, float)):
            cols[k] = np.array([np.nan if v is None else float(v) for v in vals])
            if isinstance(sample, int) and np.all(np.isfinite(cols[k])):
                cols[k] = cols[k].astype(np.int64)
        else:
            cols[k] = np.array(["" if v is None else str(v) for v in vals])
    return Table(cols)


def build_catalog(out_dir: str | Path, fmt: str = "fits") -> Path:
    out_dir = Path(out_dir)
    paths = sorted((out_dir / "spectra").glob("*.h5"))
    table = rows_to_table(collect_rows(paths))
    ext = {"fits": "fits", "parquet": "parquet", "csv": "csv"}[fmt]
    target = out_dir / f"catalog.{ext}"
    if fmt == "parquet":
        table.write(target, format="parquet", overwrite=True)
    elif fmt == "csv":
        table.write(target, format="ascii.csv", overwrite=True)
    else:
        table.write(target, overwrite=True)
    return target


def read_catalog(path: str | Path):
    from astropy.table import Table

    p = Path(path)
    if p.suffix == ".parquet":
        return Table.read(p, format="parquet")
    if p.suffix == ".csv":
        return Table.read(p, format="ascii.csv")
    return Table.read(p)
