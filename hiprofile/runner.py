"""Parallel, resumable fitting of a list of spectra.

One spectrum is one task. Locally the tasks run in a process pool (``jobs``);
on a cluster each SLURM array task processes one chunk of the list
(``chunk=(i, n)``, i = 0..n-1, e.g. ``--chunk $SLURM_ARRAY_TASK_ID/$N``).
Every spectrum writes its own HDF5 file atomically; with ``resume`` finished
files are skipped. Seeds derive from the global seed and the spectrum id, so a
resumed or re-chunked run gives identical results. A failing spectrum is
logged and stored with ``status = failed``; the run continues.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hiprofile.config import Config
from hiprofile.io.results import result_path, write_result
from hiprofile.io.spectra import LoadOptions, SpectrumSource

log = logging.getLogger("hiprofile")

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS")


@dataclass
class TaskOutcome:
    id: str
    status: str
    path: str
    runtime: float
    flagged: bool = False
    error: str = ""


def select_chunk(items: Sequence[Any], chunk: tuple[int, int] | None) -> list[Any]:
    if chunk is None:
        return list(items)
    i, n = chunk
    if not 0 <= i < n:
        raise ValueError("chunk index must satisfy 0 <= i < n")
    return [it for j, it in enumerate(items) if j % n == i]


def parse_chunk(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    i, n = text.split("/")
    return int(i), int(n)


def _plot_wanted(cfg: Config, flagged: bool, sid: str) -> bool:
    mode = cfg.output.plots
    if mode == "none":
        return False
    if mode == "all":
        return True
    if flagged:
        return True
    if mode == "sample":
        # Deterministic pseudo-random selection by id
        import hashlib

        h = int.from_bytes(hashlib.blake2b(sid.encode(), digest_size=4).digest(), "little")
        return (h / 2**32) < cfg.output.plot_fraction
    return False


def fit_task(
    source: SpectrumSource, cfg_text: str, out_dir: str, opts: LoadOptions | None
) -> TaskOutcome:
    """Worker: load, fit, write, plot. Never raises."""
    from hiprofile.inference.fit import FitResult, fit_spectrum

    cfg = Config.model_validate_json(cfg_text)
    path = result_path(out_dir, source.id)
    t0 = time.perf_counter()
    try:
        spec = source.load(opts)
        res = fit_spectrum(spec, cfg)
    except Exception as exc:  # noqa: BLE001 - loading errors end up in the catalogue
        res = FitResult(spectrum_id=source.id, status="failed", error=f"{type(exc).__name__}: {exc}")
        res.flags = {"failed": True}
        res.meta = {"path": source.path}
    res.runtime = time.perf_counter() - t0
    write_result(res, path)
    if res.status == "ok" and _plot_wanted(cfg, res.flagged, source.id):
        try:
            from hiprofile.plots.spectrum import plot_all

            plot_all(res, Path(out_dir) / "plots")
        except Exception as exc:  # noqa: BLE001 - plotting is optional
            log.warning("%s: plotting failed: %s", source.id, exc)
    return TaskOutcome(source.id, res.status, str(path), res.runtime, res.flagged, res.error.split("\n")[0])


def _init_worker() -> None:
    logging.getLogger().setLevel(logging.WARNING)


def run_fits(
    sources: Sequence[SpectrumSource],
    cfg: Config,
    out_dir: str | Path,
    *,
    jobs: int = 1,
    chunk: tuple[int, int] | None = None,
    resume: bool = False,
    opts: LoadOptions | None = None,
    progress: Callable[[TaskOutcome, int, int], None] | None = None,
) -> list[TaskOutcome]:
    out_dir = Path(out_dir)
    (out_dir / "spectra").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(cfg.dump())
    todo = select_chunk(sources, chunk)
    if resume:
        before = len(todo)
        todo = [s for s in todo if not result_path(out_dir, s.id).exists()]
        log.info("resume: %d of %d spectra already done", before - len(todo), before)
    cfg_text = cfg.model_dump_json()
    outcomes: list[TaskOutcome] = []
    total = len(todo)
    if jobs <= 1:
        for s in todo:
            o = fit_task(s, cfg_text, str(out_dir), opts)
            outcomes.append(o)
            if progress:
                progress(o, len(outcomes), total)
        return outcomes
    # Spawned workers import NumPy fresh, so the thread limits take effect and
    # the pool does not oversubscribe the machine.
    saved = {k: os.environ.get(k) for k in THREAD_VARS}
    os.environ.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    os.environ["XLA_FLAGS"] = (saved["XLA_FLAGS"] or "") + " --xla_cpu_multi_thread_eigen=false"
    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=_init_worker) as pool:
            futs = {pool.submit(fit_task, s, cfg_text, str(out_dir), opts): s for s in todo}
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    o = fut.result()
                except Exception as exc:  # noqa: BLE001 - e.g. a worker killed by the OOM killer
                    o = TaskOutcome(s.id, "failed", "", 0.0, True, f"worker crashed: {exc}")
                    log.error("%s: %s", s.id, o.error)
                outcomes.append(o)
                if progress:
                    progress(o, len(outcomes), total)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return outcomes


def summarize(outcomes: Sequence[TaskOutcome]) -> dict[str, Any]:
    rt = np.array([o.runtime for o in outcomes]) if outcomes else np.zeros(0)
    return {
        "n": len(outcomes),
        "ok": sum(o.status == "ok" for o in outcomes),
        "failed": sum(o.status != "ok" for o in outcomes),
        "flagged": sum(o.flagged for o in outcomes),
        "runtime_median": float(np.median(rt)) if rt.size else 0.0,
        "runtime_total": float(rt.sum()),
    }
