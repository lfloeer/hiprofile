"""Command line interface: ``hiprofile fit | simulate | validate | report | plot``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from hiprofile import __version__


def _setup_logging(out_dir: Path | None, verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(out_dir / "hiprofile.log"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    for noisy in ("matplotlib", "jax", "h5py", "ultranest", "nautilus"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _load_config(args: argparse.Namespace):
    from hiprofile.config import Config

    cfg = Config.load(getattr(args, "config", None))
    upd: dict[str, Any] = {}
    if getattr(args, "mode", None):
        upd.setdefault("sampler", {})["mode"] = args.mode
    if getattr(args, "backend", None):
        upd.setdefault("sampler", {})["backend"] = args.backend
    if getattr(args, "gpu", False):
        upd.setdefault("sampler", {})["likelihood_backend"] = "jax"
    if getattr(args, "plots", None):
        upd.setdefault("output", {})["plots"] = args.plots
    if getattr(args, "seed", None) is not None:
        upd["seed"] = args.seed
    return cfg.updated(**upd) if upd else cfg


def _load_options(args: argparse.Namespace):
    from hiprofile.io.spectra import LoadOptions

    return LoadOptions(
        velocity=args.velocity_column,
        frequency=args.frequency_column,
        flux=args.flux_column,
        sigma=args.sigma_column,
        mask=args.mask_column,
        velocity_unit=args.velocity_unit,
        frequency_unit=args.frequency_unit,
        convention=args.convention,
    )


def cmd_fit(args: argparse.Namespace) -> int:
    from hiprofile.io.spectra import list_sources
    from hiprofile.plots.report import report_from_dir
    from hiprofile.runner import parse_chunk, run_fits, summarize

    out = Path(args.out)
    _setup_logging(out, args.verbose)
    log = logging.getLogger("hiprofile")
    cfg = _load_config(args)
    sources = list_sources(args.spectra)
    if len(sources) > 1 and cfg.priors.amplitude_scale == "auto":
        log.warning(
            "priors.amplitude_scale is 'auto' (10 x the noise of the band-integrated flux per spectrum); "
            "fix it for comparable evidences and detection probabilities across the survey"
        )
    chunk = parse_chunk(args.chunk)
    log.info(
        "hiprofile %s: %d spectra, mode %s, backend %s, %d jobs%s",
        __version__,
        len(sources),
        cfg.sampler.mode,
        cfg.sampler.backend,
        args.jobs,
        f", chunk {args.chunk}" if chunk else "",
    )
    t0 = time.time()

    def progress(o, i, n):
        msg = f"[{i}/{n}] {o.id}: {o.status} ({o.runtime:.1f}s){' flagged' if o.flagged else ''}"
        if o.status != "ok":
            log.warning("%s %s", msg, o.error)
        else:
            log.info(msg)

    outcomes = run_fits(
        sources,
        cfg,
        out,
        jobs=args.jobs,
        chunk=chunk,
        resume=args.resume,
        opts=_load_options(args),
        progress=progress,
    )
    summ = summarize(outcomes)
    summ["wall_time"] = time.time() - t0
    log.info("done: %s", json.dumps(summ))
    if chunk is None and not args.no_report:
        info = report_from_dir(out, cfg.output.catalog_format)
        log.info("catalogue %s, report %s", info["catalog"], info["report"])
    return 0 if summ["failed"] == 0 else 2


def cmd_simulate(args: argparse.Namespace) -> int:
    import yaml

    from hiprofile.io.catalog import rows_to_table
    from hiprofile.io.spectra import write_spectrum
    from hiprofile.simulate import SimConfig, simulate_many

    out = Path(args.out)
    (out / "spectra").mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    cfg = SimConfig.model_validate(data or {})
    sims = simulate_many(cfg, args.n, args.seed, prefix=args.prefix)
    lines, truth = [], []
    for spec, tr in sims:
        p = out / "spectra" / f"{spec.id}.{args.format}"
        write_spectrum(spec, p)
        lines.append(f"{p.relative_to(out)} {spec.id}")
        truth.append(tr.row())
    (out / "spectra.txt").write_text("\n".join(lines) + "\n")
    rows_to_table(truth).write(out / "truth.csv", format="ascii.csv", overwrite=True)
    (out / "simulate.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    print(f"wrote {len(sims)} spectra to {out}/spectra, list {out}/spectra.txt, truth {out}/truth.csv")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from hiprofile.runner import parse_chunk
    from hiprofile.validate import run_stage

    out = Path(args.out)
    _setup_logging(out, args.verbose)
    cfg = _load_config(args)
    stages = args.stage if args.stage else ["sbc"]
    chunk = parse_chunk(args.chunk)
    for st in stages:
        run_stage(
            st,
            out,
            args.n,
            cfg,
            jobs=args.jobs,
            seed=args.seed or 1,
            spectra=args.spectra,
            resume=args.resume,
            chunk=chunk,
        )
        if chunk is None:
            print((out / f"validate_{st}" / "summary.md").read_text())
        else:
            print(f"{st}: chunk {args.chunk} fitted; analyse with the same command, --resume and no --chunk")
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    from collections import defaultdict

    from hiprofile.inference.batch import screen_batch
    from hiprofile.io.catalog import rows_to_table
    from hiprofile.io.spectra import list_sources

    _setup_logging(None, args.verbose)
    cfg = _load_config(args)
    spectra = [s.load() for s in list_sources(args.spectra)]
    groups: dict[tuple[int, float], list] = defaultdict(list)
    for sp in spectra:
        n = sp.normalized()
        groups[(n.n_chan, round(n.channel_width, 9))].append(sp)
    rows = []
    for key, members in groups.items():
        for i in range(0, len(members), args.batch):
            chunk = members[i : i + args.batch]
            logging.getLogger("hiprofile").info("screening %d spectra with %d channels", len(chunk), key[0])
            rows += screen_batch(chunk, cfg, k=args.k, n_particles=args.particles, seed=cfg.seed + i)
    rows_to_table(rows).write(args.out, overwrite=True)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from hiprofile.plots.report import report_from_dir

    info = report_from_dir(args.out, args.format)
    print(f"catalogue: {info['catalog']} ({info['n']} rows)\nreport: {info['report']}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    from hiprofile.io.results import read_result
    from hiprofile.plots.spectrum import plot_all

    out = Path(args.out)
    files = sorted((out / "spectra").glob("*.h5"))
    if args.ids:
        wanted = set(args.ids)
        files = [f for f in files if f.stem in wanted]
    n = 0
    for f in files:
        res = read_result(f)
        if res.status != "ok":
            continue
        if args.flagged and not res.flagged:
            continue
        plot_all(res, out / "plots")
        n += 1
    print(f"plotted {n} spectra into {out / 'plots'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hiprofile", description="Bayesian HI profile fitting")
    ap.add_argument("--version", action="version", version=f"hiprofile {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fit", help="fit a list of spectra")
    f.add_argument("spectra", help="list file, glob pattern, catalogue table or a single spectrum")
    f.add_argument("--config", help="YAML configuration")
    f.add_argument("--out", default="results")
    f.add_argument("--jobs", type=int, default=1)
    f.add_argument("--chunk", help="i/n: process chunk i of n (SLURM array jobs)")
    f.add_argument("--resume", action="store_true", help="skip spectra with an existing result")
    f.add_argument("--mode", choices=["quick", "full", "check"])
    f.add_argument("--backend", choices=["nautilus", "dynesty", "ultranest", "smc", "quick"])
    f.add_argument("--gpu", action="store_true", help="use the JAX likelihood (CPU or GPU)")
    f.add_argument("--plots", choices=["none", "flagged", "sample", "all"])
    f.add_argument("--seed", type=int)
    f.add_argument("--no-report", action="store_true")
    f.add_argument("-v", "--verbose", action="store_true")
    g = f.add_argument_group("input columns")
    g.add_argument("--velocity-column")
    g.add_argument("--frequency-column")
    g.add_argument("--flux-column")
    g.add_argument("--sigma-column")
    g.add_argument("--mask-column")
    g.add_argument("--velocity-unit", default="km/s")
    g.add_argument("--frequency-unit", default="Hz")
    g.add_argument("--convention", default="radio", choices=["radio", "optical", "relativistic"])
    f.set_defaults(func=cmd_fit)

    s = sub.add_parser("simulate", help="generate test spectra with known truth")
    s.add_argument("--config", help="YAML simulation configuration (SimConfig)")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--out", default="sims")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--prefix", default="sim")
    s.add_argument("--format", default="fits", choices=["fits", "h5", "txt"])
    s.set_defaults(func=cmd_simulate)

    v = sub.add_parser("validate", help="run validation stages")
    v.add_argument("--stage", action="append", choices=["sbc", "detection", "misspec", "injection"])
    v.add_argument("--n", type=int, default=200)
    v.add_argument("--config", help="fit configuration (YAML)")
    v.add_argument("--out", default="validation")
    v.add_argument("--jobs", type=int, default=1)
    v.add_argument("--mode", choices=["quick", "full", "check"])
    v.add_argument("--backend", choices=["nautilus", "dynesty", "ultranest", "smc", "quick"])
    v.add_argument("--spectra", help="signal-free survey spectra for the injection stage")
    v.add_argument("--seed", type=int)
    v.add_argument("--resume", action="store_true")
    v.add_argument("--chunk", help="i/n: fit chunk i of n (SLURM arrays), then analyse with --resume")
    v.add_argument("-v", "--verbose", action="store_true")
    v.set_defaults(func=cmd_validate)

    c = sub.add_parser("screen", help="GPU/JAX batch screening: evidences of N = 0 and N = 1 at fixed K")
    c.add_argument("spectra")
    c.add_argument("--config")
    c.add_argument("--out", default="screen.fits")
    c.add_argument("--k", type=int, default=1, help="baseline order")
    c.add_argument("--batch", type=int, default=256, help="spectra per XLA batch")
    c.add_argument("--particles", type=int, default=1000)
    c.add_argument("-v", "--verbose", action="store_true")
    c.set_defaults(func=cmd_screen)

    r = sub.add_parser("report", help="build catalogue and HTML report from result files")
    r.add_argument("out", help="results directory")
    r.add_argument("--format", default="fits", choices=["fits", "parquet", "csv"])
    r.set_defaults(func=cmd_report)

    p = sub.add_parser("plot", help="plots for single or flagged spectra from stored results")
    p.add_argument("out", help="results directory")
    p.add_argument("ids", nargs="*", help="spectrum ids (default: all)")
    p.add_argument("--flagged", action="store_true")
    p.set_defaults(func=cmd_plot)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
