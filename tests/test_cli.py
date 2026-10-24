import numpy as np

from hiprofile.cli import main
from hiprofile.io.catalog import read_catalog


def test_simulate_fit_report_plot(tmp_path):
    sims = tmp_path / "sims"
    out = tmp_path / "res"
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model:\n  n_max: 1\n  k_max: 0\nsampler:\n  mode: quick\noutput:\n  plots: all\n")
    simcfg = tmp_path / "s.yaml"
    simcfg.write_text("n_chan: 200\nsnr_range: [20, 40]\n")
    assert main(["simulate", "--n", "2", "--out", str(sims), "--config", str(simcfg)]) == 0
    assert main(["fit", str(sims / "spectra.txt"), "--config", str(cfg), "--out", str(out)]) == 0
    cat = read_catalog(out / "catalog.fits")
    assert len(cat) == 2 and np.all(cat["status"] == "ok")
    assert (out / "report.html").exists()
    assert len(list((out / "plots").glob("*_spectrum.png"))) == 2
    # resume skips everything, chunking selects a subset
    assert main(["fit", str(sims / "spectra.txt"), "--config", str(cfg), "--out", str(out), "--resume"]) == 0
    assert main(["report", str(out), "--format", "csv"]) == 0
    assert main(["plot", str(out), "sim000000"]) == 0


def test_example_configurations_load():
    from pathlib import Path

    import yaml

    from hiprofile.config import Config
    from hiprofile.simulate import SimConfig

    root = Path(__file__).resolve().parents[1] / "examples"
    Config.model_validate(yaml.safe_load((root / "survey.yaml").read_text()))
    SimConfig.model_validate(yaml.safe_load((root / "simulate.yaml").read_text()))
