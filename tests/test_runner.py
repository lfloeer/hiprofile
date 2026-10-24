import numpy as np

from hiprofile.config import Config
from hiprofile.io.results import read_result, result_path
from hiprofile.io.spectra import SpectrumSource, write_spectrum
from hiprofile.runner import run_fits, select_chunk
from hiprofile.simulate import SimConfig, simulate_many

CFG = Config().updated(sampler={"mode": "quick"}, model={"n_max": 1, "k_max": 0}, output={"plots": "none"})


def _sources(tmp_path, n=3):
    sims = simulate_many(SimConfig(n_chan=160, snr_range=(10, 30)), n, seed=5)
    out = []
    for spec, _ in sims:
        p = tmp_path / f"{spec.id}.h5"
        write_spectrum(spec, p)
        out.append(SpectrumSource(str(p), spec.id))
    return out


def _same(a, b):
    ra, rb = read_result(a), read_result(b)
    assert ra.p_detect == rb.p_detect
    np.testing.assert_array_equal(ra.posteriors[1].theta, rb.posteriors[1].theta)
    np.testing.assert_array_equal(ra.posteriors[1].weights, rb.posteriors[1].weights)


def test_chunks_and_resume_give_identical_results(tmp_path):
    src = _sources(tmp_path)
    run_fits(src, CFG, tmp_path / "full")
    for i in range(2):
        run_fits(src, CFG, tmp_path / "chunked", chunk=(i, 2))
    # simulate an interrupted run: one result missing, then resume
    run_fits(src, CFG, tmp_path / "resumed")
    result_path(tmp_path / "resumed", src[1].id).unlink()
    outcomes = run_fits(src, CFG, tmp_path / "resumed", resume=True)
    assert [o.id for o in outcomes] == [src[1].id]
    for s in src:
        _same(result_path(tmp_path / "full", s.id), result_path(tmp_path / "chunked", s.id))
        _same(result_path(tmp_path / "full", s.id), result_path(tmp_path / "resumed", s.id))


def test_process_pool(tmp_path):
    src = _sources(tmp_path, 2)
    out = run_fits(src, CFG, tmp_path / "pool", jobs=2)
    assert sorted(o.status for o in out) == ["ok", "ok"]
    _ = [read_result(result_path(tmp_path / "pool", s.id)) for s in src]


def test_failed_spectrum_does_not_stop_the_run(tmp_path):
    src = [*_sources(tmp_path, 1), SpectrumSource(str(tmp_path / "missing.fits"), "missing")]
    out = run_fits(src, CFG, tmp_path / "f")
    assert {o.id: o.status for o in out} == {src[0].id: "ok", "missing": "failed"}
    assert read_result(result_path(tmp_path / "f", "missing")).flags["failed"]


def test_select_chunk():
    assert select_chunk(list(range(7)), (1, 3)) == [1, 4]


def test_validate_stage_in_chunks(tmp_path):
    from hiprofile.validate import run_stage

    cfg = Config().updated(sampler={"mode": "quick"}, output={"plots": "none"})
    for i in range(2):
        part = run_stage("sbc", tmp_path, 2, cfg, seed=3, chunk=(i, 2))
        assert part["runs"] == {} and part["chunk"] == [i, 2]
    assert not (tmp_path / "validate_sbc" / "summary.json").exists()
    full = run_stage("sbc", tmp_path, 2, cfg, seed=3, resume=True)
    assert full["runs"]["sbc"]["n_fitted"] == 2
    assert (tmp_path / "validate_sbc" / "summary.md").exists()
