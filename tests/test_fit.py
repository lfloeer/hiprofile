import numpy as np
import pytest

from hiprofile import Spectrum
from hiprofile.config import Config
from hiprofile.inference.fit import ModelEvidence, fit_spectrum, model_probabilities, spectrum_seed
from hiprofile.io.results import read_result, write_result

QUICK = Config().updated(sampler={"mode": "quick"}, model={"k_max": 1, "n_max": 1})


def test_quick_fit_recovers_line(line_spectrum, tmp_path):
    spec, truth = line_spectrum
    res = fit_spectrum(spec, QUICK)
    assert res.status == "ok", res.error
    assert res.p_detect > 0.999
    s = res.posteriors[1].summary()
    assert abs(s["v_ctr_1"] - truth[0]) < 5
    assert abs(s["delta_v_1"] - truth[1]) < 15
    assert abs(s["S_1"] - truth[4]) / truth[4] < 0.1
    assert s["W50_1_p16"] < s["W50_1"] < s["W50_1_p84"]
    assert res.model_band.shape == (5, spec.n_chan)
    # round trip
    p = write_result(res, tmp_path / "r.h5")
    back = read_result(p)
    assert back.p_detect == res.p_detect
    np.testing.assert_array_equal(back.posteriors[1].theta, res.posteriors[1].theta)
    assert back.catalog_row()["best_n"] == 1


def test_pure_noise_is_not_detected(rng):
    v = 1000.0 + 5.0 * np.arange(300)
    spec = Spectrum(v, 0.02 * rng.normal(size=v.size), sigma=np.full(v.size, 0.02), id="noise")
    res = fit_spectrum(spec, QUICK)
    assert res.status == "ok", res.error
    assert res.p_detect < 0.5
    assert res.best_n == 0


def test_fit_is_reproducible(line_spectrum):
    spec, _ = line_spectrum
    a = fit_spectrum(spec, QUICK)
    b = fit_spectrum(spec, QUICK)
    assert a.p_detect == b.p_detect
    np.testing.assert_array_equal(a.posteriors[1].theta, b.posteriors[1].theta)


def test_failure_is_reported():
    v = np.arange(20) * 5.0
    spec = Spectrum(v, np.full(v.size, np.nan), id="bad")
    res = fit_spectrum(spec, QUICK)
    assert res.status == "failed"
    assert res.flags["failed"]


def test_seed_depends_on_id_only():
    assert spectrum_seed(1, "a") == spectrum_seed(1, "a") != spectrum_seed(1, "b")


def test_model_probabilities():
    evs = [
        ModelEvidence(n=0, k=0, log_z=0.0, log_z_err=0, log_z_gauss=0, method="a"),
        ModelEvidence(n=1, k=0, log_z=10.0, log_z_err=0, log_z_gauss=0, method="a"),
    ]
    p = model_probabilities(evs, 0.5, 1)
    assert p[1] > 0.999 and abs(sum(p.values()) - 1) < 1e-12


def test_rfi_spikes_are_masked(line_spectrum):
    spec, _ = line_spectrum
    y = spec.flux.copy()
    y[[30, 31, 350]] += 0.4
    res = fit_spectrum(Spectrum(spec.velocity, y, sigma=spec.sigma, id="rfi"), QUICK)
    assert set(np.flatnonzero(res.mask)) >= {30, 31, 350}
    assert res.metrics["rfi_fraction"] < 0.03


@pytest.mark.slow
def test_full_mode(line_spectrum):
    spec, _truth = line_spectrum
    cfg = Config().updated(model={"k_max": 1, "n_max": 1})
    res = fit_spectrum(spec, cfg)
    assert res.status == "ok", res.error
    q = fit_spectrum(spec, QUICK)
    for a, b in zip(res.evidences, q.evidences, strict=True):
        assert abs(a.log_z - b.log_z) < 1.0
