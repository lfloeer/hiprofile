import numpy as np
import pytest

from hiprofile import Spectrum
from hiprofile.config import Config
from hiprofile.inference.fit import ModelEvidence, fit_spectrum, model_probabilities, spectrum_seed
from hiprofile.io.results import read_result, write_result
from hiprofile.model import SpectralGrid, profile

QUICK = Config().updated(sampler={"mode": "quick"}, model={"k_min": 0, "k_max": 1, "n_max": 1})


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
    assert 0.0 <= s["f_1"] <= 1.0
    post = res.posteriors[1]
    assert post.k_values == [0, 1] and post.k_probability is not None
    np.testing.assert_allclose(post.k_probability.sum(), 1.0)
    assert set(post.k_samples) <= {0, 1}
    assert {e.method for e in res.evidences if e.n == 1} == {"quick"}
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


def test_rfi_burst_is_fully_masked_and_not_fitted_as_a_line(grid, rng):
    """A three-channel burst must be masked completely, not fitted as a narrow second line."""
    truth = np.array([2000.0, 180.0, 12.0, 0.15, 4.0, 0.25])
    y = profile(grid, truth) + 0.02 * rng.normal(size=grid.n_chan)
    burst = [120, 121, 300, 301, 302, 380]
    y[burst] += np.array([0.3, 0.5, 0.4, 0.9, 0.4, 0.3])
    spec = Spectrum(grid.velocity, y, sigma=np.full(grid.n_chan, 0.02), id="burst")
    res = fit_spectrum(spec, QUICK.updated(model={"n_max": 2}))
    assert res.status == "ok", res.error
    assert set(np.flatnonzero(res.mask)) >= set(burst)
    assert res.best_n == 1
    assert not res.flags["unresolved"]
    s = res.posteriors[1].summary()
    half = 0.5 * (s["v_ctr_1_p84"] - s["v_ctr_1_p16"])  # ~1 sigma; this faint line has ~20 km/s
    assert abs(s["v_ctr_1"] - truth[0]) < 2.5 * half


def test_residual_peaks_find_narrow_and_broad_features(rng):
    from hiprofile.inference.fit import residual_peaks

    r = rng.normal(size=600)
    x = np.arange(600)
    r += 4.0 * np.exp(-0.5 * ((x - 100) / 2.0) ** 2)  # narrow, 4 sigma peak
    r += 1.5 * np.exp(-0.5 * ((x - 400) / 12.0) ** 2)  # broad, weak per channel
    good = np.ones(600, bool)
    good[250:260] = False
    peaks = residual_peaks(r, good, 3)
    chans = [c for c, _ in peaks]
    assert any(abs(c - 100) <= 2 for c in chans) and any(abs(c - 400) <= 6 for c in chans)
    widths = dict(peaks)
    assert widths[min(chans, key=lambda c: abs(c - 400))] > widths[min(chans, key=lambda c: abs(c - 100))]


def test_second_fainter_line_is_found(grid, rng):
    """A bright line plus a well separated line at S/N ~ 7: random starts miss it, residual seeds find it."""
    y = profile(grid, [1500.0, 150.0, 12.0, 0.1, 20.0, 0.25]) + profile(
        grid, [2500.0, 60.0, 10.0, 0.0, 3.0, 0.3]
    )
    y += 0.02 * rng.normal(size=grid.n_chan)
    y[[40, 41, 42, 350]] += [0.3, 0.6, 0.3, 0.4]  # a 3-channel RFI burst and a spike
    spec = Spectrum(grid.velocity, y, sigma=np.full(grid.n_chan, 0.02), id="pair")
    res = fit_spectrum(spec, QUICK.updated(model={"n_max": 2}))
    assert res.status == "ok", res.error
    masked = set(np.flatnonzero(res.mask))
    assert masked >= {40, 41, 42, 350}
    assert not masked & set(range(290, 310)), "the second line must not be masked as RFI"
    assert res.best_n == 2, res.p_n
    s = res.posteriors[2].summary()
    assert abs(s["v_ctr_1"] - 1500.0) < 10 and abs(s["v_ctr_2"] - 2500.0) < 20


def test_outlier_mask_keeps_long_runs():
    from hiprofile.likelihood.rfi import outlier_mask

    q = np.zeros(60)
    q[[5, 7]] = 0.9  # spike pair with a one-channel hole
    q[20:23] = 0.9  # three-channel burst
    q[30:34] = 0.9  # four channels: a line, not RFI
    q[[40, 41, 44, 45]] = 0.9  # six-channel run with a two-channel gap: a line
    bad = outlier_mask(q, 0.5)
    assert set(np.flatnonzero(bad)) == {5, 6, 7, 20, 21, 22}
    assert set(np.flatnonzero(outlier_mask(q, 0.5, max_width=None))) == {
        5,
        6,
        7,
        20,
        21,
        22,
        30,
        31,
        32,
        33,
        40,
        41,
        44,
        45,
    }


def test_estimated_noise_correlation_in_the_fit(grid, rng):
    """Noise smoothed by a kernel the fit does not know: noise.correlation = estimate measures it."""
    k = np.exp(-0.5 * (np.arange(-3, 4) / 0.8) ** 2)
    k /= k.sum()
    noise = np.convolve(rng.normal(size=grid.n_chan + 6), k, mode="valid") / np.sqrt(np.sum(k**2))
    y = profile(grid, [2000.0, 180.0, 12.0, 0.15, 10.0, 0.25]) + 0.02 * noise
    spec = Spectrum(grid.velocity, y, id="corr")
    for prefit in (True, False):
        res = fit_spectrum(spec, QUICK.updated(noise={"correlation": "estimate"}, rfi={"prefit": prefit}))
        assert res.status == "ok", res.error
        assert abs(res.metrics["noise_rho1"] - 0.672) < 0.1
        assert abs(res.metrics["sigma_rms"] - 0.02) < 0.003
        assert res.best_n == 1


def test_rfi_inflate_and_symmetric_background(line_spectrum):
    """The less common pre-stage options run and keep the detection."""
    spec, truth = line_spectrum
    y = spec.flux.copy()
    y[[30, 31, 350]] += 0.4
    bad = Spectrum(spec.velocity, y, sigma=spec.sigma, id="rfi2")
    for action, background in (("inflate", "positive"), ("mask", "symmetric")):
        res = fit_spectrum(bad, QUICK.updated(rfi={"action": action, "background": background}))
        assert res.status == "ok", res.error
        assert res.best_n == 1
        assert abs(res.posteriors[1].summary()["v_ctr_1"] - truth[0]) < 5
        if action == "inflate":
            assert not res.mask.any()
            assert res.sigma[31] > 3 * res.sigma[100]  # inflated, not masked
        else:
            assert set(np.flatnonzero(res.mask)) >= {30, 31, 350}


def test_signal_free_ranges_set_the_noise(line_spectrum):
    spec, _ = line_spectrum
    noisy = Spectrum(spec.velocity, spec.flux, id="nosigma")  # no sigma column: estimated
    free = fit_spectrum(noisy, QUICK.updated(noise={"signal_free": [[1000.0, 1700.0]]}))
    assert free.status == "ok", free.error
    assert abs(free.metrics["sigma_rms"] - 0.02) < 0.004


def test_gaussian_rfi_component(grid, rng):
    """model.n_rfi adds a narrow Gaussian that absorbs a spike instead of masking it."""
    y = profile(grid, [2000.0, 180.0, 12.0, 0.15, 20.0, 0.25]) + 0.02 * rng.normal(size=grid.n_chan)
    y[300] += 0.5
    spec = Spectrum(grid.velocity, y, sigma=np.full(grid.n_chan, 0.02), id="gaussrfi")
    # full mode: the quick stage's importance sampling degenerates on the extra
    # narrow component (ESS collapses, medians off by ~10 km/s), the SMC does not
    cfg = Config().updated(model={"k_min": 0, "k_max": 1, "n_max": 1, "n_rfi": 1}, rfi={"prefit": False})
    res = fit_spectrum(spec, cfg.updated(output={"plots": "none"}))
    assert res.status == "ok", res.error
    assert res.best_n == 1
    s = res.posteriors[1].summary()
    assert abs(s["rfi_v_1"] - grid.velocity[300]) < 10
    assert abs(s["v_ctr_1"] - 2000.0) < 5


def test_dwarf_survives_the_pre_stage_at_coarse_channels(rng):
    """At 15 km/s channels a 25 km/s line spans two channels: it must not be masked as an RFI burst."""
    v = 500.0 + 15.0 * np.arange(170)
    grid = SpectralGrid.from_velocities(v)
    y = profile(grid, [1600.0, 20.0, 8.0, 0.0, 2.5, 0.3]) + 0.01 * rng.normal(size=v.size)
    y[[20, 120]] += 0.15  # single-channel spikes
    spec = Spectrum(v, y, sigma=np.full(v.size, 0.01), id="coarse")
    res = fit_spectrum(spec, QUICK)
    assert res.status == "ok", res.error
    assert set(np.flatnonzero(res.mask)) == {20, 120}
    assert res.best_n == 1 and abs(res.posteriors[1].summary()["v_ctr_1"] - 1600.0) < 15
    assert res.flags["unresolved"]
