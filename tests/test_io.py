import numpy as np
import pytest

from hiprofile import Spectrum
from hiprofile.io.spectra import HI_REST_FREQUENCY_HZ, list_sources, load_spectrum, write_spectrum


@pytest.mark.parametrize("suffix", [".fits", ".h5", ".txt"])
def test_round_trip(tmp_path, rng, suffix):
    v = 1000 + 5.0 * np.arange(50)
    s = Spectrum(v, rng.normal(size=50), sigma=np.full(50, 0.1), mask=np.arange(50) == 3, id="x")
    p = tmp_path / f"x{suffix}"
    write_spectrum(s, p)
    back = load_spectrum(p)
    np.testing.assert_allclose(back.velocity, v)
    np.testing.assert_allclose(back.flux, s.flux)
    np.testing.assert_allclose(back.sigma, s.sigma)
    assert back.mask[3] and back.mask.sum() == 1


def test_frequency_axis(tmp_path):
    from astropy.table import Table
    import astropy.units as u

    f = HI_REST_FREQUENCY_HZ - np.arange(64) * 24e3
    t = Table({"FREQ": f * u.Hz, "FLUX": np.zeros(64)})
    t.write(tmp_path / "f.fits")
    s = load_spectrum(tmp_path / "f.fits").normalized()
    assert abs(s.velocity[0]) < 1e-6
    assert abs(s.channel_width - 5.066) < 0.01  # 24 kHz at 1420 MHz, radio convention


def test_list_sources(tmp_path, rng):
    for i in range(3):
        write_spectrum(Spectrum(np.arange(20.0), rng.normal(size=20), id=f"s{i}"), tmp_path / f"s{i}.fits")
    (tmp_path / "list.txt").write_text("s0.fits\ns1.fits a\n# comment\ns2.fits\n")
    src = list_sources(str(tmp_path / "list.txt"))
    assert [s.id for s in src] == ["s0", "a", "s2"]
    assert len(list_sources(str(tmp_path / "*.fits"))) == 3


def test_descending_axis_is_normalized(rng):
    v = np.arange(30.0)[::-1] * 5
    s = Spectrum(v, np.arange(30.0)).normalized()
    assert np.all(np.diff(s.velocity) > 0) and s.flux[0] == 29.0
