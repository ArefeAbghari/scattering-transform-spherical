import sys
import types

import numpy as np

from my_package import filter_bank_harmonic, filter_bank_real


def _fake_healpy():
    hp = types.SimpleNamespace()
    hp.isnsideok = lambda nside: nside > 0 and nside & (nside - 1) == 0
    hp.nside2resol = lambda nside, arcmin=False: 0.01 * nside
    hp.gauss_beam = lambda fwhm, lmax=None: np.full(lmax + 1, fwhm)
    return hp


def test_filter_bank_real_returns_kymatio_like_dictionary(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    filters = filter_bank_real(nside=1, jmax=3, theta_bin=64)

    assert set(filters) == {"psi", "phi"}
    assert len(filters["psi"]) == 3
    assert len(filters["phi"]) == 3
    assert all(filter_.shape == (64,) for filter_ in filters["psi"])
    assert all(filter_.shape == (64,) for filter_ in filters["phi"])


def test_filter_bank_harmonic_returns_kymatio_like_dictionary(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    filters = filter_bank_harmonic(nside=1, jmax=2, lmax=5, theta_bin=32)

    assert set(filters) == {"psi", "phi"}
    assert len(filters["psi"]) == 2
    assert len(filters["phi"]) == 2
    assert all(filter_.shape == (6,) for filter_ in filters["psi"])
    assert all(filter_.shape == (6,) for filter_ in filters["phi"])


def test_filter_bank_harmonic_defaults_lmax(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    filters = filter_bank_harmonic(nside=2, jmax=1, theta_bin=16)

    assert filters["psi"][0].shape == (6,)
    assert filters["phi"][0].shape == (6,)


def test_filter_bank_validates_nside(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    try:
        filter_bank_real(nside=3, jmax=1)
    except ValueError as exc:
        assert "valid Healpy nside" in str(exc)
    else:
        raise AssertionError("filter_bank_real should reject invalid nside")
