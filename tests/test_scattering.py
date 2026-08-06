import sys
import types

import numpy as np

from my_package import ScatteringSph


def _fake_healpy():
    hp = types.SimpleNamespace()
    hp.isnsideok = lambda nside: nside > 0 and nside & (nside - 1) == 0
    hp.nside2resol = lambda nside, arcmin=False: 0.01 * nside
    hp.nside2npix = lambda nside: 12 * nside**2

    def map2alm(hmap, lmax=None, use_pixel_weights=True):
        return np.full(lmax + 1, np.mean(hmap), dtype=np.complex128)

    def almxfl(alm, filter_l):
        return alm * filter_l

    def alm2map(alm, nside, lmax=None):
        return np.full(12 * nside**2, np.real(np.sum(alm)))

    def gauss_beam(fwhm, lmax=None):
        return np.full(lmax + 1, fwhm)

    hp.map2alm = map2alm
    hp.almxfl = almxfl
    hp.alm2map = alm2map
    hp.gauss_beam = gauss_beam
    return hp


def test_scattering_sph_uses_nside_and_returns_dictionary(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    scattering = ScatteringSph(nside=1, J=3, order=2, theta_bin=16)
    coefficients = scattering(np.ones(12))

    assert scattering.resol == 0.01
    assert set(coefficients) == {"S0", "S1", "S2"}
    assert set(coefficients["S0"]) == {()}
    assert set(coefficients["S1"]) == {(0,), (1,), (2,)}
    assert set(coefficients["S2"]) == {(0, 1), (0, 2), (1, 2)}
    assert set(scattering.intermediate_maps_) == {"I0", "I1", "I2"}
    assert scattering.get_i1(0).shape == (12,)
    assert scattering.get_i2(1, 2).shape == (12,)


def test_scattering_sph_coefficient_lookup(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    scattering = ScatteringSph(nside=1, J=3, order=2, theta_bin=16)
    coefficients = scattering(np.ones(12))

    assert scattering.get_s1(0) == coefficients["S1"][(0,)]
    assert scattering.get_s2(1, 2) == coefficients["S2"][(1, 2)]
    assert scattering.coefficient(order=1, scales=(0,)) == coefficients["S1"][(0,)]


def test_scattering_sph_s3_uses_strictly_increasing_scales(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    scattering = ScatteringSph(nside=1, J=4, order=3, theta_bin=16)
    coefficients = scattering(np.ones(12))

    assert set(coefficients["S2"]) == {
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    }
    assert set(coefficients["S3"]) == {
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    }
    assert all(path[0] < path[1] < path[2] for path in coefficients["S3"])
    assert scattering.get_i(order=3, scales=(0, 1, 2)).shape == (12,)


def test_scattering_sph_filter_bank_matches_coefficient_format(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    scattering = ScatteringSph(nside=1, J=4, order=3, theta_bin=16)
    filters = scattering.filter_bank(space="harmonic")

    assert set(filters) == {"psi", "phi"}
    assert len(filters["psi"]) == 4
    assert len(filters["phi"]) == 4
    assert filters["psi"][0].shape == (3,)
    assert filters["phi"][0].shape == (3,)


def test_scattering_sph_filter_bank_can_return_real_filters(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    scattering = ScatteringSph(nside=1, J=2, order=2, theta_bin=16)
    filters = scattering.filter_bank(space="real")

    assert len(filters["psi"]) == 2
    assert len(filters["phi"]) == 2
    assert filters["psi"][0].shape == (16,)
    assert filters["phi"][1].shape == (16,)


def test_scattering_sph_validates_map_size(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    scattering = ScatteringSph(nside=1, J=1, order=1, theta_bin=16)

    try:
        scattering.compute_coefficients(np.ones(11))
    except ValueError as exc:
        assert "expected 12" in str(exc)
    else:
        raise AssertionError("ScatteringSph should reject maps with wrong npix")


def test_scattering_sph_can_skip_intermediate_storage(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    scattering = ScatteringSph(nside=1, J=1, order=1, theta_bin=16)
    scattering(np.ones(12), keep_maps=False)

    assert scattering.intermediate_maps_ == {}
    try:
        scattering.get_i1(0)
    except RuntimeError as exc:
        assert "keep_maps=True" in str(exc)
    else:
        raise AssertionError("get_i1 should require intermediate maps")
