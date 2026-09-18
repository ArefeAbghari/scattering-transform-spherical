import sys
import types

import numpy as np

from hawcs import ScatteringSph


NSIDE = 16
NPIX = 12 * NSIDE**2
THETA_BIN = 1000


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

    scattering = ScatteringSph(nside=NSIDE, J=3, order=2, theta_bin=THETA_BIN)
    coefficients = scattering(np.ones(NPIX))

    assert scattering.resol == 0.01 * NSIDE
    assert set(coefficients) == {"S0", "S1", "S2"}
    assert set(coefficients["S0"]) == {()}
    assert set(coefficients["S1"]) == {(0,), (1,), (2,)}
    assert set(coefficients["S2"]) == {(0, 1), (0, 2), (1, 2)}
    assert set(scattering.intermediate_maps_) == {"I0", "I1", "I2"}
    assert scattering.get_i1(0).shape == (NPIX,)
    assert scattering.get_i2(1, 2).shape == (NPIX,)


def test_scattering_sph_coefficient_lookup(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    scattering = ScatteringSph(nside=NSIDE, J=3, order=2, theta_bin=THETA_BIN)
    coefficients = scattering(np.ones(NPIX))

    assert np.allclose(scattering.get_s1(0), coefficients["S1"][(0,)], equal_nan=True)
    assert np.allclose(scattering.get_s2(1, 2), coefficients["S2"][(1, 2)], equal_nan=True)
    assert np.allclose(
        scattering.coefficient(order=1, scales=(0,)),
        coefficients["S1"][(0,)],
        equal_nan=True,
    )


def test_scattering_sph_s3_uses_strictly_increasing_scales(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    scattering = ScatteringSph(nside=NSIDE, J=4, order=3, theta_bin=THETA_BIN)
    coefficients = scattering(np.ones(NPIX))

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
    assert scattering.get_i(order=3, scales=(0, 1, 2)).shape == (NPIX,)


def test_scattering_sph_filter_bank_matches_coefficient_format(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    scattering = ScatteringSph(nside=NSIDE, J=4, order=3, theta_bin=THETA_BIN)
    filters = scattering.filter_bank(space="harmonic")

    assert set(filters) == {"psi", "phi"}
    assert len(filters["psi"]) == 4
    assert len(filters["phi"]) == 4
    assert filters["psi"][0].shape == (3 * NSIDE,)
    assert filters["phi"][0].shape == (3 * NSIDE,)


def test_scattering_sph_filter_bank_can_return_real_filters(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    scattering = ScatteringSph(nside=NSIDE, J=2, order=2, theta_bin=THETA_BIN)
    filters = scattering.filter_bank(space="real")

    assert len(filters["psi"]) == 2
    assert len(filters["phi"]) == 2
    assert filters["psi"][0].shape == (THETA_BIN,)
    assert filters["phi"][1].shape == (THETA_BIN,)


def test_scattering_sph_passes_norm_to_filter_bank(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    normalized = ScatteringSph(nside=NSIDE, J=2, order=1, theta_bin=THETA_BIN, norm="L2")
    unnormalized = ScatteringSph(nside=NSIDE, J=2, order=1, theta_bin=THETA_BIN, norm=None)

    normalized_filter = normalized.filter_bank(space="real")["psi"][0]
    unnormalized_filter = unnormalized.filter_bank(space="real")["psi"][0]

    assert not np.allclose(normalized_filter, unnormalized_filter)


def test_scattering_sph_preserves_masked_maps(monkeypatch):
    fake_hp = _fake_healpy()
    seen = {}

    def map2alm(hmap, lmax=None, use_pixel_weights=True):
        seen["is_masked"] = np.ma.isMaskedArray(hmap)
        return np.full(lmax + 1, np.ma.mean(hmap), dtype=np.complex128)

    fake_hp.map2alm = map2alm
    monkeypatch.setitem(sys.modules, "healpy", fake_hp)

    hmap = np.ma.array(np.ones(NPIX), mask=np.zeros(NPIX, dtype=bool))
    hmap.mask[0] = True

    scattering = ScatteringSph(nside=NSIDE, J=1, order=1, theta_bin=THETA_BIN)
    scattering(hmap)

    assert seen["is_masked"] is True


def test_scattering_sph_averages_coefficients_over_unmasked_pixels(monkeypatch):
    fake_hp = _fake_healpy()

    def map2alm(hmap, lmax=None, use_pixel_weights=True):
        return np.ones(lmax + 1, dtype=np.complex128)

    def alm2map(alm, nside, lmax=None):
        return np.arange(12 * nside**2, dtype=float)

    fake_hp.map2alm = map2alm
    fake_hp.alm2map = alm2map
    monkeypatch.setitem(sys.modules, "healpy", fake_hp)

    hmap = np.ma.array(np.arange(NPIX, dtype=float), mask=np.zeros(NPIX, dtype=bool))
    hmap[0] = 1e9
    hmap.mask[0] = True

    scattering = ScatteringSph(nside=NSIDE, J=1, order=1, theta_bin=THETA_BIN)
    coefficients = scattering(hmap)

    expected_mean = np.mean(np.arange(1, NPIX, dtype=float))

    assert coefficients["S0"][()] == expected_mean
    assert coefficients["S1"][(0,)] == expected_mean


def test_scattering_sph_uses_mask_as_coefficient_weights(monkeypatch):
    fake_hp = _fake_healpy()

    def map2alm(hmap, lmax=None, use_pixel_weights=True):
        return np.ones(lmax + 1, dtype=np.complex128)

    def alm2map(alm, nside, lmax=None):
        return np.arange(12 * nside**2, dtype=float)

    fake_hp.map2alm = map2alm
    fake_hp.alm2map = alm2map
    monkeypatch.setitem(sys.modules, "healpy", fake_hp)

    hmap = np.arange(NPIX, dtype=float)
    mask = np.ones(NPIX, dtype=float)
    mask[0] = 0.0
    mask[1] = 0.5

    scattering = ScatteringSph(nside=NSIDE, J=1, order=1, theta_bin=THETA_BIN)
    coefficients = scattering(hmap, mask=mask)

    expected_mean = np.sum(np.arange(NPIX, dtype=float) * mask) / np.sum(mask)

    assert coefficients["S0"][()] == expected_mean
    assert coefficients["S1"][(0,)] == expected_mean


def test_scattering_sph_combines_explicit_mask_with_masked_map(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    hmap = np.ma.array(np.arange(NPIX, dtype=float), mask=np.zeros(NPIX, dtype=bool))
    hmap[0] = 1e9
    hmap.mask[0] = True
    mask = np.ones(NPIX, dtype=float)
    mask[1] = 0.5

    scattering = ScatteringSph(nside=NSIDE, J=1, order=0, theta_bin=THETA_BIN)
    coefficients = scattering(hmap, mask=mask)

    combined_weights = mask.copy()
    combined_weights[0] = 0.0
    expected_mean = np.sum(np.ma.filled(hmap, 0) * combined_weights) / np.sum(
        combined_weights
    )

    assert coefficients["S0"][()] == expected_mean


def test_scattering_sph_nan_mask_falls_back_to_masked_map(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    hmap = np.ma.array(np.arange(NPIX, dtype=float), mask=np.zeros(NPIX, dtype=bool))
    hmap[0] = 1e9
    hmap.mask[0] = True

    scattering = ScatteringSph(nside=NSIDE, J=1, order=0, theta_bin=THETA_BIN)
    coefficients = scattering(hmap, mask=np.nan)

    assert coefficients["S0"][()] == np.mean(np.arange(1, NPIX, dtype=float))


def test_scattering_sph_validates_coefficient_mask(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    scattering = ScatteringSph(nside=NSIDE, J=1, order=0, theta_bin=THETA_BIN)
    hmap = np.ones(NPIX)

    invalid_masks = [
        np.ones(NPIX - 1),
        np.full(NPIX, -0.1),
        np.full(NPIX, 1.1),
        np.full(NPIX, np.nan),
        np.zeros(NPIX),
    ]

    for mask in invalid_masks:
        try:
            scattering(hmap, mask=mask)
        except ValueError:
            pass
        else:
            raise AssertionError("ScatteringSph should reject invalid masks")


def test_scattering_sph_validates_map_size(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    scattering = ScatteringSph(nside=NSIDE, J=1, order=1, theta_bin=THETA_BIN)

    try:
        scattering.compute_coefficients(np.ones(NPIX - 1))
    except ValueError as exc:
        assert f"expected {NPIX}" in str(exc)
    else:
        raise AssertionError("ScatteringSph should reject maps with wrong npix")


def test_scattering_sph_can_skip_intermediate_storage(monkeypatch):
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    scattering = ScatteringSph(nside=NSIDE, J=1, order=1, theta_bin=THETA_BIN)
    scattering(np.ones(NPIX), keep_maps=False)

    assert scattering.intermediate_maps_ == {}
    try:
        scattering.get_i1(0)
    except RuntimeError as exc:
        assert "keep_maps=True" in str(exc)
    else:
        raise AssertionError("get_i1 should require intermediate maps")
