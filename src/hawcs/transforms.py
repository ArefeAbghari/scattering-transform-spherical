"""Spherical scattering transforms based on Healpy."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .utils import beam2bl


def _import_healpy():
    try:
        import healpy as hp
    except ImportError as exc:
        raise ImportError(
            "healpy is required for scattering transforms. "
            "Install the package with its runtime dependencies first."
        ) from exc
    return hp


def _default_lmax(nside: int, lmax: Optional[int]) -> int:
    return 3 * nside - 1 if lmax is None else lmax


def _as_harmonic_filter(
    filter_: np.ndarray,
    lmax: int,
    theta: Optional[np.ndarray] = None,
) -> np.ndarray:
    filter_arr = np.asarray(filter_)
    if filter_arr.shape == (lmax + 1,):
        return filter_arr

    if theta is None:
        theta = np.linspace(0, np.pi, filter_arr.size)

    return beam2bl(filter_arr, theta, lmax)


def compute_s1(
    hmap: np.ndarray,
    wavelet_filters: Sequence[np.ndarray],
    jmax: int,
    nside: int,
    gaus_l: Optional[Sequence[np.ndarray]] = None,
    lmax: Optional[int] = None,
    theta: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Compute first-order spherical scattering coefficients.

    ``wavelet_filters`` may contain harmonic filters with length ``lmax + 1`` or
    real-space filters sampled on ``theta``.
    """

    hp = _import_healpy()
    lmax = _default_lmax(nside, lmax)

    s1 = np.zeros(jmax)
    i1 = []
    mapalm = hp.map2alm(hmap, lmax=lmax, use_pixel_weights=True)

    for j, filter_ in enumerate(wavelet_filters[:jmax]):
        wavelet_l = _as_harmonic_filter(filter_, lmax, theta=theta)
        filtered_alm = hp.almxfl(mapalm, wavelet_l)
        modulus_map = np.abs(hp.alm2map(filtered_alm, nside, lmax=lmax))
        i1.append(modulus_map)

        if gaus_l is None:
            s1_map = modulus_map
        else:
            i1alm = hp.map2alm(modulus_map, lmax=lmax, use_pixel_weights=True)
            s1_map = hp.alm2map(hp.almxfl(i1alm, gaus_l[j]), nside, lmax=lmax)

        s1[j] = np.mean(s1_map)

    return s1, i1


def compute_s2(
    i1: Sequence[np.ndarray],
    wavelet_l: Sequence[np.ndarray],
    jmax: int,
    nside: int,
    gaus_l: Optional[Sequence[np.ndarray]] = None,
    lmax: Optional[int] = None,
) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
    """Compute second-order spherical scattering coefficients."""

    hp = _import_healpy()
    lmax = _default_lmax(nside, lmax)

    s2 = np.zeros((jmax, jmax))
    i2 = []

    for j1, modulus_map_1 in enumerate(i1[:jmax]):
        mapalm1 = hp.map2alm(modulus_map_1, lmax=lmax, use_pixel_weights=True)
        i2_row = []

        for j2, filter_l in enumerate(wavelet_l[:jmax]):
            filtered_alm = hp.almxfl(mapalm1, filter_l)
            modulus_map_2 = np.abs(hp.alm2map(filtered_alm, nside, lmax=lmax))
            i2_row.append(modulus_map_2)

            if gaus_l is None:
                s2_map = modulus_map_2
            else:
                i2alm = hp.map2alm(modulus_map_2, lmax=lmax, use_pixel_weights=True)
                s2_map = hp.alm2map(hp.almxfl(i2alm, gaus_l[j2]), nside, lmax=lmax)

            s2[j1, j2] = np.mean(s2_map)

        i2.append(i2_row)

    return s2, i2


def compute_s3(
    i2: Sequence[Sequence[np.ndarray]],
    wavelet_l: Sequence[np.ndarray],
    jmax: int,
    nside: int,
    gaus_l: Optional[Sequence[np.ndarray]] = None,
    lmax: Optional[int] = None,
) -> np.ndarray:
    """Compute third-order spherical scattering coefficients."""

    hp = _import_healpy()
    lmax = _default_lmax(nside, lmax)
    s3 = np.zeros((jmax, jmax, jmax))

    for j1 in range(jmax):
        for j2 in range(j1 + 1, jmax):
            mapalm2 = hp.map2alm(i2[j1][j2], lmax=lmax, use_pixel_weights=True)

            for j3, filter_l in enumerate(wavelet_l[:jmax]):
                filtered_alm = hp.almxfl(mapalm2, filter_l)
                modulus_map_3 = np.abs(hp.alm2map(filtered_alm, nside, lmax=lmax))

                if gaus_l is None:
                    s3_map = modulus_map_3
                else:
                    i3alm = hp.map2alm(modulus_map_3, lmax=lmax, use_pixel_weights=True)
                    s3_map = hp.alm2map(hp.almxfl(i3alm, gaus_l[j3]), nside, lmax=lmax)

                s3[j1, j2, j3] = np.mean(s3_map)

    return s3


__all__ = ["compute_s1", "compute_s2", "compute_s3"]
