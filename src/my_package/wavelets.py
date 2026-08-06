"""Wavelet and smoothing filters on the sphere."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from .utils import beam2bl, integrate

FilterBankDict = Dict[str, List[np.ndarray]]


def _import_healpy():
    try:
        import healpy as hp
    except ImportError as exc:
        raise ImportError(
            "healpy is required to build filters from nside. "
            "Install the package with its runtime dependencies first."
        ) from exc
    return hp


def gabor(freq: float, sigma: float, theta: np.ndarray) -> np.ndarray:
    """Generate a complex Gabor profile in real space."""

    theta = np.asarray(theta, dtype=float)
    arg = -(theta**2) / (2 * sigma**2) + 1j * freq * theta
    return np.exp(arg) #/ (2 * np.pi * sigma**2)


def morlet(freq: float, sigma: float, theta: np.ndarray) -> np.ndarray:
    """Generate a zero-mean Morlet wavelet profile in real space."""

    wavelet = gabor(freq, sigma, theta)
    low_pass = gabor(0.0, sigma, theta)
    correction = integrate(wavelet, theta) / integrate(low_pass, theta)
    return wavelet - correction * low_pass

def gaussian(
    sigma: float, theta: np.ndarray) -> np.ndarray:
    """Generate a real-space Gaussian smoothing profile."""

    arg = -(theta**2) / (2 * sigma**2)

    return np.exp(arg)


def theta_grid(theta_bin: int) -> np.ndarray:
    """Return the angular grid used by real-space filters."""

    if theta_bin < 2:
        raise ValueError("theta_bin must be at least 2")
    return np.linspace(0, np.pi, theta_bin)


def scale(resol: float, j: int) -> float:
    """Return the dyadic angular scale for index ``j``."""

    if resol <= 0:
        raise ValueError("resol must be positive")
    if j < 0:
        raise ValueError("j must be non-negative")
    return resol * 2**j


def sigma(resol: float, j: int) -> float:
    """Return the Gaussian envelope width for index ``j``."""

    return 0.8 * scale(resol, j)


def frequency(resol: float, j: int) -> float:
    """Return the Morlet oscillation frequency for index ``j``."""

    return 3.0 * np.pi / (4.0 * scale(resol, j))


def filter_bank_real (
    nside: int,
    jmax: int,
    theta_bin: int = 1000,
) :
    """Build real-space Morlet filters for dyadic scales."""

    hp = _import_healpy()
    _validate_jmax(jmax)
    _validate_nside(nside, hp)
    resol = hp.nside2resol(nside, arcmin=False)
    theta = theta_grid(theta_bin)
    
    filters = {}
    #filters["psi"] = []
    #filters["phi"] = []

    filters["psi"] = [
        morlet(frequency(resol, j), sigma(resol, j), theta)
        for j in range(jmax)
    ]
    filters["phi"] = [ gaussian(sigma(resol, j), theta)
        for j in range(jmax)]
    return  filters

def filter_bank_harmonic (
    nside: int,
    jmax: int,
    lmax: int = None,
    theta_bin: int = 1000,
) :
    """Build harmonic-space Morlet filters for dyadic scales."""

    hp = _import_healpy()
    _validate_jmax(jmax)
    _validate_nside(nside, hp)
    if lmax == None: 
        lmax = 3 * nside -1 
    _validate_lmax(lmax)
    resol = hp.nside2resol(nside, arcmin=False)
    theta = theta_grid(theta_bin)
    
    filters = {}
    #filters["psi"] = []
    #filters["phi"] = []

    filters["psi"] = [
        beam2bl(morlet(frequency(resol, j), sigma(resol, j), theta), theta, lmax)
        for j in range(jmax)
    ]
    filters["phi"] = [hp.gauss_beam(2 * np.sqrt(2 * np.log(2)) * sigma(resol, j), lmax=lmax)
        for j in range(jmax)]
    return  filters

def _validate_jmax(jmax: int) -> None:
    if jmax < 1:
        raise ValueError("jmax must be at least 1")


def _validate_lmax(lmax: int) -> None:
    if lmax < 0:
        raise ValueError("lmax must be non-negative")


def _validate_nside(nside: int, hp) -> None: 
    if nside < 1:
        raise ValueError("nside must be positive")
    if hasattr(hp, "isnsideok") and not hp.isnsideok(nside):
        raise ValueError("nside must be a valid Healpy nside")

__all__ = [
    "FilterBankDict",
    "frequency",
    "gabor",
    "morlet",
    "scale",
    "sigma",
    "theta_grid",
    "filter_bank_real",
    "filter_bank_harmonic",
]
