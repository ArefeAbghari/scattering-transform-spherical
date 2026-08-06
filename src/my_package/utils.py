"""Numerical helpers for spherical filters."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from scipy.integrate import trapezoid
from scipy.integrate import simpson

ArrayLike = Union[np.ndarray, Sequence[float]]


def _validate_same_shape(beam: np.ndarray, theta: np.ndarray) -> None:
    if beam.shape != theta.shape:
        raise ValueError("beam and theta must have the same shape")


def beam2bl(beam: ArrayLike, theta: ArrayLike, lmax: int) -> np.ndarray:
    """Convert an axisymmetric beam profile ``b(theta)`` to ``b_l``.

    Parameters
    ----------
    beam
        Circular beam profile sampled at the angles in ``theta``.
    theta
        Angular radii in radians. Must have the same shape as ``beam``.
    lmax
        Maximum multipole to compute.
    """

    if lmax < 0:
        raise ValueError("lmax must be non-negative")

    beam_arr = np.asarray(beam)
    theta_arr = np.asarray(theta, dtype=float)
    _validate_same_shape(beam_arr, theta_arr)

    x = np.cos(theta_arr)
    sin_theta = np.sin(theta_arr)
    window = np.zeros(lmax + 1, dtype=np.result_type(beam_arr, np.complex128))

    p_l_minus_2 = np.ones_like(theta_arr)
    window[0] = trapezoid(beam_arr * p_l_minus_2 * sin_theta, theta_arr)

    if lmax >= 1:
        p_l_minus_1 = x.copy()
        window[1] = trapezoid(beam_arr * p_l_minus_1 * sin_theta, theta_arr)

        for ell in range(2, lmax + 1):
            p_l = ((2 * ell - 1) * x * p_l_minus_1 - (ell - 1) * p_l_minus_2) / ell
            window[ell] = trapezoid(beam_arr * p_l * sin_theta, theta_arr)
            p_l_minus_2, p_l_minus_1 = p_l_minus_1, p_l

    return 2 * np.pi * window


def integrate(beam: ArrayLike, theta: ArrayLike) -> np.number:
    """Integrate an axisymmetric beam over the sphere."""

    beam_arr = np.asarray(beam)
    theta_arr = np.asarray(theta, dtype=float)
    _validate_same_shape(beam_arr, theta_arr)

    return 2 * np.pi * trapezoid(beam_arr * np.sin(theta_arr), theta_arr)


def integrate_dir(beam: ArrayLike, theta: ArrayLike, phi: ArrayLike) -> np.number:
    """Integrate a directional beam sampled over ``theta`` and ``phi``."""

    beam_arr = np.asarray(beam)
    theta_arr = np.asarray(theta, dtype=float)
    phi_arr = np.asarray(phi, dtype=float)
    _validate_same_shape(beam_arr, theta_arr)

    theta_integral = trapezoid(beam_arr * np.sin(theta_arr), theta_arr)
    phi_integral = trapezoid(np.ones_like(phi_arr), phi_arr)
    return theta_integral * phi_integral


__all__ = ["beam2bl", "integrate", "integrate_dir"]
