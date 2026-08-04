"""High-level spherical scattering interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .wavelets import (
    FilterBankDict,
    filter_bank_real, 
    filter_bank_harmonic
)

ScalePath = Tuple[int, ...]
CoefficientDict = Dict[str, Dict[ScalePath, Union[float, complex]]]
IntermediateDict = Dict[str, Dict[ScalePath, np.ndarray]]




def _import_healpy():
    try:
        import healpy as hp
    except ImportError as exc:
        raise ImportError(
            "healpy is required for ScatteringSph. "
            "Install the package with its runtime dependencies first."
        ) from exc
    return hp


@dataclass
class ScatteringSph:
    """Compute spherical scattering coefficients for Healpy maps.

    Parameters
    ----------
    nside
        Healpy map resolution parameter.
    J
        Number of dyadic wavelet scales.
    order
        Maximum scattering order to compute.
    lmax
        Maximum multipole. Defaults to ``3 * nside - 1``.
    theta_bin
        Number of angular samples used to build real-space filters.
    """

    nside: int
    J: int
    order: int = 2
    lmax: Optional[int] = None
    theta_bin: int = 1000
    _hp: object = field(default=None, init=False, repr=False)
    _resol: float = field(init=False, repr=False)
    _wavelet_l: Optional[List[np.ndarray]] = field(default=None, init=False, repr=False)
    _gaussian_l: Optional[List[np.ndarray]] = field(default=None, init=False, repr=False)
    coefficients_: Optional[CoefficientDict] = field(default=None, init=False)
    intermediate_maps_: Optional[IntermediateDict] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._hp = _import_healpy()

        if self.nside < 1:
            raise ValueError("nside must be positive")
        if hasattr(self._hp, "isnsideok") and not self._hp.isnsideok(self.nside):
            raise ValueError("nside must be a valid Healpy nside")
        if self.J < 1:
            raise ValueError("J must be at least 1")
        if self.order < 0:
            raise ValueError("order must be non-negative")
        if self.theta_bin < 2:
            raise ValueError("theta_bin must be at least 2")

        self.lmax = 3 * self.nside - 1 if self.lmax is None else self.lmax
        if self.lmax < 0:
            raise ValueError("lmax must be non-negative")

        self._resol = self._hp.nside2resol(self.nside, arcmin=False)

    @property
    def resol(self) -> float:
        """Pixel resolution in radians, derived from ``nside``."""

        return self._resol

    def wavelet_filters(self) -> List[np.ndarray]:
        """Return harmonic-space wavelet filters."""

        if self._wavelet_l is None:
            self._wavelet_l = filter_bank_harmonic(
                nside=self.nside,
                jmax=self.J,
                lmax=self.lmax,
                theta_bin=self.theta_bin,
            )["psi"]

        return [filter_.copy() for filter_ in self._wavelet_l]

    def gaussian_filters(self) -> List[np.ndarray]:
        """Return harmonic-space Gaussian smoothing filters."""

        if self._gaussian_l is None:
            self._gaussian_l = filter_bank_harmonic(
                nside=self.nside,
                jmax=self.J,
                lmax=self.lmax,
                theta_bin=self.theta_bin,
            )["phi"]

        return [filter_.copy() for filter_ in self._gaussian_l]

    def filter_bank(self, space: str = "harmonic") -> FilterBankDict:
        """Return ``psi`` and ``phi`` filters in real or harmonic space."""

        if space == "real":
            return filter_bank_real(
                nside=self.nside,
                jmax=self.J,
                theta_bin=self.theta_bin,
            )
        if space == "harmonic":
            return filter_bank_harmonic(
                nside=self.nside,
                jmax=self.J,
                lmax=self.lmax,
                theta_bin=self.theta_bin,
            )
        raise ValueError("space must be 'real' or 'harmonic'")

    def compute_coefficients(
        self,
        hmap: np.ndarray,
        smooth: bool = False,
        keep_maps: bool = True,
    ) -> CoefficientDict:
        """Compute scattering coefficients up to ``self.order``.

        The result is grouped by order, and scales are strictly increasing
        after first order. For example,
        ``coefficients["S1"][(2,)]`` gives the first-order coefficient at scale
        ``j=2`` and ``coefficients["S2"][(1, 3)]`` gives the second-order
        coefficient along path ``j1=1, j2=3``.
        """

        hmap = np.asarray(hmap)
        expected_npix = self._hp.nside2npix(self.nside)
        if hmap.size != expected_npix:
            raise ValueError(
                f"hmap has {hmap.size} pixels, expected {expected_npix} for nside={self.nside}"
            )

        coefficients: CoefficientDict = {"S0": {(): self._mean_value(hmap)}}
        intermediate_maps: IntermediateDict = {"I0": {(): hmap.copy()}} if keep_maps else {}
        if self.order == 0:
            self.coefficients_ = coefficients
            self.intermediate_maps_ = intermediate_maps
            return coefficients

        wavelets_l = self.wavelet_filters()
        gaussians_l = self.gaussian_filters() if smooth else None

        current_maps: List[Tuple[ScalePath, np.ndarray]] = [((), hmap)]
        for current_order in range(1, self.order + 1):
            order_key = f"S{current_order}"
            intermediate_key = f"I{current_order}"
            coefficients[order_key] = {}
            if keep_maps:
                intermediate_maps[intermediate_key] = {}
            next_maps = []

            for path, input_map in current_maps:
                input_alm = self._hp.map2alm(
                    input_map,
                    lmax=self.lmax,
                    use_pixel_weights=True,
                )

                start_scale = path[-1] + 1 if path else 0
                for j in range(start_scale, self.J):
                    wavelet_l = wavelets_l[j]
                    new_path = path + (j,)
                    filtered_alm = self._hp.almxfl(input_alm, wavelet_l)
                    modulus_map = np.abs(
                        self._hp.alm2map(filtered_alm, self.nside, lmax=self.lmax)
                    )
                    coefficient_map = self._smooth_map(modulus_map, gaussians_l, j)
                    coefficients[order_key][new_path] = self._mean_value(coefficient_map)
                    if keep_maps:
                        intermediate_maps[intermediate_key][new_path] = modulus_map.copy()
                    next_maps.append((new_path, modulus_map))

            current_maps = next_maps

        self.coefficients_ = coefficients
        self.intermediate_maps_ = intermediate_maps
        return coefficients

    def __call__(
        self,
        hmap: np.ndarray,
        smooth: bool = False,
        keep_maps: bool = True,
    ) -> CoefficientDict:
        """Compute coefficients and store them on the object."""

        return self.compute_coefficients(hmap, smooth=smooth, keep_maps=keep_maps)

    def coefficient(
        self,
        order: int,
        scales: Sequence[int],
        coefficients: Optional[CoefficientDict] = None,
    ) -> Union[float, complex]:
        """Read one coefficient from a coefficient dictionary."""

        if coefficients is None:
            coefficients = self._require_coefficients()
        return coefficients[f"S{order}"][tuple(scales)]

    def get_s(self, order: int, scales: Sequence[int]) -> Union[float, complex]:
        """Return a coefficient from the last computed result."""

        return self.coefficient(order=order, scales=scales)

    def get_i(self, order: int, scales: Sequence[int]) -> np.ndarray:
        """Return an intermediate modulus map from the last computed result."""

        maps = self._require_intermediate_maps()
        return maps[f"I{order}"][tuple(scales)].copy()

    # def get_s1(self, j: int) -> Union[float, complex]:
    #     """Return the first-order coefficient at scale ``j``."""

    #     return self.get_s(order=1, scales=(j,))

    # def get_s2(self, j1: int, j2: int) -> Union[float, complex]:
    #     """Return the second-order coefficient at scales ``j1, j2``."""

    #     return self.get_s(order=2, scales=(j1, j2))

    # def get_i1(self, j: int) -> np.ndarray:
    #     """Return the first-order intermediate map at scale ``j``."""

    #     return self.get_i(order=1, scales=(j,))

    # def get_i2(self, j1: int, j2: int) -> np.ndarray:
    #     """Return the second-order intermediate map at scales ``j1, j2``."""

    #     return self.get_i(order=2, scales=(j1, j2))

    def _smooth_map(
        self,
        hmap: np.ndarray,
        gaussians_l: Optional[List[np.ndarray]],
        scale: int,
    ) -> np.ndarray:
        if gaussians_l is None:
            return hmap

        alm = self._hp.map2alm(hmap, lmax=self.lmax, use_pixel_weights=True)
        return self._hp.alm2map(
            self._hp.almxfl(alm, gaussians_l[scale]),
            self.nside,
            lmax=self.lmax,
        )

    @staticmethod
    def _mean_value(values: np.ndarray) -> Union[float, complex]:
        mean = np.real_if_close(np.mean(values))
        return mean.item()

    def _require_coefficients(self) -> CoefficientDict:
        if self.coefficients_ is None:
            raise RuntimeError("run the scattering object on a map before reading coefficients")
        return self.coefficients_

    def _require_intermediate_maps(self) -> IntermediateDict:
        if not self.intermediate_maps_:
            raise RuntimeError("run with keep_maps=True before reading intermediate maps")
        return self.intermediate_maps_


__all__ = ["CoefficientDict", "IntermediateDict", "ScalePath", "ScatteringSph"]
