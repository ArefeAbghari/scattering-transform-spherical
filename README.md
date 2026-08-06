# my-package

Utilities for building spherical Morlet wavelet filter banks and computing
scattering coefficients on Healpy maps.

## Installation

From this directory:

```console
python -m pip install -e ".[dev]"
```

## Quick Start

```python
import healpy as hp

from my_package import ScatteringSph

nside = 128
hmap = hp.read_map("map.fits")

scattering = ScatteringSph(nside=nside, J=4, order=2)
coefficients = scattering(hmap)

s0 = coefficients["S0"][()]
s1_j2 = coefficients["S1"][(2,)]
s2_j1_j3 = coefficients["S2"][(1, 3)]

i1_j2 = scattering.get_i1(2)
i2_j1_j3 = scattering.get_i2(1, 3)

harmonic_filters = scattering.filter_bank(space="harmonic")
real_filters = scattering.filter_bank(space="real")

psi_j2 = harmonic_filters["psi"][2]
phi_j2 = harmonic_filters["phi"][2]
```

Higher-order paths use strictly increasing scales. For example, `S3` contains
paths like `(0, 1, 2)`, but not `(1, 0, 2)` or `(1, 1, 2)`.

## Modules

- `my_package.utils`: numerical integration and `beam2bl`.
- `my_package.wavelets`: Gabor, Morlet, Gaussian, and filter-bank helpers.
- `my_package.transforms`: first- and second-order scattering transforms.
- `my_package.scattering`: high-level `ScatteringSph` interface.

## Development

Run the lightweight tests with:

```console
python -m pytest
```

The transform functions require `healpy`; the lower-level numerical helpers can
be tested without Healpy installed.
