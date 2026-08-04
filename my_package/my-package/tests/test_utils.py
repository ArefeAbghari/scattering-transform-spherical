import numpy as np


from my_package import beam2bl, integrate


def test_integrate_constant_beam_is_sphere_area():
    theta = np.linspace(0, np.pi, 5001)
    beam = np.ones_like(theta)

    assert np.isclose(integrate(beam, theta), 4 * np.pi, rtol=0, atol=1e-6)


def test_beam2bl_constant_beam_has_only_monopole():
    theta = np.linspace(0, np.pi, 5001)
    beam = np.ones_like(theta)

    bl = beam2bl(beam, theta, lmax=4)

    assert np.isclose(bl[0], 4 * np.pi, rtol=0, atol=1e-6)
    assert np.allclose(bl[1:], 0, atol=1e-6)


def test_beam2bl_rejects_mismatched_inputs():
    theta = np.linspace(0, np.pi, 10)
    beam = np.ones(9)

    try:
        beam2bl(beam, theta, lmax=3)
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("beam2bl should reject mismatched beam/theta arrays")

