"""
Tests for ``tsadar.utils.vector_tools.rotate``, which rotates 2D distribution functions (the
multiplexed-shot ``shot_rot`` and the rotated bi-DLM initial guess).

The distribution functions live on a symmetric cell-centered grid, so ``v = 0`` falls between the two
middle samples. Rotating about any other point adds a spurious drift: before the fix the rotation was
about index ``n / 2`` and a 90-degree turn moved an isotropic Maxwellian by about one cell.
"""

import numpy as np
import pytest
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tsadar.utils.vector_tools import rotate

N = 128
DV = 16.0 / N
V = (np.arange(N) - (N - 1) / 2) * DV
VX, VY = np.meshgrid(V, V)  # arrays indexed [vy, vx], as the 2D distribution functions are


def normalized(f):
    return f / (f.sum() * DV**2)


def centroid(f):
    return (f * VX).sum() / f.sum(), (f * VY).sum() / f.sum()


MAXWELLIAN = normalized(np.exp(-0.5 * (VX**2 + VY**2)))
# off-center and anisotropic, so the rotation sense and the exact quarter turns are both visible
TEST_EDF = normalized(np.exp(-0.5 * ((VX - 1.0) ** 2 / 1.5 + VY**2 / 0.7)))


@pytest.mark.parametrize("degrees", [30.0, 45.0, 90.0, 135.0, 180.0, 270.0])
def test_isotropic_distribution_is_invariant(degrees):
    rotated = np.asarray(rotate(jnp.asarray(MAXWELLIAN), np.deg2rad(degrees)))
    quarter_turn = degrees % 90 == 0
    # bilinear interpolation error only (exact for quarter turns): ~3e-6 in area at 45 degrees on this grid
    np.testing.assert_allclose(rotated.sum() * DV**2, 1.0, rtol=1e-12 if quarter_turn else 1e-5)
    np.testing.assert_allclose(centroid(rotated), (0.0, 0.0), atol=1e-12)
    tolerance = 1e-12 if quarter_turn else 5e-3
    assert np.abs(rotated - MAXWELLIAN).max() <= tolerance * MAXWELLIAN.max()


def test_quarter_turn_is_counter_clockwise_and_exact():
    rotated = np.asarray(rotate(jnp.asarray(TEST_EDF), np.pi / 2))
    # counter-clockwise in (vx, vy): the lobe at vx = +1 moves to vy = +1
    np.testing.assert_allclose(centroid(TEST_EDF), (1.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(centroid(rotated), (0.0, 1.0), atol=1e-6)
    # with rows indexed by vy, that is np.rot90 by -1 of the array
    np.testing.assert_allclose(rotated, np.rot90(TEST_EDF, -1), rtol=0, atol=1e-14 * TEST_EDF.max())


def test_rotation_composes_to_identity():
    there = rotate(jnp.asarray(TEST_EDF), np.deg2rad(37.0))
    back = np.asarray(rotate(there, np.deg2rad(-37.0)))
    # two bilinear interpolations smooth the peak slightly; the centroid must still come back
    np.testing.assert_allclose(centroid(back), centroid(TEST_EDF), atol=2e-3)
    assert np.abs(back - TEST_EDF).max() <= 2e-2 * TEST_EDF.max()
