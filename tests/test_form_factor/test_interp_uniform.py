"""Pins `interp_uniform` to `jnp.interp` on the grids the form factor actually uses.

Every interpolation grid in TSADAR is uniform by construction, so the binary search inside
`jnp.interp` can be replaced with O(1) index arithmetic. These tests check the replacement against
`jnp.interp` on the real grids, including the edge-fill behavior the call sites rely on.
"""
import numpy as np
import pytest
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

from jax import numpy as jnp

from tsadar.core.physics.form_factor import zprimeMaxw
from tsadar.core.physics.interpolation import interp_uniform

MINMAX, H1, H = 8.2, 1024, 0.01

XI1 = jnp.linspace(-MINMAX - jnp.sqrt(2.0) / H1, MINMAX + jnp.sqrt(2.0) / H1, H1)  # linspace grid
XI2 = jnp.array(jnp.arange(-MINMAX, MINMAX, H))  # arange grid
VX = jnp.linspace(-7.0 + 0.005, 7.0 - 0.005, 1400)  # distribution-function grid


@pytest.mark.parametrize("xp", [XI1, XI2, VX], ids=["xi1", "xi2", "vx"])
@pytest.mark.parametrize("shape", [(4000,), (3, 40, 5, 2)], ids=["flat", "4d"])
def test_matches_jnp_interp(xp, shape):
    rng = np.random.default_rng(0)
    # white noise is the worst case: adjacent samples jump by O(1), so any index or weight error
    # shows up at full amplitude rather than being smoothed away
    fp = jnp.array(rng.standard_normal(xp.size))
    x = jnp.array(rng.uniform(float(xp[0]) - 4.0, float(xp[-1]) + 4.0, shape))

    out = interp_uniform(x, xp, fp)
    assert out.shape == x.shape
    np.testing.assert_allclose(np.asarray(out), np.asarray(jnp.interp(x, xp, fp)), rtol=0, atol=1e-11)


def test_matches_jnp_interp_on_zprime_tables():
    """The `Zpi` call sites, with their scalar and array-valued edge fills."""
    Zpi = jnp.array(zprimeMaxw(XI2))
    rng = np.random.default_rng(1)
    xii = jnp.array(rng.uniform(-40.0, 40.0, (2, 300, 5, 2)))

    real = interp_uniform(xii, XI2, Zpi[0, :], left=xii**-2, right=xii**-2)
    imag = interp_uniform(xii, XI2, Zpi[1, :], left=0, right=0)

    np.testing.assert_allclose(
        np.asarray(real), np.asarray(jnp.interp(xii, XI2, Zpi[0, :], left=xii**-2, right=xii**-2)),
        rtol=0, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(imag), np.asarray(jnp.interp(xii, XI2, Zpi[1, :], left=0, right=0)), rtol=0, atol=1e-12
    )


def test_edges_and_grid_points():
    """Exact grid points, both endpoints, and far out-of-range queries."""
    rng = np.random.default_rng(2)
    fp = jnp.array(rng.standard_normal(XI2.size))
    x = jnp.concatenate([XI2, jnp.array([XI2[0], XI2[-1], -1e9, 1e9, float(XI2[0]) - 1e-9, 0.0])])

    for kw in ({}, {"left": 0.0, "right": 0.0}, {"left": -3.0, "right": 7.0}):
        np.testing.assert_allclose(
            np.asarray(interp_uniform(x, XI2, fp, **kw)),
            np.asarray(jnp.interp(x, XI2, fp, **kw)),
            rtol=0,
            atol=1e-11,
        )

    # on-grid queries must return the sample itself, not a neighbor blend
    np.testing.assert_allclose(np.asarray(interp_uniform(XI2, XI2, fp)), np.asarray(fp), rtol=0, atol=1e-11)


def test_default_fill_is_the_endpoint_value():
    fp = jnp.array([1.0, 2.0, 3.0, 4.0])
    xp = jnp.linspace(0.0, 3.0, 4)
    out = interp_uniform(jnp.array([-10.0, 0.0, 1.5, 3.0, 10.0]), xp, fp)
    np.testing.assert_allclose(np.asarray(out), np.array([1.0, 1.0, 2.5, 4.0, 4.0]), rtol=0, atol=1e-14)
