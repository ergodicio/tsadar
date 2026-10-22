"""
The DLM shape parameter m uses the deck bounds, and the tabulated DLM extends below m = 2 so that the
Maxwellian (m = 2) is an interior point with a two-sided gradient, not the edge of a flat region.
"""

import jax
import numpy as np
import pytest
import scipy.io as sio
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tsadar.core.modules.distribution_functions.base import DLM1V, DLM_M_MIN, dlm_projection_1d
from tsadar.core.modules.distribution_functions.spherical_harmonics import SphericalHarmonics

TABLE = r"tsadar/external/numDistFuncs/DLM_x_-4_-10_10_m_-1_2_5.mat"
X = np.linspace(-10, 10, 200001)


def dlm_cfg(val, lb=1.8, ub=5.0, nvx=128):
    return {"active": True, "nvx": nvx, "params": {"m": {"val": val, "lb": lb, "ub": ub}}}


def test_closed_form_reproduces_the_stored_table():
    table = sio.loadmat(TABLE)["IT"]
    for column, m in ((0, 2.0), (5, 2.5), (10, 3.0), (20, 4.0), (30, 5.0)):
        stored = table[:, column]
        big = stored > 1e-12 * stored.max()
        np.testing.assert_allclose(dlm_projection_1d(m, X)[big], stored[big], rtol=1e-7)


@pytest.mark.parametrize("m", [1.8, 1.9])
def test_extended_columns_are_normalized_dlm_projections(m):
    f = dlm_projection_1d(m, X)
    dx = X[1] - X[0]
    np.testing.assert_allclose(f.sum() * dx, 1.0, rtol=1e-8)
    np.testing.assert_allclose((X**2 * f).sum() * dx, 1.0, rtol=1e-6)
    # heavier-tailed than the Maxwellian below m = 2
    assert dlm_projection_1d(m, 4.0) > dlm_projection_1d(2.0, 4.0)


def test_m_uses_deck_bounds():
    dist = DLM1V(dlm_cfg(2.0, lb=1.8, ub=5.0), activate=False)
    assert dist.m_shift == pytest.approx(1.8) and dist.m_scale == pytest.approx(3.2)
    assert float(dist.get_unnormed_params()["m"]) == pytest.approx(2.0)
    # an unchanged deck range keeps the previous mapping
    legacy = DLM1V(dlm_cfg(3.0, lb=2.0, ub=5.0), activate=False)
    assert legacy.m_shift == pytest.approx(2.0) and legacy.m_scale == pytest.approx(3.0)


@pytest.mark.parametrize("lb, ub", [(1.7, 5.0), (2.0, 5.5), (3.0, 3.0)])
def test_bounds_outside_the_table_are_rejected(lb, ub):
    with pytest.raises(ValueError, match="DLM m bounds"):
        DLM1V(dlm_cfg(3.0, lb=lb, ub=ub))


def test_unchanged_for_m_at_or_above_two():
    """The m >= 2 columns and nodes are the stored ones, so existing DLM fe values do not move."""
    for m in (2.0, 2.37, 3.0, 4.6):
        new = DLM1V(dlm_cfg(m, lb=1.8), activate=False)()
        old = DLM1V(dlm_cfg(m, lb=2.0), activate=False)()
        np.testing.assert_array_equal(np.asarray(new), np.asarray(old))


def test_maxwellian_is_an_interior_point_with_a_two_sided_gradient():
    dist = DLM1V(dlm_cfg(2.0, lb=1.8), activate=False)
    weight = jnp.asarray(dist.vx) ** 4  # a tail-sensitive functional of fe

    def tail_moment(m):
        return jnp.sum(weight * dist.call_matte(m)) * (dist.vx[1] - dist.vx[0])

    below = float(jax.grad(tail_moment)(1.95))
    above = float(jax.grad(tail_moment)(2.05))
    assert below < 0 and above < 0  # the 4th moment falls as m rises, on both sides of 2
    assert abs(below) > 0.1 * abs(above)
    assert tail_moment(1.9) > tail_moment(2.0) > tail_moment(2.1)


def test_spherical_harmonics_use_deck_bounds():
    cfg = {
        "nvx": 32, "active": True, "type": "sphericalharmonic", "dim": 2,
        "params": {"Nl": 1, "init_m": 1.9, "nvr": 32, "flm_type": "arbitrary", "learn_log": True,
                   "m": {"lb": DLM_M_MIN, "ub": 5.0, "val": 3.0}, "smooth": True},
    }
    dist = SphericalHarmonics(cfg)
    assert float(dist.get_unnormed_m()) == pytest.approx(1.9)
    cfg["params"]["init_m"] = 1.7
    with pytest.raises(ValueError, match="init_m"):
        SphericalHarmonics(cfg)
