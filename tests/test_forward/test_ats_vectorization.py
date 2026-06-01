"""Regression tests locking the vectorized ATS reductions to their original semantics.

The angular (ATS) integration tests skip on CPU, so these pure-function checks guard
the vectorized replacements of the former unrolled Python loops in
``thomson_diagnostic._bin_average`` and ``irf.add_ATS_IRF`` against the original
list-comprehension implementations.
"""
import numpy as np
import pytest
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

from jax import numpy as jnp, vmap

from tsadar.core.thomson_diagnostic import _bin_average


def _old_reduce(ThryE, lamAxisE, lam_step, ang_step):
    ThryE = jnp.array([jnp.average(ThryE[:, i : i + lam_step], axis=1) for i in range(0, ThryE.shape[1], lam_step)])
    ThryE = jnp.array([jnp.average(ThryE[:, i : i + ang_step], axis=1) for i in range(0, ThryE.shape[1], ang_step)])
    lamAxisE = jnp.array([jnp.average(lamAxisE[i : i + lam_step], axis=0) for i in range(0, lamAxisE.shape[0], lam_step)])
    return ThryE, lamAxisE


def _new_reduce(ThryE, lamAxisE, lam_step, ang_step):
    ThryE = _bin_average(ThryE, lam_step, axis=1)
    ThryE = _bin_average(ThryE, ang_step, axis=0)
    lamAxisE = _bin_average(lamAxisE, lam_step, axis=0)
    return ThryE, lamAxisE


def _old_conv(modlE, inst_func_ang, inst_func_lam):
    ThryE = jnp.array([jnp.convolve(modlE[:, i], inst_func_ang, "same") for i in range(modlE.shape[1])])
    ThryE = jnp.array([jnp.convolve(ThryE[:, i], inst_func_lam, "same") for i in range(ThryE.shape[1])])
    return ThryE


def _new_conv(modlE, inst_func_ang, inst_func_lam):
    ThryE = vmap(lambda col: jnp.convolve(col, inst_func_ang, "same"), in_axes=1, out_axes=1)(modlE)
    ThryE = vmap(lambda row: jnp.convolve(row, inst_func_lam, "same"), in_axes=0, out_axes=0)(ThryE)
    return ThryE


@pytest.mark.parametrize(
    "nang, npts, lam_step, ang_step",
    [
        (1024, 2048, 2, 1),  # angular shapes (divisible)
        (200, 1000, 5, 4),  # divisible
        (130, 1003, 7, 3),  # ragged final window on both axes
    ],
)
def test_bin_average_matches_loop(nang, npts, lam_step, ang_step):
    rng = np.random.default_rng(0)
    ThryE = jnp.array(rng.random((nang, npts)))
    lamAxisE = jnp.array(rng.random((npts,)))

    oT, oL = _old_reduce(ThryE, lamAxisE, lam_step, ang_step)
    nT, nL = _new_reduce(ThryE, lamAxisE, lam_step, ang_step)

    assert oT.shape == nT.shape and oL.shape == nL.shape
    np.testing.assert_allclose(np.asarray(nT), np.asarray(oT), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(nL), np.asarray(oL), rtol=0, atol=1e-12)


@pytest.mark.parametrize("nang, npts", [(256, 512), (300, 200), (64, 333)])
def test_ats_irf_conv_matches_loop(nang, npts):
    rng = np.random.default_rng(0)
    modlE = jnp.array(rng.random((nang, npts)))
    inst_func_ang = jnp.array(rng.random((nang,)))
    inst_func_lam = jnp.array(rng.random((npts,)))

    old = _old_conv(modlE, inst_func_ang, inst_func_lam)
    new = _new_conv(modlE, inst_func_ang, inst_func_lam)

    assert old.shape == new.shape
    np.testing.assert_allclose(np.asarray(new), np.asarray(old), rtol=0, atol=1e-12)
