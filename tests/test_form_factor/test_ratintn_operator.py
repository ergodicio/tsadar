"""Pins the precomputed `ratintn` operator to the quadrature it replaces.

`FormFactor.__call__` used to rebuild the `chiERratprim` quadrature on every forward pass even
though `ratintn` is exactly linear in its first argument and the other two are fixed by the grids
set in `__init__`. These tests check that the constant matrix assembled by `ratintn_operator`
reproduces the original `vmap(ratintn)` call to roundoff, on the actual `FormFactor` grids.
"""
import numpy as np
import pytest
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

from jax import numpy as jnp, vmap

from tsadar.core.physics import ratintn


def _form_factor_grids():
    """The xi1 / xi2 grids exactly as `FormFactor.__init__` builds them."""
    minmax, h1, h = 8.2, 1024, 0.01
    xi1 = jnp.linspace(-minmax - jnp.sqrt(2.0) / h1, minmax + jnp.sqrt(2.0) / h1, h1)
    xi2 = jnp.array(jnp.arange(-minmax, minmax, h))
    return xi1, xi2


def _old_chiERratprim(ratdf, xi1, xi2):
    return vmap(ratintn.ratintn, in_axes=(None, 0, None))(ratdf, xi1[None, :] - xi2[:, None], xi1)[:, 0]


def _ratdf(xi1, kind):
    dxi = xi1[1] - xi1[0]
    rng = np.random.default_rng(0)
    if kind == "maxwellian":
        return jnp.gradient(jnp.exp(-(xi1**2) / 2.0), dxi)
    if kind == "super_gaussian":
        return jnp.gradient(jnp.exp(-jnp.abs(xi1 / 1.3) ** 3.5), dxi)
    if kind == "two_stream":
        f = jnp.exp(-((xi1 - 1.5) ** 2) / 2.0) + 0.3 * jnp.exp(-((xi1 + 2.5) ** 2) / 0.5)
        return jnp.gradient(f, dxi)
    if kind == "random":
        return jnp.array(rng.standard_normal(xi1.size))
    if kind == "zeros":
        return jnp.zeros(xi1.size)
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["maxwellian", "super_gaussian", "two_stream", "random", "zeros"])
def test_operator_matches_quadrature(kind):
    xi1, xi2 = _form_factor_grids()
    M = ratintn.ratintn_operator(xi1[None, :] - xi2[:, None], xi1)
    ratdf = _ratdf(xi1, kind)

    old = np.asarray(_old_chiERratprim(ratdf, xi1, xi2))
    new = np.asarray(M @ ratdf)

    assert M.shape == (xi2.size, xi1.size)
    np.testing.assert_allclose(new, old, rtol=0, atol=1e-12)


def test_operator_is_the_linear_map():
    """The operator is only valid because `ratintn` is linear in `f`; assert that directly."""
    xi1, xi2 = _form_factor_grids()
    a = _ratdf(xi1, "maxwellian")
    b = _ratdf(xi1, "two_stream")

    combined = _old_chiERratprim(3.0 * a + 2.5 * b, xi1, xi2)
    separate = 3.0 * _old_chiERratprim(a, xi1, xi2) + 2.5 * _old_chiERratprim(b, xi1, xi2)

    np.testing.assert_allclose(np.asarray(combined), np.asarray(separate), rtol=1e-10, atol=1e-12)


def test_operator_batch_shape():
    """A single `g` row reproduces the unbatched `ratintn` call."""
    xi1, xi2 = _form_factor_grids()
    ratdf = _ratdf(xi1, "maxwellian")

    for i in (0, 700, xi2.size - 1):
        g = xi1 - xi2[i]
        M = ratintn.ratintn_operator(g, xi1)
        assert M.shape == (xi1.size,)
        np.testing.assert_allclose(
            float(M @ ratdf), float(ratintn.ratintn(ratdf, g, xi1)[0]), rtol=0, atol=1e-12
        )
