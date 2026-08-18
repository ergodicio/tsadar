"""
Numerical tests for the ``custom_jvp`` on :mod:`tsadar.core.physics.ratintn`.

``ratcen`` was rewritten from a single fused expression into ``P * f_av + Q * f_dif``, where ``P`` and ``Q``
depend only on ``g`` and carry a hand-written tangent rule. Two properties have to hold: the refactor is
behaviour preserving against the fused form it replaced, and the resulting derivative genuinely exploits the
linearity in ``f`` instead of differentiating through both sides of the near-pole branch.
"""

import numpy as np
import pytest
from jax import config, jacfwd, jacrev, jit, grad, vjp, vmap, numpy as jnp

config.update("jax_enable_x64", True)

from tsadar.core.physics.ratintn import ratintn


def _ratintn_pre_refactor(f, g, z):
    """The implementation this change replaced, kept verbatim as the behavioural reference."""

    if len(jnp.shape(f)) == 1:
        f = jnp.transpose(f[..., jnp.newaxis])

    fdif = f[:, 1:-1] - f[:, 0:-2]
    gdif = g[1:-1] - g[0:-2]
    fav = 0.5 * (f[:, 1:-1] + f[:, 0:-2])
    gav = 0.5 * (g[1:-1] + g[0:-2])
    tmp = fav * gdif - gav * fdif

    use_rf = jnp.abs(gdif) < 1.0e-4 * jnp.abs(gav)
    gav_safe = jnp.where(use_rf, gav, 1.0)
    rf = fav / gav_safe + tmp * gdif / (12.0 * gav_safe**3)
    rfn = fdif / gdif + tmp * jnp.log((gav + (0.5 + 0j) * gdif) / (gav - 0.5 * gdif)) / gdif**2

    return jnp.sum(jnp.real(jnp.where(use_rf[None, :], rf, rfn)) * (z[1:-1] - z[0:-2]), 1)


def _grid(kind):
    """
    Return ``(f, g, z)`` for a regime of interest.

    The branch mask is ``|gdif| < 1e-4 * |gav|``, so which branch runs is a property of the grid and the pole
    location, not of ``f``. Each case below is asserted to actually select what its name claims.
    """

    rng = np.random.default_rng(0)

    if kind == "exact":
        # pole inside the grid, spacing nowhere negligible against gav -> exact (logarithmic) branch
        z = jnp.linspace(-3.0, 3.0, 201)
        g = z - 0.37
    elif kind == "taylor":
        # a short span sitting far from the pole -> gdif is negligible against gav -> Taylor branch
        z = jnp.linspace(-1.0e-3, 1.0e-3, 51)
        g = z + 10.0
    elif kind == "mixed":
        # a fine patch far from the pole and a coarse patch straddling it -> both branches live at once.
        # The pole is deliberately off-node: landing it exactly on a grid point makes the integral itself
        # logarithmically singular, which is a property of the problem rather than of either implementation.
        z = jnp.concatenate([jnp.linspace(-3.0, -2.9, 600), jnp.linspace(-0.05, 0.05, 101)])
        g = z - 0.0115
    elif kind == "zero_gav":
        # power-of-two spacing on an even, symmetric grid makes gav land on exactly 0 for one interval,
        # which is the state GPU rounding produces and where the old double-where was load-bearing
        z = (jnp.arange(200) - 199 / 2) * 2.0**-5
        g = z
    else:
        raise ValueError(kind)

    return jnp.asarray(rng.standard_normal(z.size)), g, z


def _branch_counts(g):
    gdif = g[1:-1] - g[0:-2]
    gav = 0.5 * (g[1:-1] + g[0:-2])
    use_rf = jnp.abs(gdif) < 1.0e-4 * jnp.abs(gav)

    return int(jnp.sum(use_rf)), int(jnp.sum(~use_rf))


ALL_GRIDS = ["exact", "taylor", "mixed", "zero_gav"]


def test_grids_select_the_branches_they_claim():
    """Guard the fixtures themselves -- a silently one-sided grid would make the tests below vacuous."""

    assert _branch_counts(_grid("exact")[1]) == (0, 199)
    assert _branch_counts(_grid("taylor")[1]) == (49, 0)

    taylor, exact = _branch_counts(_grid("mixed")[1])
    assert taylor > 0 and exact > 0

    _, g, _ = _grid("zero_gav")
    gav = 0.5 * (g[1:-1] + g[0:-2])
    assert jnp.sum(gav == 0.0) == 1, "fixture no longer produces an exactly-zero gav"


@pytest.mark.parametrize("kind", ALL_GRIDS)
def test_primal_matches_pre_refactor(kind):
    """The rewrite reassociates the arithmetic but must not move the answer."""

    f, g, z = _grid(kind)
    np.testing.assert_allclose(ratintn(f, g, z), _ratintn_pre_refactor(f, g, z), rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("kind", ALL_GRIDS)
def test_gradients_match_pre_refactor(kind):
    """Both cotangents must agree with differentiating straight through the fused expression."""

    f, g, z = _grid(kind)
    w = jnp.asarray(np.random.default_rng(1).standard_normal(1))

    def loss(ff, gg, fn):
        return jnp.sum(fn(ff, gg, z) * w)

    np.testing.assert_allclose(
        grad(loss, argnums=0)(f, g, ratintn),
        grad(loss, argnums=0)(f, g, _ratintn_pre_refactor),
        rtol=1e-11,
        atol=0.0,
    )
    np.testing.assert_allclose(
        grad(loss, argnums=1)(f, g, ratintn),
        grad(loss, argnums=1)(f, g, _ratintn_pre_refactor),
        rtol=1e-9,
        atol=1e-12,
    )


@pytest.mark.parametrize("kind", ALL_GRIDS)
def test_exactly_linear_in_f(kind):
    """``ratintn`` is a linear operator in ``f``: it kills zero and respects superposition exactly."""

    f, g, z = _grid(kind)
    other = jnp.asarray(np.random.default_rng(2).standard_normal(f.size))
    alpha, beta = 2.5, -0.75

    np.testing.assert_array_equal(ratintn(jnp.zeros_like(f), g, z), jnp.zeros_like(ratintn(f, g, z)))
    np.testing.assert_allclose(
        ratintn(alpha * f + beta * other, g, z),
        alpha * ratintn(f, g, z) + beta * ratintn(other, g, z),
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize("kind", ALL_GRIDS)
def test_vjp_in_f_is_the_transpose(kind):
    """
    The point of the refactor: with ``f`` outside every branch, its VJP is the adjoint of the same stencil.

    Checked as the defining adjoint identity ``<ct, A f> == <A^T ct, f>`` rather than against a reference
    gradient, so it holds independently of how either implementation computes the coefficients.
    """

    f, g, z = _grid(kind)
    out, pull = vjp(lambda ff: ratintn(ff, g, z), f)
    ct = jnp.asarray(np.random.default_rng(3).standard_normal(out.shape))

    np.testing.assert_allclose(jnp.sum(ct * out), jnp.sum(pull(ct)[0] * f), rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("kind", ["exact", "taylor", "zero_gav"])
def test_forward_and_reverse_mode_agree(kind):
    """
    Forward mode has to keep working, and has to agree with reverse mode.

    The tangent rule is written with ``custom_jvp`` precisely so JAX can transpose it; a ``custom_vjp`` would
    define reverse mode only and make ``jacfwd`` raise, which would break the Levenberg-Marquardt path.
    """

    f, g, z = _grid(kind)

    np.testing.assert_allclose(
        jacfwd(lambda gg: ratintn(f, gg, z))(g), jacrev(lambda gg: ratintn(f, gg, z))(g), rtol=1e-11, atol=1e-14
    )
    np.testing.assert_allclose(
        jacfwd(lambda ff: ratintn(ff, g, z))(f), jacrev(lambda ff: ratintn(ff, g, z))(f), rtol=1e-11, atol=1e-14
    )


def test_gradients_stay_finite_when_gav_is_exactly_zero():
    """
    The failure mode the custom rule exists to remove.

    Where the pole is centred in an interval ``gav`` is exactly zero, so the unselected Taylor branch is inf.
    Reverse mode through the old fused form multiplied that inf by the branch mask's zero and produced nan on
    GPU; only the double-``where`` kept it finite. The tangent rule never evaluates the unselected branch, and
    it divides by the interval endpoints rather than by ``gav``, so nothing here can go non-finite.
    """

    f, g, z = _grid("zero_gav")
    gav = 0.5 * (g[1:-1] + g[0:-2])
    assert jnp.any(gav == 0.0)

    value = ratintn(f, g, z)
    df = grad(lambda ff: jnp.sum(ratintn(ff, g, z)))(f)
    dg = grad(lambda gg: jnp.sum(ratintn(f, gg, z)))(g)

    assert jnp.all(jnp.isfinite(value))
    assert jnp.all(jnp.isfinite(df))
    assert jnp.all(jnp.isfinite(dg))


def test_vmap_over_g_matches_a_loop():
    """The 1D form factor maps over a stack of ``g`` rows; batching must not perturb value or gradient."""

    f, _, z = _grid("exact")
    poles = jnp.asarray([0.37, -1.21, 0.511, 2.53])
    stacked = z[None, :] - poles[:, None]

    batched = vmap(ratintn, in_axes=(None, 0, None))(f, stacked, z)
    looped = jnp.stack([ratintn(f, z - p, z) for p in poles])
    np.testing.assert_allclose(batched, looped, rtol=1e-12, atol=0.0)

    w = jnp.asarray(np.random.default_rng(4).standard_normal((poles.size, 1)))
    dg_batched = grad(lambda gg: jnp.sum(vmap(ratintn, in_axes=(None, 0, None))(f, gg, z) * w))(stacked)
    dg_looped = jnp.stack([grad(lambda gg, i=i: jnp.sum(ratintn(f, gg, z) * w[i]))(z - p) for i, p in enumerate(poles)])
    np.testing.assert_allclose(dg_batched, dg_looped, rtol=1e-11, atol=1e-14)


@pytest.mark.parametrize("kind", ALL_GRIDS)
def test_jit_is_transparent(kind):
    """Nothing in the custom rule depends on being traced eagerly."""

    f, g, z = _grid(kind)
    loss = lambda ff, gg: jnp.sum(ratintn(ff, gg, z))

    np.testing.assert_allclose(jit(ratintn)(f, g, z), ratintn(f, g, z), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(jit(grad(loss, argnums=1))(f, g), grad(loss, argnums=1)(f, g), rtol=0.0, atol=0.0)
