"""Correctness and autodiff tests for the principal-value quadrature."""

import numpy as np
import pytest
from scipy.special import dawsn
from jax import config, grad, jacfwd, jacrev, jit, jvp, vjp, vmap, numpy as jnp

config.update("jax_enable_x64", True)

from tsadar.core.physics.ratintn import ratintn, ratintn_operator


DTYPES = [jnp.float32, jnp.float64]


def _smooth_numerator(z):
    """A non-symmetric smooth numerator with nontrivial pole derivatives."""

    return jnp.exp(-0.7 * (z - 0.23) ** 2) * (1.0 + 0.15 * z + 0.03 * z**2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_all_intervals_are_included_in_direct_and_operator(dtype):
    """The last adjacent pair contributes in both the direct and matrix paths."""

    z = jnp.linspace(0.0, 1.0, 10, dtype=dtype)
    denominator = jnp.ones_like(z)

    constant = jnp.ones_like(z)
    np.testing.assert_allclose(ratintn(constant, denominator, z), 1.0, rtol=0.0, atol=5 * jnp.finfo(dtype).eps)
    np.testing.assert_allclose(
        ratintn_operator(denominator, z) @ constant,
        1.0,
        rtol=0.0,
        atol=5 * jnp.finfo(dtype).eps,
    )

    # Only the formerly omitted last interval is nonzero. Its trapezoid area is dz.
    final_interval = jnp.zeros_like(z).at[-1].set(2.0)
    expected = z[-1] - z[-2]
    np.testing.assert_allclose(
        ratintn(final_interval, denominator, z),
        expected,
        rtol=0.0,
        atol=5 * jnp.finfo(dtype).eps,
    )
    np.testing.assert_allclose(
        ratintn_operator(denominator, z) @ final_interval,
        expected,
        rtol=0.0,
        atol=5 * jnp.finfo(dtype).eps,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_every_node_and_midpoint_has_finite_primal_jvp_and_vjp(dtype):
    """Exercise every possible singular interval, including exact-node cancellation."""

    z = jnp.linspace(-4.0, 4.0, 17, dtype=dtype)
    f = _smooth_numerator(z)
    poles = jnp.concatenate((z[1:-1], 0.5 * (z[:-1] + z[1:])))

    def pole_value(pole):
        return ratintn(f, z - pole, z)[0]

    values, tangents = vmap(lambda pole: jvp(pole_value, (pole,), (jnp.ones_like(pole),)))(poles)
    reverse = vmap(grad(pole_value))(poles)
    _, numerator_pullback = vjp(
        lambda numerator: vmap(lambda pole: ratintn(numerator, z - pole, z)[0])(poles),
        f,
    )
    numerator_vjp = numerator_pullback(jnp.linspace(0.5, 1.5, poles.size, dtype=dtype))[0]

    assert jnp.all(jnp.isfinite(values))
    assert jnp.all(jnp.isfinite(tangents))
    assert jnp.all(jnp.isfinite(reverse))
    assert jnp.all(jnp.isfinite(numerator_vjp))

    tolerance = 2.0e-4 if dtype == jnp.float32 else 2.0e-10
    np.testing.assert_allclose(tangents, reverse, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", DTYPES)
def test_node_values_and_derivatives_are_continuous(dtype):
    """The analytic node value agrees with representable points on both sides."""

    z = jnp.linspace(-4.0, 4.0, 17, dtype=dtype)
    f = _smooth_numerator(z)
    nodes = z[1:-1]
    offset = 32.0 * jnp.finfo(dtype).eps * jnp.maximum(1.0, jnp.abs(nodes))

    def value_and_tangent(pole):
        return jvp(
            lambda moving_pole: ratintn(f, z - moving_pole, z)[0],
            (pole,),
            (jnp.ones_like(pole),),
        )

    center_values, center_tangents = vmap(value_and_tangent)(nodes)
    left_values, left_tangents = vmap(value_and_tangent)(nodes - offset)
    right_values, right_tangents = vmap(value_and_tangent)(nodes + offset)

    if dtype == jnp.float32:
        rtol, atol = 3.0e-4, 3.0e-5
    else:
        rtol, atol = 2.0e-10, 2.0e-11

    np.testing.assert_allclose(left_values, center_values, rtol=rtol, atol=atol)
    np.testing.assert_allclose(right_values, center_values, rtol=rtol, atol=atol)
    np.testing.assert_allclose(left_tangents, center_tangents, rtol=rtol, atol=atol)
    np.testing.assert_allclose(right_tangents, center_tangents, rtol=rtol, atol=atol)


@pytest.mark.parametrize("kind", ["plasma_dispersion", "plasma_dispersion_derivative"])
def test_grid_convergence_against_maxwellian_plasma_dispersion(kind):
    """Converge to the analytic real Maxwellian Z or Z-prime principal value."""

    maximum_errors = []
    for size in (65, 129, 257):
        z = jnp.linspace(-8.0, 8.0, size)
        spacing = z[1] - z[0]
        poles = jnp.asarray([0.0, 0.5 * spacing, 0.37, 1.2])

        if kind == "plasma_dispersion":
            numerator = jnp.exp(-(z**2)) / jnp.sqrt(jnp.pi)
            reference = -2.0 * dawsn(np.asarray(poles))
        else:
            numerator = -2.0 * z * jnp.exp(-(z**2)) / jnp.sqrt(jnp.pi)
            reference = -2.0 * (1.0 - 2.0 * np.asarray(poles) * dawsn(np.asarray(poles)))

        values = vmap(lambda pole: ratintn(numerator, z - pole, z)[0])(poles)
        maximum_errors.append(float(np.max(np.abs(np.asarray(values) - reference))))

    assert maximum_errors[1] < maximum_errors[0]
    assert maximum_errors[2] < maximum_errors[1]
    assert maximum_errors[2] < 2.0e-6


@pytest.mark.parametrize("dtype", DTYPES)
def test_operator_matches_direct_quadrature_at_nodes_and_midpoints(dtype):
    """The precomputed linear map uses the identical complete, node-safe rule."""

    z = jnp.linspace(-4.0, 4.0, 65, dtype=dtype)
    f = _smooth_numerator(z)
    poles = jnp.concatenate((z[1:-1], 0.5 * (z[:-1] + z[1:])))
    denominators = z[None, :] - poles[:, None]

    direct = vmap(ratintn, in_axes=(None, 0, None))(f, denominators, z)[:, 0]
    operator = ratintn_operator(denominators, z)
    matrix_values = operator @ f

    tolerance = 3.0e-5 if dtype == jnp.float32 else 2.0e-12
    assert jnp.all(jnp.isfinite(operator))
    np.testing.assert_allclose(matrix_values, direct, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("location", ["node", "midpoint"])
def test_forward_and_reverse_modes_agree_at_singular_locations(dtype, location):
    """Keep the forward-mode and transposed derivative paths robust from #125."""

    z = jnp.linspace(-3.0, 3.0, 33, dtype=dtype)
    f = _smooth_numerator(z)
    pole = z[17] if location == "node" else 0.5 * (z[17] + z[18])
    denominator = z - pole

    forward_g = jacfwd(lambda g: ratintn(f, g, z))(denominator)
    reverse_g = jacrev(lambda g: ratintn(f, g, z))(denominator)
    forward_f = jacfwd(lambda numerator: ratintn(numerator, denominator, z))(f)
    reverse_f = jacrev(lambda numerator: ratintn(numerator, denominator, z))(f)

    assert jnp.all(jnp.isfinite(forward_g))
    assert jnp.all(jnp.isfinite(reverse_g))
    assert jnp.all(jnp.isfinite(forward_f))
    assert jnp.all(jnp.isfinite(reverse_f))

    tolerance = 3.0e-4 if dtype == jnp.float32 else 2.0e-10
    np.testing.assert_allclose(forward_g, reverse_g, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(forward_f, reverse_f, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("bounds", "shift", "branch"),
    [
        ((0.0, 1.0), 2.0, "exact"),
        ((-1.0e-3, 1.0e-3), 10.0, "taylor"),
    ],
)
def test_rational_fallback_has_correct_jvp_and_vjp(dtype, bounds, shift, branch):
    """Cover both custom-JVP branches with an affine root outside the domain."""

    z = jnp.linspace(*bounds, 33, dtype=dtype)
    numerator = jnp.ones_like(z)
    shift = jnp.asarray(shift, dtype=dtype)
    denominator = z + shift

    interval_ratio = jnp.abs(jnp.diff(denominator)) / jnp.abs(
        0.5 * (denominator[1:] + denominator[:-1])
    )
    if branch == "taylor":
        assert jnp.all(interval_ratio < 1.0e-4)
    else:
        assert jnp.all(interval_ratio >= 1.0e-4)

    def shifted_integral(moving_shift):
        return ratintn(numerator, z + moving_shift, z)[0]

    value, forward_shift = jvp(shifted_integral, (shift,), (jnp.ones_like(shift),))
    _, denominator_pullback = vjp(
        lambda moving_denominator: ratintn(numerator, moving_denominator, z)[0],
        denominator,
    )
    denominator_vjp = denominator_pullback(jnp.ones((), dtype=dtype))[0]
    reverse_shift = jnp.sum(denominator_vjp)

    lower, upper = bounds
    expected_value = np.log((shift + upper) / (shift + lower))
    expected_shift = 1.0 / (shift + upper) - 1.0 / (shift + lower)

    assert jnp.all(jnp.isfinite(denominator_vjp))
    assert denominator_vjp[0] != 0.0
    assert denominator_vjp[-1] != 0.0

    tolerance = 5.0e-4 if dtype == jnp.float32 else 2.0e-10
    np.testing.assert_allclose(value, expected_value, rtol=tolerance, atol=tolerance * abs(expected_value))
    np.testing.assert_allclose(
        forward_shift,
        expected_shift,
        rtol=tolerance,
        atol=tolerance * abs(expected_shift),
    )
    np.testing.assert_allclose(reverse_shift, forward_shift, rtol=tolerance, atol=tolerance * abs(expected_shift))


def test_principal_value_path_is_exactly_linear_in_numerator():
    """Singularity subtraction and spline interpolation preserve the operator structure."""

    z = jnp.linspace(-3.0, 3.0, 65)
    f = _smooth_numerator(z)
    other = jnp.cos(0.4 * z) * jnp.exp(-0.1 * z**2)
    denominator = z - z[31]
    alpha, beta = 2.5, -0.75

    np.testing.assert_array_equal(ratintn(jnp.zeros_like(f), denominator, z), jnp.zeros(1))
    np.testing.assert_allclose(
        ratintn(alpha * f + beta * other, denominator, z),
        alpha * ratintn(f, denominator, z) + beta * ratintn(other, denominator, z),
        rtol=2.0e-12,
        atol=2.0e-13,
    )


def test_vmap_and_jit_are_transparent():
    """The production batching and compilation transforms do not change the result."""

    z = jnp.linspace(-3.0, 3.0, 65)
    f = _smooth_numerator(z)
    poles = jnp.asarray([z[12], -0.37, 0.5 * (z[40] + z[41]), z[52]])
    denominators = z[None, :] - poles[:, None]

    batched = vmap(ratintn, in_axes=(None, 0, None))(f, denominators, z)
    looped = jnp.stack([ratintn(f, denominator, z) for denominator in denominators])
    np.testing.assert_allclose(batched, looped, rtol=2.0e-12, atol=2.0e-13)

    for denominator in denominators:
        np.testing.assert_allclose(
            jit(ratintn)(f, denominator, z),
            ratintn(f, denominator, z),
            rtol=0.0,
            atol=0.0,
        )
