"""Small, device-verified GPU regressions for the principal-value quadrature."""

import jax
import numpy as np
import pytest
from jax import config, grad, jit, jvp, vjp, numpy as jnp

config.update("jax_enable_x64", True)

from tsadar.core.physics.ratintn import ratintn, ratintn_operator


def _gpu_device():
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if not devices:
        pytest.skip("JAX has no GPU backend")
    return devices[0]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("location", ["node", "midpoint"])
def test_principal_value_primal_jvp_vjp_and_operator_run_on_gpu(dtype, location):
    """Exercise the finite singular paths on a deliberately small accelerator grid."""

    device = _gpu_device()
    with jax.default_device(device):
        z = jnp.linspace(-4.0, 4.0, 17, dtype=dtype)
        numerator = jnp.exp(-0.7 * (z - 0.23) ** 2) * (1.0 + 0.15 * z)
        pole = z[8] if location == "node" else 0.5 * (z[8] + z[9])

        @jit
        def evaluate(f, moving_pole):
            def pole_value(candidate):
                return ratintn(f, z - candidate, z)[0]

            value, forward = jvp(pole_value, (moving_pole,), (jnp.ones_like(moving_pole),))
            reverse = grad(pole_value)(moving_pole)
            _, numerator_pullback = vjp(
                lambda moving_f: ratintn(moving_f, z - moving_pole, z)[0],
                f,
            )
            numerator_vjp = numerator_pullback(jnp.ones((), dtype=dtype))[0]
            operator_value = ratintn_operator(z - moving_pole, z) @ f
            return value, forward, reverse, numerator_vjp, operator_value

        outputs = evaluate(numerator, pole)
        for output in outputs:
            output.block_until_ready()

    assert all(next(iter(output.devices())).platform == "gpu" for output in outputs)
    assert all(jnp.all(jnp.isfinite(output)) for output in outputs)

    tolerance = 3.0e-4 if dtype == jnp.float32 else 2.0e-10
    value, forward, reverse, _, operator_value = outputs
    np.testing.assert_allclose(forward, reverse, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(operator_value, value, rtol=tolerance, atol=tolerance)
