from jax import numpy as jnp


def interp_uniform(x: jnp.ndarray, xp: jnp.ndarray, fp: jnp.ndarray, left=None, right=None) -> jnp.ndarray:
    """
    Piecewise linear interpolation of `fp` onto `x`, for a *uniformly spaced* grid `xp`.

    Drop-in replacement for `jnp.interp(x, xp, fp, left=left, right=right)` when `xp` is uniform.
    `jnp.interp` locates every query point with a binary search (`searchsorted`); on a uniform grid
    the index is simply `floor((x - xp[0]) / dx)`, so the search is dropped entirely. Every
    interpolation grid in the form factor is uniform by construction (`linspace` / `arange`), so the
    two agree to floating point association error.

    Args:
        x (jnp.ndarray): Query points, any shape.
        xp (jnp.ndarray): 1D uniformly spaced grid in ascending order.
        fp (jnp.ndarray): 1D array of values sampled on `xp`.
        left: Value to return for `x < xp[0]`, defaults to `fp[0]`. May be an array broadcastable
            against `x`, matching `jnp.interp`.
        right: Value to return for `x > xp[-1]`, defaults to `fp[-1]`. May be an array broadcastable
            against `x`, matching `jnp.interp`.

    Returns:
        jnp.ndarray: Interpolated values with the shape of `x`.
    """

    # averaging over the whole grid rather than taking xp[1] - xp[0] keeps the rounding error in the
    # step from accumulating into the fractional index at the far end of the grid
    dx = (xp[-1] - xp[0]) / (xp.size - 1)
    t = (x - xp[0]) / dx
    # clip before the cast so an out-of-range query can never produce an out-of-range index
    i = jnp.clip(jnp.floor(t), 0, xp.size - 2).astype(jnp.int32)
    w = t - i

    out = fp[i] * (1.0 - w) + fp[i + 1] * w
    out = jnp.where(x < xp[0], fp[0] if left is None else left, out)
    out = jnp.where(x > xp[-1], fp[-1] if right is None else right, out)

    return out
