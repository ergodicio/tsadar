"""Small vector/array helpers (component-wise add/sub/dot/div on tuples, dimension expansion, 2D rotation)
used across the physics and distribution-function code."""
from jax import numpy as jnp


def vadd(a, b):
    """
    Custom function for vector addition where a and b are tuples of ND-arrays with the first element being the ND-array
    of x-values and the second element being the ND-array of y-values.

    Args:
        a: tuple of ND-arrays where the first element is x-values and the second is y-values
        b: tuple of ND-arrays where the first element is x-values and the second is y-values

    Returns:
        c: tuple of ND-arrays where the first element is x-values and the second is y-values

    """
    return (a[0] + b[0], a[1] + b[1])


def vsub(a, b):
    """
    Custom function for vector subtraction where a and b are tuples of ND-arrays with the first element being the ND-array
    of x-values and the second element being the ND-array of y-values.

    Args:
        a: tuple of ND-arrays where the first element is x-values and the second is y-values
        b: tuple of ND-arrays where the first element is x-values and the second is y-values

    Returns:
        c: tuple of ND-arrays where the first element is x-values and the second is y-values

    """
    return (a[0] - b[0], a[1] - b[1])


def vdot(a, b):
    """
    Custom function for vector dot product where a and b are tuples of ND-arrays with the first element being the
    ND-array of x-values and the second element being the ND-array of y-values. If on of the vector only has one
    component it is treated as a scalar.

    Args:
        a: tuple of ND-arrays where the first element is x-values and the second is y-values or an ND-array
        b: tuple of ND-arrays where the first element is x-values and the second is y-values or an ND-array

    Returns:
        c: ND-array or tuple of ND-arrays based off the operation being a dot product or scalar product
    """
    if isinstance(a, tuple):
        if isinstance(b, tuple):
            return a[0] * b[0] + a[1] * b[1]
        else:
            return (a[0] * b, a[1] * b)
    else:
        return (a * b[0], a * b[1])


def vdiv(a, b):
    """
    Custom function for division of a vector by a scalar.

    Args:
        a: tuple of ND-arrays where the first element is x-values and the second is y-values
        b: ND-array

    Returns:
        c: tuple of ND-arrays
    """
    # custom function for vector divided by a scalar
    if isinstance(a, tuple):
        if isinstance(b, tuple):
            raise ValueError("vector must be divided by a scalar")
        else:
            return (a[0] / b, a[1] / b)
    else:
        raise ValueError("vector must be divided by a scalar")


def v_add_dim(a):
    """
    Custom function for adding an array dimension to the ND-arrays in vectors.

    Args:
        a: vector to add a dimension to, tuple of ND-arrays where the first element is x-values and the second is
            y-values

    Returns:
        c: vector with additional dimension, tuple of ND-arrays
    """
    return (a[0][..., jnp.newaxis], a[1][..., jnp.newaxis])


# performs a counterclockwise rotation, rotation angle in radians
def rotate(A, theta):
    """
    Rotates a matrix A by an angle theta in the counter-clockwise direction.

    ``A`` is indexed ``[y, x]`` (as ``jnp.meshgrid(vx, vy)`` lays out a 2D distribution function) and is
    rotated about its geometric center ``((n - 1) / 2)`` in each axis, which is the ``v = 0`` point of a
    symmetric cell-centered velocity grid, so an isotropic array is unchanged by any rotation and a
    rotation by a multiple of 90 degrees is an exact permutation of the samples. Values are bilinearly
    interpolated; samples that fall outside the array are zero.

    Args:
        A: Matrix to be rotated
        theta: rotation angle in radians

    Returns:
        A_rotated: rotated Matrix, same shape as A
    """
    ny, nx = A.shape
    center_x, center_y = (nx - 1) / 2, (ny - 1) / 2
    x, y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny))
    # source position of every output pixel: rotate the output grid by -theta about the center
    cos, sin = jnp.cos(-theta), jnp.sin(-theta)
    source_x = cos * (x - center_x) - sin * (y - center_y) + center_x
    source_y = sin * (x - center_x) + cos * (y - center_y) + center_y
    x0, y0 = jnp.floor(source_x), jnp.floor(source_y)
    frac_x, frac_y = source_x - x0, source_y - y0
    x0, y0 = x0.astype(int), y0.astype(int)

    def sample(ix, iy):
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        return jnp.where(inside, A[jnp.clip(iy, 0, ny - 1), jnp.clip(ix, 0, nx - 1)], 0.0)

    A_rotated = (
        (1 - frac_x) * (1 - frac_y) * sample(x0, y0)
        + (1 - frac_x) * frac_y * sample(x0, y0 + 1)
        + frac_x * (1 - frac_y) * sample(x0 + 1, y0)
        + frac_x * frac_y * sample(x0 + 1, y0 + 1)
    )

    return A_rotated
