from jax import numpy as jnp


def ratintn(f: jnp.ndarray, g: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """
    Integrate f / g dz taking each to be piecwise linear.This is more accurate when f / g has a near - pole in an
    interval f, g and z are 1D complex arrays.

    Based on newlip routine by Ed Williams.
    Args:
        f (jnp.ndarray): 1D complex array representing the numerator of the rational function to be integrated.
        g (jnp.ndarray): 1D complex array representing the denominator of the rational function.
        z (jnp.ndarray): 1D complex array representing the variable of integration.
    Returns:
        jnp.ndarray: The integrated values of f / g over z.
    """

    if len(jnp.shape(f)) == 1:
        f = jnp.transpose(f[..., jnp.newaxis])

    zdif = z[1:-1] - z[0:-2]
    out = jnp.sum(ratcen(f, g) * zdif, 1)
    return out


def ratcen(f: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    """
    Return "rationally centered" f / g such that int_s(1) ^ s(0) ds f(s) / g(s) = sum(ratcen(f, g) * s(dif)) when
    f and g are linear functions of s.
    This allows accurate integration through near poles of f / g

    Based on newlip routine by Ed Williams.
    Args:
        f (jnp.ndarray): 2D complex array (shape: [batch, N]) representing the numerator values of the rational function.
        g (jnp.ndarray): 1D complex array (shape: [N]) representing the denominator values of the rational function.
    Returns:
        jnp.ndarray: 2D real array (shape: [batch, N-2]) containing the rationally centered values for integration.

    """

    fdif = f[:, 1:-1] - f[:, 0:-2]
    gdif = g[1:-1] - g[0:-2]
    fav = 0.5 * (f[:, 1:-1] + f[:, 0:-2])
    gav = 0.5 * (g[1:-1] + g[0:-2])

    tmp = fav * gdif - gav * fdif
    rf = fav / gav + tmp * gdif / (12.0 * gav**3)

    rfn = fdif / gdif + tmp * jnp.log((gav + (0.5 + 0j) * gdif) / (gav - 0.5 * gdif)) / gdif**2

    out = jnp.where((jnp.abs(gdif) < 1.0e-4 * jnp.abs(gav))[None, :], rf, rfn)
    return jnp.real(out)
