from jax.scipy.special import gamma
from jax import numpy as jnp
from jax import vmap
from jax.scipy.special import sph_harm
from tsadar.misc.vector_tools import rotate
from interpax import interp2d


def sph_harm_dist(Nl, Nm, vr, flm):
    phi = 0.0
    vx = jnp.concatenate([-vr[::-1], vr])
    vx, vy = jnp.meshgrid(vx, vx)
    th = jnp.arctan2(vy, vx)
    vr_vxvy = jnp.sqrt(vx**2 + vy**2)
    fvxvy = jnp.zeros(jnp.shape(vx))

    for i in range(Nl + 1):
        for j in range(i + 1):
            _flmvxvy = jnp.interp(vr_vxvy, vr, flm[i][j], right=1e-16)
            _sph_harm = vmap(sph_harm, in_axes=(None, None, None, 0, None))(
                jnp.array([j]), jnp.array([i]), phi, th.reshape(-1, order="C"), 2
            ).reshape(jnp.shape(vx), order="C")
            fvxvy += _flmvxvy * jnp.real(_sph_harm)

    return (vx, vy), fvxvy


# we will probably want to add input checks to ensure the proper fields are defined
def DLM_1D(m, h):
    """
    Produces a 1-D Dum-Langdon-Matte distribution parametrized by a super-gaussian order m.

    Args:
        m: (int) Super-Gaussian order
        h: (int) resolution of normalized velocity grid, i.e. spacing of the grid

    Returns:
        vx: normalized velocity grid
        fe_num: numerical distribution function
    """

    def SG(vx, m):
        x0 = jnp.sqrt(3 * gamma(3 / m) / gamma(5 / m))
        return jnp.exp(-((jnp.abs(vx) / x0) ** m))

    vx = jnp.arange(-8, 8, h)
    fe_num = jnp.array([trapz(SG(jnp.sqrt(vx**2 + vz**2), m), h) for vz in vx])

    # x0 = jnp.sqrt(3 * gamma(3 / m) / gamma(5 / m))
    # fe_num = jnp.exp(-((jnp.abs(vx) / x0) ** m))
    fe_num = fe_num / trapz(fe_num, h)
    return vx, fe_num


def SG_1D(m, h):
    """
    Produces a 1-D Super-Gaussian distribution parametrized by a super-gaussian order m.

    Args:
        m: (int) Super-Gaussian order
        h: (int) resolution of normalized velocity grid, i.e. spacing of the grid

    Returns:
        vx: normalized velocity grid
        fe_num: numerical distribution function
    """
    vx = jnp.arange(-8, 8, h)
    x0 = jnp.sqrt(3 * gamma(3 / m) / gamma(5 / m))
    fe_num = jnp.exp(-((jnp.abs(vx) / x0) ** m))
    fe_num = fe_num / trapz(fe_num, h)
    return vx, fe_num


# Warning: These super-gaussian orders do not follow Matte
def DLM_2D(m, h):
    """
    Produces a 2-D symmetric Dum-Langdon-Matte distribution parametrized by a super-gaussian order m.

    Args:
        m: (int) Super-Gaussian order
        h: (int) resolution of normalized velocity grid, i.e. spacing of the grid

    Returns:
        (vx, vy): tuple of the normalized velocity grids in x and y
        fe_num: numerical distribution function
    """
    vx = jnp.arange(-8, 8, h)
    vy = jnp.arange(-8, 8, h)
    vx, vy = jnp.meshgrid(vx, vy)
    x0 = jnp.sqrt(3 * gamma(3 / m) / gamma(5 / m))
    fe_num = jnp.exp(-((jnp.sqrt(vx**2 + vy**2) / x0) ** m))
    fe_num = fe_num / trapz(trapz(fe_num, h), h)
    return (vx, vy), fe_num


# Warning: These super-gaussian orders do not follow Matte
def BiDLM(mx, my, tasym, theta, h):
    """
    Produces a 2-D Dum-Langdon-Matte distribution that can have different widths and super-gaussian orders in the 2
    dimensions.

    Args:
        mx: (int) Super-Gaussian order for the x direction
        my: (int) Super-Gaussian order for the y direction
        tasym: (int) Temperature asymetry, where the y direction will have an effective temperature of Te*tasym. x
        direction will have an effective temperature of Te.
        theta: (int) counter-clockwise rotation of the distribution in radians
        h: (int) resolution of normalized velocity grid, i.e. spacing of the grid

    Returns:
        (vx, vy): tuple of the normalized velocity grids in x and y
        fe_num: numerical distribution function
    """
    vx = jnp.arange(-8, 8, h)
    vy = jnp.arange(-8, 8, h)
    vx, vy = jnp.meshgrid(vx, vy)
    x0x = jnp.sqrt(3 * gamma(3 / mx) / gamma(5 / mx))
    x0y = jnp.sqrt(3 * gamma(3 / my) / gamma(5 / my))
    fe_num = jnp.exp(-((jnp.abs(vx) / x0x) ** mx) - (jnp.abs(vy) / (x0y * jnp.sqrt(tasym))) ** my)
    fe_num = rotate(fe_num, theta)
    # fe_num = fe_num / calc_moment(fe_num,(vx,vy),0)

    renorm = jnp.sqrt(
        calc_moment(fe_num, (vx, vy), 2) / (2 * calc_moment(fe_num, (vx, vy), 0))
    )  # the 2 is to make the moment equal the number of dimensions, not sure on this
    h2 = h / renorm
    # vx2 = jnp.arange(-8/renorm, 8/renorm, h2)
    vx2 = vx[0] / renorm
    vy2 = vx[0] / renorm
    # vy2 = jnp.arange(-8/renorm, 8/renorm, h2)
    # print(jnp.shape(fe_num))
    # print(jnp.shape(vx2))
    # print(h2)
    # print(jnp.shape(jnp.log(fe_num)))
    fe_num = jnp.exp(
        interp2d(vx.flatten(), vy.flatten(), vx2, vy2, jnp.log(fe_num), extrap=[-100, -100], method="linear").reshape(
            jnp.shape(vx), order="F"
        )
    )
    fe_num = fe_num / calc_moment(fe_num, (vx, vy), 0)

    return (vx, vy), fe_num


# not positive on the normalizations for f1 vs f0 so dt may not be =lambda_ei/LT
def Spitzer_3V(dt, vq, h):
    """
    Produces a 2-D Spitzer-Harm distribution with the f1 direction given in 3-space.

    Args:
        dt: (int) Knudsen number determining the magnitude of the perturbation
        vq: array or list with 3 elements giving the direction of the f1 perturbation in x,y,z
        h: (int) resolution of normalized velocity grid, i.e. spacing of the grid

    Returns:
        (vx, vy): tuple of the normalized velocity grids in x and y
        fe_num: numerical distribution function
    """
    # likely to OOM (probably a shortcut by calculating the anlge out of the plane and multiplying f1 by cos of that angle)
    x = jnp.arange(-8, 8, h)
    y = jnp.arange(-8, 8, h)
    z = jnp.arange(-8, 8, h)
    vx, vy, vz = jnp.meshgrid(x, y, z)
    # vq = vq/jnp.sqrt(vq[0]**2 + vq[1]**2 + vq[2]**2)
    f0 = 1 / (2 * jnp.pi) ** (3 / 2) * jnp.exp(-(vx**2 + vy**2 + vz**2) / 2)
    f1 = (
        dt
        * jnp.sqrt(2 / (9 * jnp.pi))
        * (vx * vq[0] + vy * vq[1] + vz * vq[2]) ** 4
        * (4 - (vx * vq[0] + vy * vq[1] + vz * vq[2]) / 2)
        * f0
    )
    fe_num = f0 + f1
    fe_num = trapz(fe_num, h)  # integrate over z
    fe_num = fe_num / trapz(trapz(fe_num, h), h)  # renormalize

    # redefine to coordinates
    vx, vy = jnp.meshgrid(x, y)

    return (vx, vy), fe_num


def Spitzer_2V(dt, vq, h):
    """
    Produces a 2-D Spitzer-Harm distribution with the f1 direction given in the plane.

    Args:
        dt: (float) Knudsen number determining the magnitude of the perturbation
        vq: array or list with 2 elements giving the direction of the f1 perturbation in x,y
        h: (float) resolution of normalized velocity grid, i.e. spacing of the grid

    Returns:
        (vx, vy): tuple of the normalized velocity grids in x and y
        fe_num: numerical distribution function
    """
    x = jnp.arange(-8, 8, h)
    y = jnp.arange(-8, 8, h)
    vx, vy = jnp.meshgrid(x, y)
    # vq = vq/jnp.sqrt(vq[0]**2 + vq[1]**2)
    f0 = 1 / (2 * jnp.pi) ** (3 / 2) * jnp.exp(-(vx**2 + vy**2) / 2)
    f1 = dt * jnp.sqrt(2 / (9 * jnp.pi)) * (vx * vq[0] + vy * vq[1]) ** 4 * (4 - (vx * vq[0] + vy * vq[1]) / 2) * f0
    fe_num = f0 + f1
    fe_num = fe_num / trapz(trapz(fe_num, h), h)  # renormalize

    return (vx, vy), fe_num


def calc_moment(f, v, m):
    """
    Calculates the moment of the distribtuion function specified by m

    Args:
        f: function to calculate the moment of
        m: moment 0, 1, or 2
        v: velocity grid

    Returns:
        moment_val: value of the mth moment
    """
    # print(jnp.shape(f))
    # print(jnp.shape(v))
    if len(jnp.shape(f)) == 1:
        moment_val = trapz(v**m * f, v[1] - v[0])
    elif len(jnp.shape(f)) == 2:
        moment_val = trapz(
            trapz((v[0] ** 2 + v[1] ** 2) ** (m / 2) * f, v[0][0][1] - v[0][0][0]), v[1][1][0] - v[1][0][0]
        )

    return moment_val


def trapz(y, dx):
    """
    JAX compatible trapizoidal intergration.

    Args:
        y: numerical array to be integrated
        dx: spacing of the associated x-axis

    Returns:
        z: integral of ydx
    """
    return 0.5 * (dx * (y[..., 1:] + y[..., :-1])).sum(-1)


# def MoraYahi_3V(dt, vq, m, h)
