"""A 3-V spherical-harmonic hypothesis reduced to the ARTS2D 2-V marginal.

The angular functions are real (cosine) spherical harmonics evaluated after embedding
physical velocity as ``(X, Y, Z) = (vy, vz, vx)``. Consequently JAX's angles are
``theta = acos(vx / |v|)`` (polar) and ``phi = atan2(vz, vy)`` (azimuth); on the
``vz=0`` plane, ``phi`` is zero or pi according to the sign of ``vy``. JAX uses the argument order
``sph_harm_y(degree=l, order=m, theta, phi)``.  For ``m > 0`` we use the conventional
real basis ``sqrt(2) * (-1)**m * Re(Y_l^m)``.  In particular, the ``(l, m) = (1, 0)``
and ``(1, 1)`` modes are proportional to ``vx / |v|`` and ``vy / |v|`` respectively.

This is a 3-D spherical-harmonic angular basis, not a 2-D polar Fourier basis. The
positive 3-V distribution is integrated over ``vz`` before ARTS2D sees it; a central
slice is never substituted for that marginal.
"""
from typing import Dict, Callable
from collections import defaultdict
from functools import partial

from jax import numpy as jnp, vmap, Array
from jax.nn import sigmoid, relu
from jax.random import PRNGKey
from jax.scipy.special import gamma, sph_harm_y
import equinox as eqx
from numpy.polynomial.legendre import leggauss

from .base import DistributionFunction2V, smooth1d


def _clip_unit_interval(value: Array) -> Array:
    """Project a normalized scalar into its physical closed interval."""

    return jnp.clip(value, 0.0, 1.0)


class FLM_NN(eqx.Module):
    """
    A neural network module for modeling spherical harmonics coefficients (FLM) as a function of the radial velocity `vr`. This module uses two separate MLPs to predict the magnitude and sign of the FLM coefficients, combining them to produce the final output.

    Attributes:
        
        flm_mag (eqx.nn.MLP): MLP that predicts the (logarithmic) magnitude of the FLM coefficients.
        flm_sign (eqx.nn.MLP): MLP that predicts the sign of the FLM coefficients.
        vr (Array): Radial velocity array over which the FLM coefficients are evaluated.
    
    Args:
        
        vr (Array): The radial velocity array.
    
    Methods:
        
        __call__(**kwargs):
            
            Computes the FLM coefficients for the given input.
            Args:
                
                f00 (float or Array): The normalization factor for the magnitude.
            
            Returns:
                
                flm (Array): The computed FLM coefficients as a function of `vr`.
    
    """
    flm_mag: eqx.nn.MLP
    flm_sign: eqx.nn.MLP
    vr: Array

    def __init__(self, vr):
        super().__init__()
        self.flm_mag = eqx.nn.MLP(1, 1, 32, 3, final_activation=relu, key=PRNGKey(0))
        self.flm_sign = eqx.nn.MLP(1, 1, 32, 3, final_activation=jnp.tanh, key=PRNGKey(42))
        self.vr = vr

    def __call__(self, **kwargs):
        f00 = kwargs["f00"]
        flm_mag = -vmap(self.flm_mag)(self.vr[:, None])[:, 0]  # from minus inf to 0
        flm_mag = jnp.power(10.0, flm_mag)  # from 0 to 1
        flm_mag *= f00  # from 0 to f00
        flm_sign = vmap(self.flm_sign)(self.vr[:, None])[:, 0]
        flm = flm_mag * flm_sign
        return flm


class FLM_MY(eqx.Module):
    """
    Compute the first-order Legendre moment (FLM) of a distribution function.

    This module uses the Mora & Yahi (1982) model for thermal heat-flux reduction
    in laser-produced plasmas.

    Attributes:
        vr (Array): Array of velocity values (normalized to thermal velocity).
        dt (Array): Trainable scaling factor applied to the FLM coefficient.

    References:
        Mora, P. & Yahi, H. (1982). Thermal heat-flux reduction in laser-produced plasmas.
        Phys. Rev. A 26, 2259–2261.
    """
    vr: Array
    dt: Array

    def __init__(self, vr: Array, dt: float):
        super().__init__()
        self.vr = vr
        # Equinox treats Python scalars as static metadata.  Coefficients must be JAX
        # array leaves so partitioning and autodiff can expose them to the optimizer.
        self.dt = jnp.asarray(dt)

    def __call__(self, **kwargs):
        m_f0 = kwargs["m_f0"]
        f00 = kwargs["f00"]

        # Uses eq. 3 from
        # Mora, P. & Yahi, H. Thermal heat-flux reduction in laser-produced plasmas. Phys. Rev. A 26, 2259–2261 (1982).
        # v0 = 1.0  # distributions are normalized to vth anyway
        # lambda_e = (
        #     1.0  # this is the thermal mean free path but really, it is just normalizing the gradient scale lengths.
        # )
        # # So as long as the gradient scale lengths are provided in units of mean free path and just set this to 1.
        # ve = gamma(5.0 / m_f0) / 3 / gamma(3.0 / m_f0) * v0

        # uu = self.vr / v0
        # lambda_v = lambda_e * (self.vr / ve) ** 4.0
        # coeff = (
        #     m_f0 / 2 * uu**m_f0 - 5 * m_f0 / 12 * gamma(8 / m_f0) / gamma(6 / m_f0) * uu ** (m_f0 - 2) - 1.5
        # ) * lambda_v

        uu = self.vr *jnp.sqrt(gamma(5.0 / m_f0) / 3 / gamma(3.0 / m_f0))
        coeff = (
            m_f0 / 2 * uu**m_f0 - 5 * m_f0 / 12 * gamma(8 / m_f0) / gamma(6 / m_f0) * uu ** (m_f0 - 2) - 1.5
        ) * (self.vr) ** 4.0

        return coeff * self.dt * f00


class ArbitraryVr(eqx.Module):
    """
    ArbitraryVr is a model for generating numerical radial functions for the spherical harmonics to produce a 2D distribution function.
    
    Attributes:
        smooth (Callable): A function that applies 1D smoothing to input arrays, parameterized by window size.
        flm_sign (Array): Learnable parameters representing the sign component of the function, initialized to zeros.
        flm_mag (Array): Learnable parameters representing the magnitude component of the function, initialized to zeros.
    Args:
        nvr (int): The number of radial velocity grid points, used to set the size of parameters and smoothing window.
    Methods:
        __call__(**kwargs):
            Computes the radial function by applying smoothing, nonlinearities (tanh and sigmoid), and scaling.
            Returns the resulting array representing the function values.
    """
    smooth: Callable
    flm_sign: Array
    flm_mag: Array

    def __init__(self, nvr):
        super().__init__()
        self.smooth = partial(smooth1d, window_size=nvr // 4)
        self.flm_sign = jnp.zeros(nvr)
        self.flm_mag = jnp.zeros(nvr)

    def __call__(self, **kwargs):
        flm_sign = jnp.tanh(self.smooth(self.flm_sign))
        flm_mag = -sigmoid(self.smooth(self.flm_mag)) * 10
        # This module parameterizes the harmonic coefficient relative to f00. Keeping
        # the ratio bounded makes it suitable for the positive log-anisotropy model.
        flm = 10**flm_mag * flm_sign * kwargs["f00"]

        return flm


class SphericalHarmonics(DistributionFunction2V):
    """Represent a positive 3-V harmonic hypothesis through its Cartesian 2-V marginal.

    The full velocity distribution uses a truncated real spherical-harmonic
    log-density perturbation with neural-network, Mora-Yahi, or arbitrary radial
    coefficients. It is normalized in three velocities and integrated over the
    unobserved ``vz`` coordinate before being returned to ARTS2D.
    Attributes:
        vr (Array): Radial velocity grid.
        polar_theta (Array): JAX polar angle measured from the physical ``vx`` axis.
        azimuth_phi (Array): JAX azimuth in the embedded ``(vy, 0)`` plane.
        vr_vxvy (Array): Radial grid in (vx, vy) coordinates.
        vz (Array): Gauss-Legendre nodes for the unobserved normalized velocity.
        vz_weights (Array): Corresponding finite-interval quadrature weights.
        Nl (int): Maximum degree of the spherical-harmonic expansion.
        flm (Dict[int, Dict[int, Callable]]): Spherical-harmonic radial coefficients.
        m_scale (float): Scaling factor for the 'm' parameter.
        m_shift (float): Shift for the 'm' parameter.
        act_fun (Callable): Activation function for 'm' parameter normalization.
        normed_m (Array): Normalized 'm' parameter, defining the super-gaussian order for the f0 term.
        flm_type (str): Type of parameterization for spherical harmonics coefficients.
    Args:
        dist_cfg (dict): Configuration dictionary containing distribution parameters.
    Methods:
        get_unnormed_params():
            Returns the unnormalized spherical harmonics coefficients and parameters.
        get_unnormed_m():
            Returns the unnormalized 'm' parameter (Super-gaussian order) for the distribution.
        get_f00():
            Computes the isotropic 3-V radial component.
        __call__():
            Returns the normalized Cartesian 2-V marginal on the ``(vy, vx)`` grid.
    Raises:
        NotImplementedError: If an unsupported 'flm_type' or spherical harmonics index is requested.
    """
    vr: Array
    polar_theta: Array
    azimuth_phi: Array
    vr_vxvy: Array
    vz: Array
    vz_weights: Array
    Nl: int
    flm: Dict[int, Dict[int, Callable]]
    m_scale: float
    m_shift: float
    act_fun: Callable
    normed_m: Array
    flm_type: str
    anisotropy_log_limit: float

    def __init__(self, dist_cfg):
        super().__init__(dist_cfg)

        # The radial functions belong to a 3-V hypothesis. Cover the corners of the
        # Cartesian integration box, with a small interpolation guard beyond them.
        vmax = 6.0 * 1.05 * jnp.sqrt(3.0)
        dvr = vmax / dist_cfg["params"]["nvr"]
        self.vr = jnp.linspace(dvr / 2, vmax - dvr / 2, dist_cfg["params"]["nvr"])

        vx, vy = jnp.meshgrid(self.vx, self.vx)
        radius = jnp.hypot(vx, vy)
        safe_radius = jnp.where(radius > 0, radius, 1.0)
        self.polar_theta = jnp.where(
            radius > 0,
            jnp.arccos(jnp.clip(vx / safe_radius, -1.0, 1.0)),
            0.0,
        )
        self.azimuth_phi = jnp.where(vy < 0, jnp.pi, 0.0)
        self.vr_vxvy = jnp.sqrt(vx**2 + vy**2)
        nvz = int(dist_cfg["params"].get("nvz", max(64, dist_cfg["params"]["nvr"])))
        if nvz < 2:
            raise ValueError("nvz must contain at least two quadrature nodes")
        vz_nodes, vz_weights = leggauss(nvz)
        self.vz = jnp.asarray(6.0 * vz_nodes)
        self.vz_weights = jnp.asarray(6.0 * vz_weights)
        self.Nl = dist_cfg["params"]["Nl"]
        self.anisotropy_log_limit = float(dist_cfg["params"].get("anisotropy_log_limit", 8.0))
        if self.anisotropy_log_limit <= 0:
            raise ValueError("anisotropy_log_limit must be positive")

        self.flm = defaultdict(dict)

        init_m = dist_cfg["params"]["init_m"]

        self.m_scale = 3.0
        self.m_shift = 2.0
        self.act_fun = _clip_unit_interval
        initial_fraction = (init_m - self.m_shift) / self.m_scale
        if not 0.0 <= initial_fraction <= 1.0:
            raise ValueError("init_m must lie in the closed interval [2, 5]")
        # Store the normalized physical fraction directly. JAX gives ``clip`` its
        # centered boundary subgradient, so exact endpoints remain trainable toward
        # the feasible interior rather than being frozen at an infinite logit.
        self.normed_m = jnp.asarray(initial_fraction)

        self.flm[0][0] = self.get_f00()
        self.flm_type = dist_cfg["params"]["flm_type"]

        for i in range(1, self.Nl + 1):
            for j in range(i + 1):
                if self.flm_type.casefold() == "nn":
                    self.flm[i][j] = FLM_NN(self.vr)

                elif self.flm_type.casefold() == "mora-yahi":
                    if i == 1 and j == 0:
                        self.flm[i][j] = FLM_MY(self.vr, dist_cfg["params"]["dtx"])
                    elif i == 1 and j == 1:
                        self.flm[i][j] = FLM_MY(self.vr, dist_cfg["params"]["dty"])
                    else:
                        raise NotImplementedError("Mora-Yahi only supports l=1, m=0 and l=1, m=1")

                elif self.flm_type.casefold() == "arbitrary":
                    self.flm[i][j] = ArbitraryVr(dist_cfg["params"]["nvr"])

                else:
                    raise NotImplementedError(f"Unknown flm_type: {dist_cfg['params']['flm_type']}")

    def get_unnormed_params(self):
        """
        Computes and returns the unnormalized parameters for the spherical harmonics distribution.
        This method constructs a dictionary of spherical harmonics coefficients (`flm_dict`) up to order `self.Nl`.
        The zeroth order coefficient (f00) is obtained from `self.get_f00()`. For higher orders, the coefficients
        are computed using the corresponding functions in `self.flm`, with keyword arguments including the
        unnormalized moment (`m_f0`) and the zeroth order coefficient (`f00`).
        Returns:
            dict: The physical shape parameter under ``m`` and a nested ``flm``
                dictionary indexed first by degree and then by order.
        """
        flm_dict = {l: {} for l in range(self.Nl + 1)}
        flm_dict[0][0] = self.get_f00()
        kwargs = {"m_f0": self.get_unnormed_m(), "f00": flm_dict[0][0]}
        for i in range(1, self.Nl + 1):
            for j in range(i + 1):
                flm_dict[i][j] = self.flm[i][j](**kwargs)

        return {"m": self.get_unnormed_m(), "flm": flm_dict}

    @staticmethod
    def _real_harmonic_at(
        degree: int,
        order: int,
        polar_theta: Array,
        azimuth_phi: Array,
        radius: Array,
    ) -> Array:
        """Evaluate one real spherical harmonic at broadcast-compatible angles.

        ``jax.scipy.special.sph_harm_y`` requires degree before order in current
        JAX releases.  Vectorizing scalar angle pairs avoids relying on its unusual
        broadcasting rules while keeping ``degree`` and ``order`` compile-time values.
        Every anisotropic harmonic is defined to vanish at the coordinate singularity;
        the tolerance also catches the roundoff-sized nominal origin of odd grids.
        """

        polar_theta, azimuth_phi, radius = jnp.broadcast_arrays(
            polar_theta, azimuth_phi, radius
        )
        values = vmap(sph_harm_y, in_axes=(None, None, 0, 0))(
            jnp.asarray([degree]),
            jnp.asarray([order]),
            polar_theta.reshape(-1, order="C"),
            azimuth_phi.reshape(-1, order="C"),
        ).reshape(polar_theta.shape, order="C")
        if order == 0:
            real_values = jnp.real(values)
        else:
            real_values = jnp.sqrt(2.0) * (-1) ** order * jnp.real(values)

        if degree > 0:
            coordinate_scale = jnp.maximum(jnp.max(jnp.abs(radius)), 1.0)
            origin_tolerance = 32.0 * jnp.finfo(radius.dtype).eps * coordinate_scale
            real_values = jnp.where(radius <= origin_tolerance, 0.0, real_values)
        return real_values

    def _real_harmonic(self, degree: int, order: int) -> Array:
        """Evaluate one projected real harmonic on the observed velocity plane."""

        return self._real_harmonic_at(
            degree,
            order,
            self.polar_theta,
            self.azimuth_phi,
            self.vr_vxvy,
        )

    def get_unnormed_m(self):
        """Returns the unnormalized (physical) super-Gaussian shape parameter "m" for the f00 component."""
        return self.act_fun(self.normed_m) * self.m_scale + self.m_shift

    def get_f00(self):
        """Compute the exactly normalized isotropic radial 3-V component.

        Returns:
            Values of ``f00`` on the radial grid. The analytic normalization is
            three-dimensional and does not depend on radial-grid quadrature.
        """
        return self._f00_at_radius(self.vr)

    def _f00_at_radius(self, radius: Array) -> Array:
        """Evaluate the exactly normalized isotropic 3-V super-Gaussian.

        Velocities are normalized by ``vTe = sqrt(Te / me)``. The scale is chosen so
        ``integral f3 d3v = 1`` and ``<vx^2 + vy^2 + vz^2> = 3`` for every shape
        parameter, hence the isotropic 2-V marginal always has in-plane second moment 2.
        """

        shape = self.get_unnormed_m()
        v0 = jnp.sqrt(3.0 * gamma(3.0 / shape) / gamma(5.0 / shape))
        normalization = shape / (4.0 * jnp.pi * v0**3 * gamma(3.0 / shape))
        return normalization * jnp.exp(-((radius / v0) ** shape))

    def _evaluate_3d_unnormalized(self, vx: Array, vy: Array, vz: Array) -> Array:
        """Evaluate the positive 3-V hypothesis before its density normalization."""

        vx, vy, vz = jnp.broadcast_arrays(vx, vy, vz)
        radius = jnp.sqrt(vx**2 + vy**2 + vz**2)
        safe_radius = jnp.where(radius > 0, radius, 1.0)
        polar_theta = jnp.where(
            radius > 0,
            jnp.arccos(jnp.clip(vx / safe_radius, -1.0, 1.0)),
            0.0,
        )
        azimuth_phi = jnp.arctan2(vz, vy)

        f00_radial = self.get_f00()
        log_anisotropy = jnp.zeros_like(radius)
        kwargs = {"m_f0": self.get_unnormed_m(), "f00": f00_radial}
        # The VJP of division contains the square of its denominator. Flooring at
        # sqrt(tiny), rather than tiny, prevents that square from underflowing when
        # high-order super-Gaussian tails have already rounded f00 to zero.
        safe_f00 = jnp.maximum(
            f00_radial,
            jnp.sqrt(jnp.asarray(jnp.finfo(f00_radial.dtype).tiny)),
        )
        for degree in range(1, self.Nl + 1):
            for order in range(degree + 1):
                relative_radial_coefficient = self.flm[degree][order](**kwargs) / safe_f00
                relative_coefficient = jnp.interp(
                    radius.reshape(-1),
                    self.vr,
                    relative_radial_coefficient,
                    left=0.0,
                    right=0.0,
                ).reshape(radius.shape)
                log_anisotropy += relative_coefficient * self._real_harmonic_at(
                    degree, order, polar_theta, azimuth_phi, radius
                )

        # A bounded log-density perturbation is smooth, strictly positive, linear in
        # each harmonic coefficient near zero, and cannot overflow in sparsely sampled
        # tails. Density normalization is applied to the 3-V model before marginalizing.
        bounded_log_anisotropy = self.anisotropy_log_limit * jnp.tanh(
            log_anisotropy / self.anisotropy_log_limit
        )
        return self._f00_at_radius(radius) * jnp.exp(bounded_log_anisotropy)

    def _quadrature_3d(self) -> Array:
        """Evaluate the unnormalized 3-V hypothesis on the marginal quadrature grid."""

        return self._evaluate_3d_unnormalized(
            self.vx[None, None, :],
            self.vx[None, :, None],
            self.vz[:, None, None],
        )

    def get_3d_distribution(self) -> Array:
        """Return the normalized 3-V hypothesis on ``(vz, vy, vx)`` quadrature nodes."""

        f3 = self._quadrature_3d()
        dv = self.vx[1] - self.vx[0]
        normalization = jnp.sum(f3 * self.vz_weights[:, None, None]) * dv**2
        return f3 / normalization

    def __call__(self):
        """
        Build a positive 3-V harmonic hypothesis and return its out-of-plane marginal.
        Returns:
            jnp.ndarray: The normalized Cartesian marginal ``integral f3 dvz`` on
                the ``(vy, vx)`` grid.
        Notes:
            - The full 3-V hypothesis is normalized before marginalization. There is no
              central-slice substitution, hard positivity clip, or post-hoc 2-V rescaling.
        """

        f3 = self.get_3d_distribution()
        return jnp.sum(f3 * self.vz_weights[:, None, None], axis=0)
