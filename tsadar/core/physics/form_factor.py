"""The FormFactor class: calculates the collisionless Thomson scattering structure factor / spectral
density function S(k, omega) for a given plasma condition, scattering geometry, and electron distribution
function, in both 1D (calc_chi_vals-based) and 2D (calc_in_2D) forms."""
from jax import numpy as jnp, vmap, device_put, device_count, devices
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

import scipy.interpolate as sp
from functools import partial, lru_cache

import os
import numpy as np
from interpax import interp2d, interp1d
from jax.lax import cond, scan, stop_gradient, map as jmap
from jax import checkpoint

from . import ratintn
from .interpolation import interp_uniform
from ...utils.vector_tools import vsub, vdot, vdiv

BASE_FILES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "external")

# Angles in the tabulated projection used by the 2D form factor (see
# FormFactor._build_sinogram). Single source of truth: the config layer defers to this
# rather than repeating the number.
DEFAULT_N_BETA = 1024

# Angles rotated per batch when building the sinogram. Bounds peak memory, which scales as
# BETA_BATCH_SIZE * nvx**2; sized for the nvx=128 of the shipped 2D decks. Shipped configs
# range nvx from 32 to 320, so the top end is ~6x this footprint.
BETA_BATCH_SIZE = 32


def _charge_weighted_flow(Z, fract, flow):
    """Return the charge-weighted ion flow, preserving a singleton species axis.

    ``ud`` is defined relative to the ion fluid. For a multispecies plasma the
    order-independent ion-fluid velocity is
    ``sum_s(Z_s * fract_s * flow_s) / sum_s(Z_s * fract_s)``.

    ``flow`` may be either an array (the 1-D path) or a tuple of arrays containing
    Cartesian components (the 2-D path). The species axis is always the last axis.
    """

    weights = Z * fract
    weight_sum = jnp.sum(weights, axis=-1, keepdims=True)

    def average(component):
        return jnp.sum(weights * component, axis=-1, keepdims=True) / weight_sum

    if isinstance(flow, tuple):
        return tuple(average(component) for component in flow)
    return average(flow)


def _electron_resonance(k, omega, electron_flow, vTe):
    """Return ``(beta, xi, |k|)`` for the longitudinal electron response.

    The projection direction is fixed by ``k`` alone. Flow enters only through the
    signed scalar resonance coordinate, so a perpendicular flow cannot rotate the
    Radon projection and ``xi == 0`` does not require a special angular convention.
    """

    k_mag = jnp.sqrt(vdot(k, k))
    k_hat = vdiv(k, k_mag)
    beta = jnp.atan2(k_hat[1], k_hat[0])
    xi = (omega / k_mag - vdot(electron_flow, k_hat)) / vTe
    return beta, xi, k_mag


def _principal_value_integral(df, vx, xi):
    """Evaluate ``PV integral df(v) / (v - xi) dv`` with a finite node limit.

    ``ratintn`` is retained away from an exact grid-node pole. At a node its two
    logarithmic endpoint contributions are individually infinite, producing ``nan``
    before their principal-value cancellation. The symmetric limit of the same
    quadrature avoids that undefined intermediate. Because the piecewise-linear
    interpolant has no finite derivative at a knot for a general sampled EDF, the
    exact-node tangent is defined as the centered slope across one velocity cell.
    This gives the optimizer a finite, grid-scale smoothing convention that does
    not depend on floating-point precision.
    """

    denominator = vx - xi
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(vx)))
    pole_tol = 16 * jnp.finfo(vx.dtype).eps * scale
    at_grid_node = jnp.min(jnp.abs(denominator)) <= pole_tol

    def symmetric_limit(_):
        spacing = jnp.abs(vx[1] - vx[0])
        xi_fixed = stop_gradient(xi)

        def evaluate(pole):
            return jnp.squeeze(ratintn.ratintn(df, vx - pole, vx))

        # Use a fixed fraction of a cell for the value limit so float32 and float64
        # follow the same numerical convention. The O(offset**2) symmetric error is
        # negligible compared with the underlying grid discretization.
        value_offset = 1.0e-3 * spacing
        value = 0.5 * (
            evaluate(xi_fixed - value_offset) + evaluate(xi_fixed + value_offset)
        )

        # A half-cell displacement lands between neighboring knots. Replacing only
        # the xi tangent leaves derivatives through `df` and `vx` untouched.
        half_cell = 0.5 * spacing
        xi_slope = (evaluate(xi_fixed + half_cell) - evaluate(xi_fixed - half_cell)) / (
            2 * half_cell
        )
        return value + xi_slope * (xi - xi_fixed)

    return cond(
        at_grid_node,
        symmetric_limit,
        lambda _: jnp.squeeze(ratintn.ratintn(df, denominator, vx)),
        operand=None,
    )


@lru_cache(maxsize=1)
def _load_zprime_tables():
    rdWT = np.vstack(np.loadtxt(os.path.join(BASE_FILES_PATH, "files", "rdWT.txt")))
    idWT = np.vstack(np.loadtxt(os.path.join(BASE_FILES_PATH, "files", "idWT.txt")))
    return rdWT, idWT


def zprimeMaxw(xi):
    """
    Calculates the derivative of the plasma dispersion function (Z-prime) for an array of normalized phase velocities (xi)
    using a combination of tabulated values and asymptotic approximations.
    For values of xi between -10 and 10, the function uses interpolated data from precomputed tables. For values outside
    this range, it applies the asymptotic approximation as described in Eqn. 5.2.10 of the Thomson scattering reference.
    Args:
        xi (np.ndarray): Array of normalized phase velocities (must be in ascending order).
    Returns:
        Zp (np.ndarray): 2D array where the first row contains the real components and the second row contains the imaginary
        components of Z-prime evaluated at each value of xi.
    """

    rdWT, idWT = _load_zprime_tables()

    ai = xi < -10
    bi = xi > 10

    rinterp = sp.interp1d(rdWT[:, 0], rdWT[:, 1], "linear")
    rZp = np.concatenate((xi[ai] ** -2, rinterp(xi), xi[bi] ** -2))
    iinterp = sp.interp1d(idWT[:, 0], idWT[:, 1], "linear")
    iZp = np.concatenate((0 * xi[ai], iinterp(xi), 0 * xi[bi]))

    Zp = np.vstack((rZp, iZp))
    return Zp


class FormFactor:
    """
    FormFactor class for calculating the Thomson scattering structure factor or spectral density function.
    This class encapsulates all static values and methods required for repeated calculations of the Thomson
    scattering structure factor or spectral density function, supporting both 1D and 2D electron distribution
    functions (EDFs), multiple plasma conditions, and scattering angles.
    Args:
        lambda_range (list): Starting and ending wavelengths over which to calculate the spectrum.
        npts (int): Number of wavelength points to use in the calculation.
        lam_shift (float): Wavelength shift to apply.
        scattering_angles (dict): Dictionary containing scattering angles (in degrees).
        num_grad_points (int): Number of gradient points for plasma parameter profiles.
        ud_ang (float): Angle between electron drift and x-axis (degrees).
        va_ang (float): Angle between ion flow and x-axis (degrees).
        n_beta (int): Number of angles at which the 2D EDF's projection is tabulated. Trades accuracy for
            speed in the 2D calculation; 0 disables the tabulation and rotates the EDF exactly at every
            evaluation point instead.
    Attributes:
        C (float): Speed of light in cm/s.
        Me (float): Electron mass in keV/C^2.
        Mp (float): Proton mass in keV/C^2.
        npts (int): Number of wavelength points.
        h (float): Step size for velocity grid.
        xi1, xi2 (jnp.ndarray): Grids for velocity integration.
        Zpi (jnp.ndarray): Precomputed plasma dispersion function values.
        chiERratprim_op (jnp.ndarray): Precomputed constant matrix for the 1D dispersion relation
            integral, applied to the distribution function derivative on each forward pass.
        lam_shift (float): Wavelength shift.
        scattering_angles (dict): Scattering angles.
        num_grad_points (int): Number of gradient points.
        vmap_calc_chi_vals (callable): Vectorized susceptibility calculation.
        ud_angle, va_angle (float): Electron drift and ion flow angles.
        calc_all_chi_vals (callable): Method for calculating susceptibility, possibly parallelized.
    Methods:
        __call__(params):
            Calculates the standard collisionless Thomson spectral density function S(k,omg) for 1D EDFs.
                params (dict): Plasma and distribution function parameters.
                formfactor (jnp.ndarray): Calculated spectrum.
                lams (jnp.ndarray): Wavelength axis.
        rotate(vx, df, angle, reshape=False):
            Rotates a 2D array by a given angle in radians.
                vx (jnp.ndarray): Velocity grid.
                df (jnp.ndarray): 2D distribution function.
                angle (float): Rotation angle in radians.
                reshape (bool): Whether to reshape the output.
                jnp.ndarray: Rotated/interpolated 2D array.
        project(vx, DF, beta):
            Projects the 2D distribution function onto the direction beta (its Radon transform).
                vx (jnp.ndarray): Velocity grid.
                DF (jnp.ndarray): 2D distribution function.
                beta (float): Angle in radians.
                jnp.ndarray: 1D projected distribution function.
        scan_calc_chi_vals(carry, xs):
            Calculates susceptibility values at a given point in the distribution function using scan.
                carry (tuple): (velocity grid, sinogram as returned by _build_sinogram).
                xs (tuple): (angle, signed_xi_at, klde_mag_at).
                tuple: Updated carry and (fe_vphi, chiEI, chiERrat).
        calc_chi_vals(vx, sinogram, inputs):
            Calculates susceptibility values at a given point in the distribution function.
                vx (jnp.ndarray): Velocity grid.
                sinogram (tuple): Tabulated projection, or the 2D distribution function when n_beta is 0.
                inputs (tuple): (angle, signed_xi_at, klde_mag_at).
                tuple: (fe_vphi, chiEI, chiERrat).
        _calc_all_chi_vals_(vx, DF, beta, xi, klde_mag):
            Calculates susceptibility values for all desired points xie (batch or vectorized).
                vx (jnp.ndarray): Velocity grid.
                DF (jnp.ndarray): 2D distribution function.
                beta (jnp.ndarray): Angles.
                xi (jnp.ndarray): Signed normalized resonance coordinates.
                klde_mag (jnp.ndarray): Magnitudes of wavevector times Debye length.
                tuple: (fe_vphi, chiEI, chiERrat).
        parallel_calc_all_chi_vals(x, DF, beta, xi, klde_mag):
            Parallelized calculation of susceptibility values across devices.
                x (jnp.ndarray): Velocity grid.
                DF (jnp.ndarray): 2D distribution function.
                beta, xi, klde_mag (jnp.ndarray): Parameters for susceptibility calculation.
                tuple: (fe_vphi, chiEI, chiERrat).
        calc_in_2D(params):
            Calculates the collisionless Thomson spectral density function S(k,omg) for a 2D numerical EDF.
                params (dict): Plasma and distribution function parameters.
                formfactor (jnp.ndarray): Calculated spectrum.
                lams (jnp.ndarray): Wavelength axis.
    """
    def __init__(
        self,
        lambda_range,
        npts,
        lam_shift,
        scattering_angles,
        num_grad_points,
        ud_ang,
        va_ang,
        calc_gain,
        n_beta=DEFAULT_N_BETA,
    ):

        # basic quantities
        self.C = 2.99792458e10
        self.Me = 510.9896 / self.C**2  # electron mass keV/C^2
        self.Mp = self.Me * 1836.1  # proton mass keV/C^2
        # self.lambda_range = lambda_range
        self.npts = npts
        self.h = 0.01
        minmax = 8.2
        h1 = 1024  # 1024
        lamAxis = jnp.linspace(lambda_range[0], lambda_range[1], npts)
        self.omgL_num = 2 * jnp.pi * 1e7 * self.C
        omgs = 2e7 * jnp.pi * self.C / lamAxis  # Scattered frequency axis(1 / sec)
        self.omgs = omgs[None, ..., None, None]  # [1, npts, 1, 1]

        self.xi1 = jnp.linspace(-minmax - jnp.sqrt(2.0) / h1, minmax + jnp.sqrt(2.0) / h1, h1)
        self.xi2 = jnp.array(jnp.arange(-minmax, minmax, self.h))
        self.Zpi = jnp.array(zprimeMaxw(self.xi2))

        # `ratintn` is exactly linear in its first argument and, in the 1D path, the other two are
        # built from the fixed grids above. The quadrature is therefore a constant matrix applied to
        # `ratdf`, which is the only thing that varies across evaluations. Build it once.
        self.chiERratprim_op = ratintn.ratintn_operator(self.xi1[None, :] - self.xi2[:, None], self.xi1)
        self.lam_shift = lam_shift
        self.scattering_angles = scattering_angles
        self.num_grad_points = num_grad_points

        self.vmap_calc_chi_vals = vmap(checkpoint(self.calc_chi_vals), in_axes=(None, None, 0), out_axes=0)
        self.ud_angle, self.va_angle = ud_ang, va_ang

        # Number of angles in the tabulated projection (see `_build_sinogram`). A falsy
        # value selects the exact per-point rotation instead, which is what the
        # tabulation is validated against.
        self.n_beta = int(n_beta) if n_beta else 0
        if 0 < self.n_beta < 4:
            # The Catmull-Rom stencil is 4 wide, so a shorter grid wraps onto itself and
            # returns quietly wrong numbers rather than failing.
            raise ValueError(
                f"n_beta must be 0 (exact rotation) or at least 4, got {self.n_beta}. "
                "Values in 1-3 are too short for the interpolation stencil."
            )

        #option to include calculation of SBS and SRS gain
        self.calc_gain = calc_gain

        # Create a Sharding object to distribute a value across devices:
        is_gpu_present = any(["gpu" == device.platform for device in devices()])
        self.calc_all_chi_vals = self._calc_all_chi_vals_

        if is_gpu_present:
            num_gpus = device_count(backend="gpu")
            if num_gpus > 1:
                print(
                    f"If this is a 2D Angular calculation, it will be parallelized across {num_gpus} GPUs. Otherwise, only a single GPU is used"
                )
                mesh = Mesh(devices=mesh_utils.create_device_mesh((device_count(backend="gpu"),)), axis_names=("x"))
                self.sharding = NamedSharding(mesh, P("x"))
                self.calc_all_chi_vals = self.parallel_calc_all_chi_vals
            else:
                self.calc_all_chi_vals = self._calc_all_chi_vals_

    def __call__(self, params):
        """
        Calculates the standard collisionless Thomson spectral density function S(k,omg) and is capable of handling
        multiple plasma conditions and scattering angles. Distribution functions can be arbitrary as calculations of the
        susceptibility is done on-the-fly. Calculations are done in 4 dimension with the following shape,
        [number of gradient-points, number of wavelength points, number of angles, number of ion-species].

        In angular, `fe` is a Tuple, Distribution function (DF), normalized velocity (x), and angles from k_L to f1 in
        radians

        Args:
            params: ThomsonParams object, contains all the parameters from the input deck

        Returns:
            formfactor: array of the calculated spectrum, has the shape [number of gradient-points, number of
                wavelength points, number of angles]
            lams: wavelength axis
        """

        ne = (
            1.0e20
            * params["electron"]["ne"]
            * jnp.linspace(
                (1 - params["general"]["ne_gradient"] / 200),
                (1 + params["general"]["ne_gradient"] / 200),
                self.num_grad_points,
            )
        )[:, None, None, None]  # [ng, 1, 1, 1]
        Te = (params["electron"]["Te"] * jnp.linspace(
            (1 - params["general"]["Te_gradient"] / 200),
            (1 + params["general"]["Te_gradient"] / 200),
            self.num_grad_points,
        ))[:, None, None, None]  # [ng, 1, 1, 1]
        lam = params["general"]["lam"] + self.lam_shift
        A = jnp.array([params[species]["A"] for species in params.keys() if "ion" in species])[None, None, None, :]  # [1, 1, 1, ns]
        Z = jnp.array([params[species]["Z"] for species in params.keys() if "ion" in species])[None, None, None, :]  # [1, 1, 1, ns]
        Ti = jnp.array([params[species]["Ti"] for species in params.keys() if "ion" in species])[None, None, None, :]  # [1, 1, 1, ns]
        Va = jnp.array([params[species]["Va"] for species in params.keys() if "ion" in species])[None, None, None, :] * 1.0e6  # [1, 1, 1, ns]
        fract = jnp.array([params[species]["fract"] for species in params.keys() if "ion" in species])[None, None, None, :]  # [1, 1, 1, ns]
        ud = params["general"]["ud"] * 1.0e6  # drift velocity in cm/s
        fe = params["electron"]["fe"]
        vx = params["electron"]["v"]

        Mi = A * self.Mp  # ion mass [1, 1, 1, ns]
        re = 2.8179e-13  # classical electron radius cm
        Esq = self.Me * self.C**2 * re  # sq of the electron charge keV cm
        constants = jnp.sqrt(4 * jnp.pi * Esq / self.Me)
        sarad = self.scattering_angles["sa"] * jnp.pi / 180  # scattering angle in radians
        sarad = jnp.reshape(sarad, [1, 1, -1, 1])  # [1, 1, na, 1]
        omgL = self.omgL_num / lam  # laser frequency Rad / s

        # calculate k and omega vectors
        omgpe = constants * jnp.sqrt(ne)  # plasma frequency Rad/cm, [ng, 1, 1, 1]
        omg = self.omgs - omgL

        ks = jnp.sqrt(self.omgs**2 - omgpe**2) / self.C
        kL = jnp.sqrt(omgL**2 - omgpe**2) / self.C
        k = jnp.sqrt(ks**2 + kL**2 - 2 * ks * kL * jnp.cos(sarad))

        ion_omgdop = omg - k * Va

        # plasma parameters
        # electrons
        vTe = jnp.sqrt(Te / self.Me)  # electron thermal velocity, [ng, 1, 1, 1]
        klde = (vTe / omgpe) * k

        # ions
        Zbar = jnp.sum(Z * fract)
        ni = fract * ne / Zbar
        omgpi = constants * Z * jnp.sqrt(ni * self.Me / Mi)
        vTi = jnp.sqrt(Ti / Mi)  # ion thermal velocity, [1, 1, 1, ns]
        kldi = (vTi / omgpi) * k

        # ion susceptibilities
        # finding derivative of plasma dispersion function along xii array
        xii = ion_omgdop / (jnp.sqrt(2.0) * vTi * k)

        # num_ion_pts = jnp.shape(xii)
        # chiI = jnp.zeros(num_ion_pts)
        ZpiR = interp_uniform(xii, self.xi2, self.Zpi[0, :], left=xii**-2, right=xii**-2)
        ZpiI = interp_uniform(xii, self.xi2, self.Zpi[1, :], left=0, right=0)
        #chiI = jnp.sum(-0.5 / (kldi**2) * (ZpiR + 1j * ZpiI), 3)
        chiI = -0.5 / (kldi**2) * (ZpiR + 1j * ZpiI) 

        # `ud` is relative to the charge-weighted ion fluid. Convert it to the
        # absolute electron flow before evaluating the electron resonance. Unlike the
        # previous ion-1 reference, this is invariant to species ordering.
        ion_bulk_flow = _charge_weighted_flow(Z, fract, Va)
        electron_flow = ion_bulk_flow + ud
        electron_omgdop = omg - k * electron_flow
        xie = electron_omgdop / (k * vTe)

        #fe_vphi = jnp.exp(jnp.interp(xie, vx, jnp.log(fe)))
        fe_vphi=jnp.exp(jnp.apply_along_axis(interp1d,0,jnp.squeeze(xie),vx,jnp.log(jnp.squeeze(fe)),extrap=[-50, -50])).reshape(jnp.shape(xie))

        df = jnp.diff(fe_vphi, 1, 1) / jnp.diff(xie, 1, 1)
        df = jnp.append(df, jnp.zeros((len(ne), 1, len(self.scattering_angles["sa"]),1)), 1) 

        chiEI = -jnp.pi / (klde**2) * 1j * df
        
        ratmod = jnp.exp(interp1d(self.xi1, vx, jnp.log(fe), extrap=[-50, -50]))
        ratdf = jnp.gradient(ratmod, self.xi1[1] - self.xi1[0])

        # xi2 = jnp.squeeze(self.xi2 - 1j*(10*Zbar*Esq*omgpe**2)/(self.Me*vTe**3))
        chiERratprim = self.chiERratprim_op @ ratdf
        chiERrat = jnp.reshape(interp_uniform(xie.flatten(), self.xi2, chiERratprim), xie.shape)
        chiERrat = -1.0 / (klde**2) * chiERrat

        chiE = chiERrat + chiEI
        chiI = jnp.sum(chiI, 3) # Sum over ion species to get total ion susceptibility
        chiI = chiI[..., jnp.newaxis]
        epsilon = 1.0 + chiE + chiI

        # This line needs to be changed if ion distribution is changed!!!
        ion_comp_fact = fract * Z**2 / Zbar / vTi
        #ion_comp_fact = fract * Zbar / vTi
        ion_comp = ion_comp_fact * (
            (jnp.abs(chiE)) ** 2.0 * jnp.exp(-(xii**2)) / jnp.sqrt(2 * jnp.pi)
        )

        ele_comp = (jnp.abs(1.0 + chiI)) ** 2.0 * fe_vphi / vTe

        SKW_ion_omg = 1.0 / k * ion_comp / ((jnp.abs(epsilon)) ** 2)

        SKW_ion_omg = jnp.sum(SKW_ion_omg, 3)
        SKW_ion_omg = SKW_ion_omg[..., jnp.newaxis]
        SKW_ele_omg = 1.0 / k * (ele_comp) / ((jnp.abs(epsilon)) ** 2)

        PsOmg = (SKW_ion_omg + SKW_ele_omg) * (1 + 2 * omg / omgL) * re**2.0 * ne
        # PsOmg = jnp.squeeze(PsOmg, axis=-1)
        lams = 2 * jnp.pi * self.C / self.omgs
        PsLam = PsOmg * 2 * jnp.pi * self.C / lams**2
        formfactor = jnp.squeeze(PsLam, axis=-1)

        if self.calc_gain['calc']:
            Ipump = self.calc_gain['Ipump']*1e14  # Convert to W/cm^2
            beam_diam_cm = self.calc_gain['beam_diam_um'] * 1e-4  # Convert um to cm
            # interaction_length_cm = jnp.linspace(0,1,8).reshape(1,1,1,8)*beam_diam_cm/jnp.sin(sarad[...,np.newaxis]) # effective interaction length cm
            interaction_length_cm = beam_diam_cm/2.0/jnp.sin(sarad) 

            nc = 1.115e21/(lam*1e-3)**2
            ne_nc = ne/nc

            a0 = 8.55e-4 * lam*1e-9 * jnp.sqrt(Ipump)
            j0 = a0**2 / jnp.sqrt(1-ne_nc)
            
            Fchi = chiE * (1.0 + chiI) / (1.0 + chiE + chiI)
          
            GD = (k**2)/4/ks * j0 * -jnp.imag(Fchi)
            GDl = jnp.mean(GD * interaction_length_cm, axis=-1)
            # formfactor = jnp.sum(formfactor[...,jnp.newaxis] * jnp.exp(GDl), axis=-1)
            formfactor = formfactor * jnp.exp(GDl)


        return formfactor, jnp.squeeze(lams, axis=-1)

    def rotate(self, vx, df, angle, reshape: bool = False) -> jnp.ndarray:
        """
        Rotate a 2D array by a specified angle in radians. This method rotates the input 2D array `df` using a rotation matrix constructed from the given angle. The rotation is performed around the origin, and the rotated coordinates are interpolated back onto the original grid using cubic interpolation.

            vx (jnp.ndarray): 1D array representing the grid points along each axis.
            df (jnp.ndarray): 2D array to be rotated.
            angle (float): Rotation angle in radians (counterclockwise).
            reshape (bool, optional): Whether to reshape the output array. Defaults to False.
        
        Returns:
            
            jnp.ndarray: The rotated and interpolated 2D array.
        
        """

        rad_angle = jnp.deg2rad(-angle)
        cos_angle = jnp.cos(rad_angle)
        sin_angle = jnp.sin(rad_angle)
        rotation_matrix = jnp.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
        _vx, _vy = jnp.meshgrid(vx, vx)
        coords = jnp.stack((_vx.flatten(), _vy.flatten()))
        rotated_coords = jnp.einsum("ij, ik->kj", rotation_matrix, coords)
        xq = rotated_coords[:, 0]
        yq = rotated_coords[:, 1]

        return interp2d(xq, yq, vx, vx, df, extrap=True, method="cubic").reshape((vx.size, vx.size), order="F")

    def project(self, vx, DF, beta):
        """
        Project the 2D distribution function onto the direction `beta`.

        This is the only thing `rotate` was ever used for: the caller immediately
        collapses the rotated 2D array along axis 0, so the full rotation is discarded
        and only this 1D line-integral (the Radon transform of `DF` at angle `beta`)
        survives.

        Args:

            vx: normalized velocity grid
            DF: 2D array, distribution function
            beta: angle of the k-vector from the x-axis, in radians

        Returns:

            1D array, the distribution function projected onto `beta`

        """
        dvx = vx[1] - vx[0]
        # ``rotate`` uses the image-rotation sign convention, whereas ``beta`` is the
        # mathematical direction (cos(beta), sin(beta)). Negate the angle so the
        # projected coordinate increases along that direction.
        return jnp.sum(checkpoint(self.rotate)(vx, DF, -beta * 180 / jnp.pi, reshape=False), axis=0) * dvx

    def _build_sinogram(self, vx, DF):
        """
        Tabulate the projection and its velocity-derivative over a uniform grid of angles.

        The projection depends on nothing but the angle, so it does not have to be
        recomputed for every point at which the susceptibility is evaluated. There are
        `num_grad_points x npts x n_angles` such points -- of order 1e5 for an ATS
        geometry -- against `n_beta` distinct angles here, so tabulating once and
        interpolating removes essentially all of the 2D interpolation work.

        The grid spans a full period and the projection is 2*pi-periodic in the angle, so
        `_interp_beta` can wrap exactly rather than extrapolating at the seam.

        Note that the derivative is taken here, before the interpolation in angle, rather
        than after it as the per-point path does. Interpolating in the angle is a linear
        combination of rows and `jnp.gradient` is linear, so the two orders agree exactly;
        doing it here costs `n_beta` gradients instead of one per evaluation point.

        Args:

            vx: normalized velocity grid
            DF: 2D array, distribution function

        Returns:

            proj: (n_beta, len(vx)) array, projected distribution function
            dproj: (n_beta, len(vx)) array, its derivative along vx

        """
        betas = jnp.linspace(0.0, 2.0 * jnp.pi, self.n_beta, endpoint=False)
        # `jmap` rather than `vmap` so peak memory stays bounded by the batch, not by
        # n_beta: each rotation materializes an intermediate of len(vx)**2 before it is
        # collapsed to a row of `proj`.
        proj = jmap(partial(self.project, vx, DF), xs=betas, batch_size=BETA_BATCH_SIZE)
        dproj = vmap(lambda p: jnp.gradient(p, vx[1] - vx[0]))(proj)
        return proj, dproj

    def _interp_beta(self, beta, table):
        """
        Periodic cubic interpolation of a tabulated quantity at the angle `beta`.

        The angle grid is uniform and covers exactly one period, so the bracketing index
        is plain arithmetic instead of a search, and the wrap at 2*pi is exact. `beta` as
        built by `calc_in_2D` lands in (-pi/2, 3*pi/2), which the modulo folds back onto
        the grid.

        Catmull-Rom rather than linear specifically because this is differentiated. A
        linear interpolant has a piecewise-*constant* derivative in the angle, so while
        its values converge at O(h**2) its gradient converges only at O(h): measured
        against an exact rotation at n_beta=1024, `d/dDF` came out ~4e-2 in relative L2,
        far too coarse to fit against. Catmull-Rom is C1 and brings that to ~7e-3 in
        relative L2 on the same EDF, for one extra pair of gathers. (The descent
        *direction* is much better than either figure -- 1 - cos ~ 3e-5 for the cubic --
        but L2 is the honest number to quote for the magnitude.)

        Args:

            beta: angle in radians
            table: (n_beta, len(vx)) array tabulated on the angle grid

        Returns:

            1D array, `table` interpolated to `beta`

        """
        idx, w = self._beta_stencil(beta)
        return sum(w[j] * table[idx[j]] for j in range(4))

    def _beta_stencil(self, beta):
        """
        Catmull-Rom stencil (4 wrapped row indices and their weights) for the angle `beta`.

        Split out from `_interp_beta` so a caller that only needs a single velocity can
        gather 4x2 scalars instead of 4 whole rows -- see `_interp_beta_v`.
        """
        t = beta * self.n_beta / (2.0 * jnp.pi)
        i0 = jnp.floor(t)
        s = t - i0
        i0 = jnp.mod(i0.astype(jnp.int32), self.n_beta)

        s2, s3 = s * s, s * s * s
        w = (
            -0.5 * s3 + s2 - 0.5 * s,
            1.5 * s3 - 2.5 * s2 + 1.0,
            -1.5 * s3 + 2.0 * s2 + 0.5 * s,
            0.5 * s3 - 0.5 * s2,
        )
        # The stencil straddles the seam near the ends of the grid; the modulo is what
        # makes the wrap exact rather than clamped.
        idx = tuple(jnp.mod(i0 + j - 1, self.n_beta) for j in range(4))
        return idx, w

    def _interp_beta_v(self, beta, vx, v, table):
        """
        Interpolate a tabulated quantity at a single (angle, velocity) point.

        Equivalent to `jnp.interp(v, vx, self._interp_beta(beta, table))` but gathers only
        the 4x2 entries the result depends on, rather than four whole rows of `table`. The
        distinction matters because this runs once per evaluation point, of which there
        are order 1e5: the full-row form moves ~8 kB per point and is memory-bound.

        `vx` is uniform, so the velocity index is arithmetic rather than a search.

        Args:

            beta: angle in radians
            vx: normalized velocity grid (uniform)
            v: velocity at which to evaluate
            table: (n_beta, len(vx)) array tabulated on the angle grid

        Returns:

            scalar, `table` interpolated to (`beta`, `v`)

        """
        idx, w = self._beta_stencil(beta)

        tv = (v - vx[0]) / (vx[1] - vx[0])
        j0 = jnp.clip(jnp.floor(tv).astype(jnp.int32), 0, vx.size - 2)
        wv = jnp.clip(tv - j0, 0.0, 1.0)  # clamp so out-of-range v holds the edge value
        return sum(w[j] * (table[idx[j], j0] * (1.0 - wv) + table[idx[j], j0 + 1] * wv) for j in range(4))

    def scan_calc_chi_vals(self, carry, xs):
        """
        Calculate the values of the susceptibility at a given point in the distribution function

        Args:

            carry: container for

                x: 1D array
                sinogram: tuple of (proj, dproj) as returned by `_build_sinogram`

            xs: container for

                element: angle in radians
                xi_at: signed normalized resonance coordinate
                klde_mag_at: float

        Returns:

            fe_vphi: float, value of the projected distribution function at the point xie
            chiEI: float, value of the imaginary part of the electron susceptibility at the point xie
            chiERrat: float, value of the real part of the electron susceptibility at the point xie

        """
        x, sinogram = carry
        fe_vphi, chiEI, chiERrat = self.calc_chi_vals(x, sinogram, xs)
        return (x, sinogram), (fe_vphi, chiEI, chiERrat)

    def calc_chi_vals(self, vx, sinogram, inputs):
        """
        Calculate the values of the susceptibility at a given point in the distribution function

        Args:

            vx: normalized velocity grid
            sinogram: tuple of (proj, dproj) as returned by `_build_sinogram`, or the 2D
                distribution function itself when `n_beta` is 0, in which case the
                projection is computed exactly at this point's angle
            inputs: container for

                element: angle in radians
                xi_at: signed normalized resonance coordinate
                klde_mag_at: float

        Returns:

            fe_vphi: float, value of the projected distribution function at the point xie
            chiEI: float, value of the imaginary part of the electron susceptibility at the point xie
            chiERrat: float, value of the real part of the electron susceptibility at the point xie

        """
        element, xi_at, klde_mag_at = inputs

        if self.n_beta:
            proj, dproj = sinogram
            # `df` is needed in full because ratintn integrates over the whole grid, but
            # the projection itself is only ever sampled at xie, so it is gathered at that
            # one point rather than as a whole row.
            df = self._interp_beta(element, dproj)
            fe_vphi = self._interp_beta_v(element, vx, xi_at, proj)
        else:
            fe_1D_k = self.project(vx, sinogram, element)
            df = jnp.gradient(fe_1D_k, vx[1] - vx[0])
            # find the location of xie in axis array
            # add the value of fe to the fe container
            fe_vphi = interp_uniform(xi_at, vx, fe_1D_k)

        dfe = interp_uniform(xi_at, vx, df)

        # Chi is really chi evaluated at the points xie
        # so the imaginary part is
        chiEI = -jnp.pi / (klde_mag_at**2) * dfe

        # The real part is the principal-value integral at the signed pole location.
        chiERrat = -1.0 / (klde_mag_at**2) * _principal_value_integral(df, vx, xi_at)
        return fe_vphi, chiEI, chiERrat

    def _calc_all_chi_vals_(self, vx, DF, beta, xi, klde_mag):
        """
        Calculate the susceptibility values for all the desired points xie

        Args:
            
            x: normalized velocity grid
            beta: angle of the k-vector form the x-axis
            DF: 2D array, distribution function
            xi: signed normalized resonance coordinates
            klde_mag: magnitude of the wavevector time debye length where the calculations need to be performed

        Returns:
            
            fe_vphi: projected distribution function
            chiEI: imaginary part of the electron susceptibility
            chiERrat: real part of the electron susceptibility

        """
        calc_chi_vals = "batch_vmap"

        flattened_inputs = (beta.flatten(), xi.flatten(), klde_mag.flatten())

        # Tabulate the projection over angles once, rather than rotating the whole 2D
        # distribution function again at every one of the (many) evaluation points. When
        # `n_beta` is 0 the distribution function is passed through and each point does
        # its own exact rotation, which is the behaviour this replaced.
        df_or_sinogram = self._build_sinogram(vx, jnp.squeeze(DF)) if self.n_beta else jnp.squeeze(DF)

        if calc_chi_vals == "scan":
            _, (fe_vphi, chiEI, chiERrat) = scan(
                self.scan_calc_chi_vals, (vx, df_or_sinogram), flattened_inputs, unroll=1
            )

        elif calc_chi_vals == "vmap":
            fe_vphi, chiEI, chiERrat = self.vmap_calc_chi_vals(vx, df_or_sinogram, flattened_inputs)

        elif calc_chi_vals == "batch_vmap":
            batch_vmap_calc_chi_vals = partial(self.calc_chi_vals, vx, df_or_sinogram)
            fe_vphi, chiEI, chiERrat = jmap(batch_vmap_calc_chi_vals, xs=flattened_inputs, batch_size=128)
        else:
            raise NotImplementedError

        fe_vphi = fe_vphi.reshape(beta.shape)
        chiEI = chiEI.reshape(beta.shape)
        chiERrat = chiERrat.reshape(beta.shape)

        return fe_vphi, chiEI, chiERrat

    def parallel_calc_all_chi_vals(self, x, DF, beta, xi, klde_mag):
        """
        Multi-device counterpart to _calc_all_chi_vals_: flattens beta/xi/klde_mag, distributes them
        across devices via self.sharding (device_put), then delegates to _calc_all_chi_vals_ to compute the
        susceptibility values in parallel before reshaping the results back to the input shape.

        Args:

            x: normalized velocity grid
            DF: 2D array, distribution function
            beta: angle of the k-vector from the x-axis
            xi: signed normalized resonance coordinates
            klde_mag: magnitude of the wavevector times debye length where the calculations need to be performed

        Returns:

            fe_vphi: projected distribution function
            chiEI: imaginary part of the electron susceptibility
            chiERrat: real part of the electron susceptibility

        """
        f_beta = beta.reshape(-1)
        f_xi = xi.reshape(-1)
        f_klde_mag = klde_mag.reshape(-1)

        flat_beta = device_put(f_beta, self.sharding)
        flat_xi = device_put(f_xi, self.sharding)
        flat_klde_mag = device_put(f_klde_mag, self.sharding)

        fe_vphi, chiEI, chiERrat = self._calc_all_chi_vals_(x, DF, flat_beta, flat_xi, flat_klde_mag)

        fe_vphi = fe_vphi.reshape(beta.shape)
        chiEI = chiEI.reshape(beta.shape)
        chiERrat = chiERrat.reshape(beta.shape)

        return fe_vphi, chiEI, chiERrat

    def _ion_flow_angles(self, ion_species):
        """Return per-species 2-D flow angles in the runtime species order."""

        if isinstance(self.va_angle, dict):
            missing = [species for species in ion_species if species not in self.va_angle]
            if missing:
                raise ValueError(f"Missing Va angle for ion species: {', '.join(missing)}")
            return jnp.asarray([self.va_angle[species] for species in ion_species])

        angles = jnp.atleast_1d(jnp.asarray(self.va_angle))
        if angles.size == 1:
            return jnp.broadcast_to(angles, (len(ion_species),))
        if angles.size != len(ion_species):
            raise ValueError(
                f"Expected one Va angle per ion species ({len(ion_species)}), got {angles.size}"
            )
        return angles

    def calc_in_2D(self, params):
        """Calculate the collisionless Thomson spectrum for a 2-D numerical EDF.

        Each ion species has its own flow vector. ``general.ud`` is the electron drift
        relative to their charge-weighted bulk flow, so the electron lab-frame velocity
        is ``sum(Z * fract * Va) / Zbar + ud``. The longitudinal EDF projection is fixed
        by ``k_hat`` and is sampled at a signed resonance coordinate.
        """

        ne = (
            1.0e20
            * params["electron"]["ne"]
            * jnp.linspace(
                1 - params["general"]["ne_gradient"] / 200,
                1 + params["general"]["ne_gradient"] / 200,
                self.num_grad_points,
            )
        )[:, None, None]
        Te = (
            params["electron"]["Te"]
            * jnp.linspace(
                1 - params["general"]["Te_gradient"] / 200,
                1 + params["general"]["Te_gradient"] / 200,
                self.num_grad_points,
            )
        )[:, None, None]
        lam = params["general"]["lam"] + self.lam_shift
        fe = params["electron"]["fe"]
        vx = params["electron"]["v"]

        ion_species = [species for species in params if species.startswith("ion-")]
        A = jnp.asarray([params[species]["A"] for species in ion_species])[None, None, None, :]
        Z = jnp.asarray([params[species]["Z"] for species in ion_species])[None, None, None, :]
        Ti = jnp.asarray([params[species]["Ti"] for species in ion_species])[None, None, None, :]
        fract = jnp.asarray([params[species]["fract"] for species in ion_species])[None, None, None, :]
        Va_mag = jnp.asarray([params[species]["Va"] for species in ion_species]) * 1.0e6

        va_angle = self._ion_flow_angles(ion_species) * jnp.pi / 180
        ion_flow = (
            (Va_mag * jnp.cos(va_angle))[None, None, None, :],
            (Va_mag * jnp.sin(va_angle))[None, None, None, :],
        )
        ud_mag = params["general"]["ud"] * 1.0e6
        ud_angle = self.ud_angle * jnp.pi / 180
        relative_electron_flow = (ud_mag * jnp.cos(ud_angle), ud_mag * jnp.sin(ud_angle))

        Mi = A * self.Mp
        re = 2.8179e-13
        Esq = self.Me * self.C**2 * re
        constants = jnp.sqrt(4 * jnp.pi * Esq / self.Me)

        # Keep the calculation axes explicit: [gradient, wavelength, angle], adding
        # species only as the final axis for ion quantities.
        sarad = self.scattering_angles["sa"][None, None, :] * jnp.pi / 180
        omgL = self.omgL_num / lam
        omgs = self.omgs[..., 0]
        omgpe = constants * jnp.sqrt(ne)
        omg = omgs - omgL

        kL = (jnp.sqrt(omgL**2 - omgpe**2) / self.C, jnp.zeros_like(omgpe))
        ks_mag = jnp.sqrt(omgs**2 - omgpe**2) / self.C
        ks = (jnp.cos(sarad) * ks_mag, jnp.sin(sarad) * ks_mag)
        k = vsub(ks, kL)

        Zbar = jnp.sum(Z * fract, axis=-1, keepdims=True)
        ion_bulk_flow = _charge_weighted_flow(Z, fract, ion_flow)
        electron_flow = (
            ion_bulk_flow[0][..., 0] + relative_electron_flow[0],
            ion_bulk_flow[1][..., 0] + relative_electron_flow[1],
        )

        vTe = jnp.sqrt(Te / self.Me)
        beta, xi, k_mag = _electron_resonance(k, omg, electron_flow, vTe)
        klde_mag = (vTe / omgpe) * k_mag

        # Each ion susceptibility retains its species-specific Doppler shift.
        k_by_species = (k[0][..., None], k[1][..., None])
        ion_omgdop = omg[..., None] - vdot(k_by_species, ion_flow)
        ni = fract * ne[..., None] / Zbar
        omgpi = constants * Z * jnp.sqrt(ni * self.Me / Mi)
        vTi = jnp.sqrt(Ti / Mi)
        kldi = (vTi / omgpi) * k_mag[..., None]
        xii = ion_omgdop / (jnp.sqrt(2.0) * vTi * k_mag[..., None])

        ZpiR = interp_uniform(xii, self.xi2, self.Zpi[0, :], left=xii**-2, right=xii**-2)
        ZpiI = interp_uniform(xii, self.xi2, self.Zpi[1, :], left=0, right=0)
        chiI = jnp.sum(-0.5 / (kldi**2) * (ZpiR + 1j * ZpiI), axis=-1)

        fe_vphi, chiEI, chiERrat = self.calc_all_chi_vals(vx, fe, beta, xi, klde_mag)
        chiE = chiERrat + 1j * chiEI
        epsilon = 1.0 + chiE + chiI

        ion_comp_fact = fract * Z**2 / Zbar / vTi
        ion_comp = (
            ion_comp_fact
            * jnp.abs(chiE[..., None]) ** 2
            * jnp.exp(-(xii**2))
            / jnp.sqrt(2 * jnp.pi)
        )
        ele_comp = jnp.abs(1.0 + chiI) ** 2 * fe_vphi / vTe

        SKW_ion_omg = jnp.sum(
            ion_comp / k_mag[..., None] / jnp.abs(epsilon[..., None]) ** 2,
            axis=-1,
        )
        SKW_ele_omg = ele_comp / k_mag / jnp.abs(epsilon) ** 2

        PsOmg = (
            (SKW_ion_omg + SKW_ele_omg)
            * (1 + 2 * omg / omgL)
            * re**2
            * ne
        )
        lams = 2 * jnp.pi * self.C / self.omgs
        formfactor = PsOmg * 2 * jnp.pi * self.C / lams[..., 0] ** 2

        return formfactor, lams
