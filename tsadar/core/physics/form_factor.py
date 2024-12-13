from jax import numpy as jnp, vmap, device_put
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

import scipy.interpolate as sp
from functools import partial

import os
import numpy as np
from interpax import interp2d
from jax.lax import scan, map as jmap
from jax import checkpoint

from . import ratintn
from ...utils.data_handling import lam_parse
from ...utils.vector_tools import vsub, vdot, vdiv

BASE_FILES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "external")


def zprimeMaxw(xi):
    """
    This function calculates the derivative of the Z - function given an array of normalized phase velocities(xi) as
    defined in Chapter 5 of the Thomson scattering book. For values of xi between - 10 and 10 a table is used. Outside
    of this range the asymptotic approximation(see Eqn. 5.2.10) is used.


    Args:
        xi: normalized phase velocities to calculate the zprime function at, these values must be in ascending order

    Returns:
        Zp: array with the real and imaginary components of Z-prime

    """

    rdWT = np.vstack(np.loadtxt(os.path.join(BASE_FILES_PATH, "files", "rdWT.txt")))
    idWT = np.vstack(np.loadtxt(os.path.join(BASE_FILES_PATH, "files", "idWT.txt")))

    ai = xi < -10
    bi = xi > 10

    rinterp = sp.interp1d(rdWT[:, 0], rdWT[:, 1], "linear")
    rZp = np.concatenate((xi[ai] ** -2, rinterp(xi), xi[bi] ** -2))
    iinterp = sp.interp1d(idWT[:, 0], idWT[:, 1], "linear")
    iZp = np.concatenate((0 * xi[ai], iinterp(xi), 0 * xi[bi]))

    Zp = np.vstack((rZp, iZp))
    return Zp


class FormFactor:
    def __init__(self, lambda_range, npts, lam_shift, scattering_angles, num_grad_points):
        """
        Creates a FormFactor object holding all the static values to use for repeated calculations of the Thomson
        scattering structure factor or spectral density function.

        Args:
            lambda_range: list of the starting and ending wavelengths over which to calculate the spectrum.
            npts: number of wavelength points to use in the calculation
            fe_dim: dimension of the electron velocity distribution function (EDF), should be 1 or 2
            vax: (optional) velocity axis coordinates that the 2D EDF is defined on

        Returns:
            Instance of the FormFactor object

        """
        # basic quantities
        self.C = 2.99792458e10
        self.Me = 510.9896 / self.C**2  # electron mass keV/C^2
        self.Mp = self.Me * 1836.1  # proton mass keV/C^2
        # self.lambda_range = lambda_range
        self.npts = npts
        self.h = 0.01
        minmax = 8.2
        h1 = 1024  # 1024
        c = 2.99792458e10
        lamAxis = jnp.linspace(lambda_range[0], lambda_range[1], npts)
        self.omgL_num = 2 * jnp.pi * 1e7 * c
        omgs = 2e7 * jnp.pi * c / lamAxis  # Scattered frequency axis(1 / sec)
        self.omgs = omgs[None, ..., None]

        self.xi1 = jnp.linspace(-minmax - jnp.sqrt(2.0) / h1, minmax + jnp.sqrt(2.0) / h1, h1)
        self.xi2 = jnp.array(jnp.arange(-minmax, minmax, self.h))
        self.Zpi = jnp.array(zprimeMaxw(self.xi2))
        self.lam_shift = lam_shift
        self.scattering_angles = scattering_angles
        self.num_grad_points = num_grad_points

        self.vmap_calc_chi_vals = vmap(checkpoint(self.calc_chi_vals), in_axes=(None, None, 0, 0, 0), out_axes=0)

        # Create a Sharding object to distribute a value across devices:
        # mesh = Mesh(devices=mesh_utils.create_device_mesh(1), axis_names=("x"))
        # self.sharding = NamedSharding(mesh, P("x"))

    def __call__(self, params):
        """
        Calculates the standard collisionless Thomson spectral density function S(k,omg) and is capable of handling
        multiple plasma conditions and scattering angles. Distribution functions can be arbitrary as calculations of the
        susceptibility is done on-the-fly. Calculations are done in 4 dimension with the following shape,
        [number of gradient-points, number of wavelength points, number of angles, number of ion-species].

        In angular, `fe` is a Tuple, Distribution function (DF), normalized velocity (x), and angles from k_L to f1 in
        radians

        Args:
            params: parameter dictionary, must contain the drift 'ud' and flow 'Va' velocities in the 'general' field
            cur_ne: electron density in 1/cm^3 [1 by number of gradient points]
            cur_Te: electron temperature in keV [1 by number of gradient points]
            A: atomic mass [1 by number of ion species]
            Z: ionization state [1 by number of ion species]
            Ti: ion temperature in keV [1 by number of ion species]
            fract: relative ion composition [1 by number of ion species]
            sa: scattering angle in degrees [1 by number of angles]
            f_and_v: a distribution function object, contains the numerical distribution function and its velocity grid

        Returns:
            formfactor: array of the calculated spectrum, has the shape [number of gradient-points, number of
                wavelength points, number of angles]
        """

        ne = (
            1.0e20
            * params["electron"]["ne"]
            * jnp.linspace(
                (1 - params["general"]["ne_gradient"] / 200),
                (1 + params["general"]["ne_gradient"] / 200),
                self.num_grad_points,
            )
        )
        Te = params["electron"]["Te"] * jnp.linspace(
            (1 - params["general"]["Te_gradient"] / 200),
            (1 + params["general"]["Te_gradient"] / 200),
            self.num_grad_points,
        )
        lam = params["general"]["lam"] + self.lam_shift
        A = [params[species]["A"] for species in params.keys() if "ion" in species]
        Z = [params[species]["Z"] for species in params.keys() if "ion" in species]
        Ti = [params[species]["Ti"] for species in params.keys() if "ion" in species]
        fract = [params[species]["fract"] for species in params.keys() if "ion" in species]
        Va = params["general"]["Va"] * 1e6  # flow velocity in 1e6 cm/s
        ud = params["general"]["ud"] * 1e6  # drift velocity in 1e6 cm/s
        fe = params["electron"]["fe"]
        vx = params["electron"]["v"]

        Mi = jnp.array(A) * self.Mp  # ion mass
        re = 2.8179e-13  # classical electron radius cm
        Esq = self.Me * self.C**2 * re  # sq of the electron charge keV cm
        constants = jnp.sqrt(4 * jnp.pi * Esq / self.Me)
        sarad = self.scattering_angles["sa"] * jnp.pi / 180  # scattering angle in radians
        sarad = jnp.reshape(sarad, [1, 1, -1])
        omgL = self.omgL_num / lam  # laser frequency Rad / s

        # calculate k and omega vectors
        omgpe = constants * jnp.sqrt(ne[..., jnp.newaxis, jnp.newaxis])  # plasma frequency Rad/cm
        omg = self.omgs - omgL

        ks = jnp.sqrt(self.omgs**2 - omgpe**2) / self.C
        kL = jnp.sqrt(omgL**2 - omgpe**2) / self.C
        k = jnp.sqrt(ks**2 + kL**2 - 2 * ks * kL * jnp.cos(sarad))

        kdotv = k * Va
        omgdop = omg - kdotv

        # plasma parameters
        # electrons
        vTe = jnp.sqrt(Te[..., jnp.newaxis, jnp.newaxis] / self.Me)  # electron thermal velocity
        klde = (vTe / omgpe) * k

        # ions
        Z = jnp.reshape(jnp.array(Z), [1, 1, 1, -1])
        Mi = jnp.reshape(Mi, [1, 1, 1, -1])
        fract = jnp.reshape(jnp.array(fract), [1, 1, 1, -1])
        Zbar = jnp.sum(Z * fract)
        ni = fract * ne[..., jnp.newaxis, jnp.newaxis, jnp.newaxis] / Zbar
        omgpi = constants * Z * jnp.sqrt(ni * self.Me / Mi)

        vTi = jnp.sqrt(jnp.array(Ti) / Mi)  # ion thermal velocity
        kldi = (vTi / omgpi) * (k[..., jnp.newaxis])

        # ion susceptibilities
        # finding derivative of plasma dispersion function along xii array
        # proper handeling of multiple ion temperatures is not implemented
        xii = 1.0 / jnp.transpose((jnp.sqrt(2.0) * vTi), [1, 0, 2, 3]) * ((omgdop / k)[..., jnp.newaxis])

        num_ion_pts = jnp.shape(xii)
        chiI = jnp.zeros(num_ion_pts)
        ZpiR = jnp.interp(xii, self.xi2, self.Zpi[0, :], left=xii**-2, right=xii**-2)
        ZpiI = jnp.interp(xii, self.xi2, self.Zpi[1, :], left=0, right=0)
        chiI = jnp.sum(-0.5 / (kldi**2) * (ZpiR + 1j * ZpiI), 3)

        # electron susceptibility
        # calculating normilized phase velcoity(xi's) for electrons
        xie = omgdop / (k * vTe) - ud / vTe

        fe_vphi = jnp.exp(jnp.interp(xie, vx, jnp.log(fe)))

        df = jnp.diff(fe_vphi, 1, 1) / jnp.diff(xie, 1, 1)
        df = jnp.append(df, jnp.zeros((len(ne), 1, len(self.scattering_angles["sa"]))), 1)

        chiEI = jnp.pi / (klde**2) * 1j * df

        ratmod = jnp.exp(jnp.interp(self.xi1, vx, jnp.log(fe)))
        ratdf = jnp.gradient(ratmod, self.xi1[1] - self.xi1[0])

        chiERratprim = vmap(ratintn.ratintn, in_axes=(None, 0, None))(
            ratdf, self.xi1[None, :] - self.xi2[:, None], self.xi1
        )

        # if len(fe) == 2:
        chiERrat = jnp.reshape(jnp.interp(xie.flatten(), self.xi2, chiERratprim[:, 0]), xie.shape)
        # else:
        #     chiERrat = jnp.interpn(jnp.arange(0, 2 * jnp.pi, 10**-1.2018), xi2, chiERratprim, beta, xie, "spline")
        chiERrat = -1.0 / (klde**2) * chiERrat

        chiE = chiERrat + chiEI
        epsilon = 1.0 + chiE + chiI

        # This line needs to be changed if ion distribution is changed!!!
        ion_comp_fact = jnp.transpose(fract * Z**2 / Zbar / vTi, [1, 0, 2, 3])
        ion_comp = ion_comp_fact * (
            (jnp.abs(chiE[..., jnp.newaxis])) ** 2.0 * jnp.exp(-(xii**2)) / jnp.sqrt(2 * jnp.pi)
        )

        ele_comp = (jnp.abs(1.0 + chiI)) ** 2.0 * fe_vphi / vTe
        # ele_compE = fe_vphi / vTe # commented because unused

        SKW_ion_omg = 1.0 / k[..., jnp.newaxis] * ion_comp / ((jnp.abs(epsilon[..., jnp.newaxis])) ** 2)

        SKW_ion_omg = jnp.sum(SKW_ion_omg, 3)
        SKW_ele_omg = 1.0 / k * (ele_comp) / ((jnp.abs(epsilon)) ** 2)
        # SKW_ele_omgE = 2 * jnp.pi * 1.0 / klde * (ele_compE) / ((jnp.abs(1 + (chiE))) ** 2) * vTe / omgpe # commented because unused

        PsOmg = (SKW_ion_omg + SKW_ele_omg) * (1 + 2 * omgdop / omgL) * re**2.0 * ne[:, None, None]
        # PsOmgE = (SKW_ele_omg) * (1 + 2 * omgdop / omgL) * re**2.0 * jnp.transpose(ne) # commented because unused
        lams = 2 * jnp.pi * self.C / self.omgs
        PsLam = PsOmg * 2 * jnp.pi * self.C / lams**2
        # PsLamE = PsOmgE * 2 * jnp.pi * C / lams**2 # commented because unused
        formfactor = PsLam

        return formfactor, lams

    def rotate(self, df, angle, reshape: bool = False) -> jnp.ndarray:
        """
        Rotate a 2D array by a given angle in radians

        Args:
            df: 2D array
            angle: angle in radians

        Return:
            interpolated 2D array
        """

        rad_angle = jnp.deg2rad(-angle)
        cos_angle = jnp.cos(rad_angle)
        sin_angle = jnp.sin(rad_angle)
        rotation_matrix = jnp.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

        rotated_coords = jnp.einsum("ij, ki->kj", rotation_matrix, self.coords)
        xq = rotated_coords[:, 0]
        yq = rotated_coords[:, 1]

        return interp2d(xq, yq, self.v, self.v, df, extrap=True, method="linear").reshape(
            (self.v.size, self.v.size), order="F"
        )

    def scan_calc_chi_vals(self, carry, xs):
        """
        Calculate the values of the susceptibility at a given point in the distribution function

        Args:
            carry: container for
                x: 1D array
                DF: 2D array
            xs: container for
                element: angle in radians
                xie_mag_at: float
                klde_mag_at: float

        Returns:
            fe_vphi: float, value of the projected distribution function at the point xie
            chiEI: float, value of the imaginary part of the electron susceptibility at the point xie
            chiERrat: float, value of the real part of the electron susceptibility at the point xie

        """
        x, DF = carry
        fe_vphi, chiEI, chiERrat = self.calc_chi_vals(x, DF, xs)
        return (x, DF), (fe_vphi, chiEI, chiERrat)

    def calc_chi_vals(self, x, DF, inputs):
        """
        Calculate the values of the susceptibility at a given point in the distribution function

        Args:
            carry: container for
                x: 1D array
                DF: 2D array
                inputs: container for
                    element: angle in radians
                    xie_mag_at: float
                    klde_mag_at: float

        Returns:
            fe_vphi: float, value of the projected distribution function at the point xie
            chiEI: float, value of the imaginary part of the electron susceptibility at the point xie
            chiERrat: float, value of the real part of the electron susceptibility at the point xie

        """
        element, xie_mag_at, klde_mag_at = inputs
        fe_2D_k = checkpoint(self.rotate)(DF, element * 180 / jnp.pi, reshape=False)
        fe_1D_k = jnp.sum(fe_2D_k, axis=0) * (x[1] - x[0])
        df = jnp.gradient(fe_1D_k, x[1] - x[0])

        # find the location of xie in axis array
        # add the value of fe to the fe container
        fe_vphi = jnp.interp(xie_mag_at, x, fe_1D_k)
        dfe = jnp.interp(xie_mag_at, x, df)

        # Chi is really chi evaluated at the points xie
        # so the imaginary part is
        chiEI = jnp.pi / (klde_mag_at**2) * dfe

        # the real part is solved with rational integration
        # giving the value at a single point where the pole is located at xie_mag[ind]
        chiERrat = (
            -1.0 / (klde_mag_at**2) * ratintn.ratintn(df, x - xie_mag_at, x)
        )  # this may need to be downsampled for run time
        return fe_vphi, chiEI, chiERrat

    def calc_all_chi_vals(self, x, DF, beta, xie_mag, klde_mag):
        """
        Calculate the susceptibility values for all the desired points xie

        Args:
            x: normalized velocity grid
            beta: angle of the k-vector form the x-axis
            DF: 2D array, distribution function
            xie_mag: magnitude of the normalized velocity points where the calculations need to be performed
            klde_mag: magnitude of the wavevector time debye length where the calculations need to be performed

        Returns:
            fe_vphi: projected distribution function
            chiEI: imaginary part of the electron susceptibility
            chiERrat: real part of the electron susceptibility

        """
        calc_chi_vals = "scan"

        flattened_inputs = (beta.flatten(), xie_mag.flatten(), klde_mag.flatten())

        if calc_chi_vals == "scan":
            _, (fe_vphi, chiEI, chiERrat) = scan(
                self.scan_calc_chi_vals, (x, jnp.squeeze(DF)), flattened_inputs, unroll=1
            )

        elif calc_chi_vals == "vmap":
            fe_vphi, chiEI, chiERrat = self.vmap_calc_chi_vals(x, jnp.squeeze(DF), flattened_inputs)

        elif calc_chi_vals == "batch_vmap":
            batch_vmap_calc_chi_vals = partial(self.calc_chi_vals, x, jnp.squeeze(DF))
            fe_vphi, chiEI, chiERrat = jmap(batch_vmap_calc_chi_vals, xs=flattened_inputs, batch_size=1024)
        else:
            raise NotImplementedError

        fe_vphi = fe_vphi.reshape(beta.shape)
        chiEI = chiEI.reshape(beta.shape)
        chiERrat = chiERrat.reshape(beta.shape)

        return fe_vphi, chiEI, chiERrat

    def parallel_calc_all_chi_vals(self, x, DF, flattened_inputs):

        beta, xie_mag, klde_mag = flattened_inputs

        flat_beta = device_put(beta, self.sharding)
        flat_xie_mag = device_put(xie_mag, self.sharding)
        flat_klde_mag = device_put(klde_mag, self.sharding)

        return self.calc_all_chi_vals(x, DF, (flat_beta, flat_xie_mag, flat_klde_mag))

    def calc_in_2D(self, params, ud_ang, va_ang, cur_ne, cur_Te, A, Z, Ti, fract, sa, f_and_v, lam):
        """
        Calculates the collisionless Thomson spectral density function S(k,omg) for a 2D numerical EDF, capable of
        handling multiple plasma conditions and scattering angles. Distribution functions can be arbitrary as
        calculations of the susceptibility are done on-the-fly. Calculations are done in 4 dimension with the following
        shape, [number of gradient-points, number of wavelength points, number of angles, number of ion-species].

        In angular, `fe` is a Tuple, Distribution function (DF), normalized velocity (x), and angles from k_L to f1 in
        radians

        Args:
            params: parameter dictionary, must contain the drift 'ud' and flow 'Va' velocities in the 'general' field
            ud_ang: angle between electron drift and x-axis
            va_ang: angle between ion flow and x-axis
            cur_ne: electron density in 1/cm^3 [1 by number of gradient points]
            cur_Te: electron temperature in keV [1 by number of gradient points]
            A: atomic mass [1 by number of ion species]
            Z: ionization state [1 by number of ion species]
            Ti: ion temperature in keV [1 by number of ion species]
            fract: relative ion composition [1 by number of ion species]
            sa: scattering angle in degrees [1 by number of angles]
            f_and_v: a distribution function object, contains the numerical distribution function and its velocity grid
            lam: probe wavelength

        Returns:
            formfactor: array of the calculated spectrum, has the shape [number of gradient-points, number of
                wavelength points, number of angles]
        """

        Te, ne, Va, ud, fe = (
            cur_Te.squeeze(-1),
            cur_ne.squeeze(-1),
            params["general"]["Va"],
            params["general"]["ud"],
            f_and_v,
        )

        Mi = jnp.array(A) * self.Mp  # ion mass
        re = 2.8179e-13  # classical electron radius cm
        Esq = self.Me * self.C**2 * re  # sq of the electron charge keV cm
        constants = jnp.sqrt(4 * jnp.pi * Esq / self.Me)
        sarad = sa * jnp.pi / 180  # scattering angle in radians
        sarad = jnp.reshape(sarad, [1, 1, -1])

        Va = Va * 1e6  # flow velocity in 1e6 cm/s
        # convert Va from mag, angle to x,y
        Va = (Va * jnp.cos(va_ang * jnp.pi / 180), Va * jnp.sin(va_ang * jnp.pi / 180))
        ud = ud * 1e6  # drift velocity in 1e6 cm/s
        # convert ua from mag, angle to x,y
        ud = (ud * jnp.cos(ud_ang * jnp.pi / 180), ud * jnp.sin(ud_ang * jnp.pi / 180))

        omgL = self.omgL_num / lam  # laser frequency Rad / s
        # calculate k and omega vectors
        omgpe = constants * jnp.sqrt(ne[..., jnp.newaxis, jnp.newaxis])  # plasma frequency Rad/cm
        # omgs = omgs[jnp.newaxis, ..., jnp.newaxis]
        omg = self.omgs - omgL

        kL = (jnp.sqrt(omgL**2 - omgpe**2) / self.C, jnp.zeros_like(omgpe))  # defined to be along the x axis
        ks_mag = jnp.sqrt(self.omgs**2 - omgpe**2) / self.C
        ks = (jnp.cos(sarad) * ks_mag, jnp.sin(sarad) * ks_mag)
        k = vsub(ks, kL)  # 2D
        k_mag = jnp.sqrt(vdot(k, k))  # 1D

        # kdotv = k * Va
        omgdop = omg - vdot(k, Va)  # 1D

        # plasma parameters

        # electrons
        vTe = jnp.sqrt(Te[..., jnp.newaxis, jnp.newaxis] / self.Me)  # electron thermal velocity
        klde_mag = (vTe / omgpe) * (k_mag[..., jnp.newaxis])  # 1D

        # ions
        Z = jnp.reshape(Z, [1, 1, 1, -1])
        Mi = jnp.reshape(Mi, [1, 1, 1, -1])
        fract = jnp.reshape(fract, [1, 1, 1, -1])
        Zbar = jnp.sum(Z * fract)
        ni = fract * ne[..., jnp.newaxis, jnp.newaxis, jnp.newaxis] / Zbar
        omgpi = constants * Z * jnp.sqrt(ni * self.Me / Mi)

        vTi = jnp.sqrt(Ti / Mi)  # ion thermal velocity
        kldi = (vTi / omgpi) * (k_mag[..., jnp.newaxis])
        # kldi = vdot((vTi / omgpi), v_add_dim(k))

        # ion susceptibilities
        # finding derivative of plasma dispersion function along xii array
        # proper handeling of multiple ion temperatures is not implemented
        xii = 1.0 / jnp.transpose((jnp.sqrt(2.0) * vTi), [1, 0, 2, 3]) * ((omgdop / k_mag)[..., jnp.newaxis])

        # probably should be generalized to an arbitrary distribtuion function but for now just assuming maxwellian
        ZpiR = jnp.interp(xii, self.xi2, self.Zpi[0, :], left=xii**-2, right=xii**-2)
        ZpiI = jnp.interp(xii, self.xi2, self.Zpi[1, :], left=0, right=0)
        chiI = jnp.sum(-0.5 / (kldi**2) * (ZpiR + jnp.sqrt(-1 + 0j) * ZpiI), 3)

        # electron susceptibility
        # calculating normilized phase velcoity(xi's) for electrons
        # xie = vsub(vdiv(omgdop, vdot(k, vTe)), vdiv(ud, vTe))
        xie = vdiv(vsub(vdot(omgdop / k_mag**2, k), ud), vTe)
        xie_mag = jnp.sqrt(vdot(xie, xie))
        DF, (x, y) = fe

        # for each vector in xie
        # find the rotation angle beta, the heaviside changes the angles to [0, 2pi)
        beta = jnp.arctan(xie[1] / xie[0]) + jnp.pi * (-jnp.heaviside(xie[0], 1) + 1)

        fe_vphi, chiEI, chiERrat = self.calc_all_chi_vals(x[0, :], DF, beta, xie_mag, klde_mag)

        chiE = chiERrat + 1j * chiEI
        epsilon = 1.0 + chiE + chiI

        # This line needs to be changed if ion distribution is changed!!!
        ion_comp_fact = jnp.transpose(fract * Z**2 / Zbar / vTi, [1, 0, 2, 3])
        ion_comp = ion_comp_fact * (
            (jnp.abs(chiE[..., jnp.newaxis])) ** 2.0 * jnp.exp(-(xii**2)) / jnp.sqrt(2 * jnp.pi)
        )

        ele_comp = (jnp.abs(1.0 + chiI)) ** 2.0 * fe_vphi / vTe

        SKW_ion_omg = 1.0 / k_mag[..., jnp.newaxis] * ion_comp / ((jnp.abs(epsilon[..., jnp.newaxis])) ** 2)

        SKW_ion_omg = jnp.sum(SKW_ion_omg, 3)
        SKW_ele_omg = 1.0 / k_mag * (ele_comp) / ((jnp.abs(epsilon)) ** 2)
        # SKW_ele_omgE = 2 * jnp.pi * 1.0 / klde * (ele_compE) / ((jnp.abs(1 + (chiE))) ** 2) * vTe / omgpe # commented because unused

        PsOmg = (SKW_ion_omg + SKW_ele_omg) * (1 + 2 * omgdop / omgL) * re**2.0 * ne[:, None, None]
        # PsOmgE = (SKW_ele_omg) * (1 + 2 * omgdop / omgL) * re**2.0 * jnp.transpose(ne) # commented because unused
        lams = 2 * jnp.pi * self.C / self.omgs
        PsLam = PsOmg * 2 * jnp.pi * self.C / lams**2
        # PsLamE = PsOmgE * 2 * jnp.pi * C / lams**2 # commented because unused
        formfactor = PsLam
        # formfactorE = PsLamE # commented because unused

        return formfactor, lams
