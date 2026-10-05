"""Independent anisotropic-Gaussian non-collective limit of the ARTS2D spectrum."""

import jax
from jax import config, numpy as jnp
import numpy as np
import pytest

config.update("jax_enable_x64", True)

from tsadar.core.physics.form_factor import FormFactor

from .checks import assert_finite_nonnegative_spectrum
from .test_maxwellian_reference import (
    C_CM_PER_S,
    CLASSICAL_ELECTRON_RADIUS_CM,
    ELECTRON_MASS_KEV_S2_PER_CM2,
)

pytestmark = pytest.mark.physics


def _unscreened_gaussian(wavelengths, angles, density, temperature, mean, covariance):
    r"""Analytic Radon Gaussian times r_e^2 n_e/(k vTe) and dω/dλ.

    A normal distribution N(mu, Sigma) projects along khat to
    N(khat.mu, khat.Sigma.khat). No production projection, susceptibility,
    wavelength conversion, or geometry helper is used for this oracle.
    """
    c = C_CM_PER_S
    ne = density * 1.0e20
    re = CLASSICAL_ELECTRON_RADIUS_CM
    omega_pe_squared = 4 * np.pi * re * c**2 * ne
    omega_l = 2 * np.pi * c / (526.5e-7)
    omega_s = 2 * np.pi * c / (wavelengths[:, None] * 1e-7)
    ks = np.sqrt(omega_s**2 - omega_pe_squared) / c
    kl = np.sqrt(omega_l**2 - omega_pe_squared) / c
    kx = ks * np.cos(np.deg2rad(angles)) - kl
    ky = ks * np.sin(np.deg2rad(angles))
    k = np.hypot(kx, ky)
    direction = np.stack((kx / k, ky / k), axis=-1)
    projected_mean = direction @ mean
    projected_variance = np.einsum(
        "...i,ij,...j->...", direction, covariance, direction
    )
    vte = np.sqrt(temperature / ELECTRON_MASS_KEV_S2_PER_CM2)
    xi = (omega_s - omega_l) / (k * vte)
    projection = np.exp(-0.5 * (xi - projected_mean) ** 2 / projected_variance)
    projection /= np.sqrt(2 * np.pi * projected_variance)
    prefactor = (
        re**2
        * ne
        / (k * vte)
        * (1 + 2 * (omega_s - omega_l) / omega_l)
        * 2
        * np.pi
        * c
        / (wavelengths[:, None] * 1e-7) ** 2
    )
    return prefactor * projection


def test_anisotropic_spectrum_approaches_unscreened_radon_gaussian(record_property):
    r"""P-NONCOLL-01: as α_TS→0, ε→1 and Sλ→analytic projected f_e.

    Decreasing density by 10^4 at fixed temperature/geometry removes screening
    and the ion term. The 0.5% limiting spectral tolerance allows finite Cartesian
    rotation/interpolation error, not an arbitrary amplitude fit. Including the
    probe wavelength exercises the ion term where its distribution is largest.
    """
    wavelengths = np.sort(np.append(np.linspace(475, 585, 97), 526.5))
    angles = np.array([35.0, 80.0, 125.0])
    mean = np.array([0.4, -0.3])
    covariance = np.array([[0.8, 0.25], [0.25, 1.5]])
    velocity = jnp.linspace(-9, 9, 257)
    vx, vy = np.meshgrid(velocity, velocity)
    centered = np.stack((vx, vy), axis=-1) - mean
    distribution = np.exp(
        -0.5
        * np.einsum("...i,ij,...j->...", centered, np.linalg.inv(covariance), centered)
    ) / (2 * np.pi * np.sqrt(np.linalg.det(covariance)))
    ff = FormFactor(
        [475, 585],
        npts=98,
        lam_shift=0.0,
        scattering_angles={"sa": angles, "weights": np.ones((1, 3)) / 3},
        num_grad_points=1,
        va_ang=0.0,
        ud_ang=0.0,
        calc_gain={"calc": False},
        n_beta=256,
    )
    params = {
        "electron": {
            "ne": 0.02,
            "Te": 0.8,
            "v": velocity,
            "fe": jnp.asarray(distribution),
        },
        "general": {"lam": 526.5, "ud": 0.0, "Te_gradient": 0.0, "ne_gradient": 0.0},
        "ion-1": {"A": 1.0, "Z": 1.0, "fract": 1.0, "Ti": 0.1, "Va": 0.0},
    }
    sinogram = ff.prepare_2D_sinogram(params)

    @jax.jit
    def evaluate(density):
        changed = {**params, "electron": {**params["electron"], "ne": density}}
        numerator, epsilon = ff.calc_2D_spectral_terms(
            changed, jnp.asarray(wavelengths), sinogram=sinogram
        )
        return numerator[:, 0, :] / jnp.abs(epsilon[:, 0, :]) ** 2, epsilon[:, 0, :]

    errors, screening = [], []
    for density in (0.02, 0.0002, 0.000002):
        actual, epsilon = map(np.asarray, evaluate(density))
        expected = _unscreened_gaussian(
            wavelengths, angles, density, 0.8, mean, covariance
        )
        assert_finite_nonnegative_spectrum(actual, case_id=f"P-NONCOLL-01/ne={density}")
        # Absolute physical amplitude matters: no peak or area rescaling.
        errors.append(np.max(np.abs(actual - expected)) / np.max(expected))
        screening.append(np.max(np.abs(epsilon - 1)))
    record_property("spectral_errors", str(errors))
    record_property("screening", str(screening))
    assert errors[-1] < 0.005, f"P-NONCOLL-01: limiting spectral errors={errors}"
    assert (
        errors[1] < errors[0] / 10
    ), f"P-NONCOLL-01: no screening convergence: {errors}"
    assert screening[-1] < 0.002, f"P-NONCOLL-01: ε did not approach 1: {screening}"
    assert (
        screening[-1] < screening[0] / 9000
    ), f"P-NONCOLL-01: χ must scale with density: {screening}"
