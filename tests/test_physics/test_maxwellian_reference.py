"""Independent Maxwellian reference and multispecies invariance checks.

These are fast, integrated CPU cases.  They drive the production 1-D
``FormFactor`` from physical plasma parameters through its wavelength-space
spectrum, while keeping the oracle independent of TSADAR's tabulated plasma
dispersion function and rational principal-value quadrature.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from scipy.special import wofz

from jax import config, numpy as jnp

config.update("jax_enable_x64", True)

from tsadar.core.physics.form_factor import FormFactor

from .checks import assert_finite_nonnegative_spectrum

pytestmark = pytest.mark.physics

C_CM_PER_S = 2.99792458e10
ELECTRON_MASS_KEV_S2_PER_CM2 = 510.9896 / C_CM_PER_S**2
PROTON_MASS_KEV_S2_PER_CM2 = 1836.1 * ELECTRON_MASS_KEV_S2_PER_CM2
CLASSICAL_ELECTRON_RADIUS_CM = 2.8179e-13
LASER_WAVELENGTH_NM = 526.5
SCATTERING_ANGLE_DEG = 60.0


def _ion(*, temperature_kev: float, fraction: float) -> dict[str, float]:
    return {
        "A": 1.0,
        "Z": 1.0,
        "Ti": temperature_kev,
        "fract": fraction,
        "Va": 0.0,
    }


def _maxwellian_params() -> dict:
    velocity = jnp.linspace(-9.0, 9.0, 257)
    distribution = jnp.exp(-0.5 * velocity**2) / jnp.sqrt(2.0 * jnp.pi)
    return {
        "electron": {
            "ne": 0.2,
            "Te": 0.5,
            "fe": distribution,
            "v": velocity,
        },
        "general": {
            "ne_gradient": 0.0,
            "Te_gradient": 0.0,
            "lam": LASER_WAVELENGTH_NM,
            "ud": 0.0,
        },
        "ion-1": _ion(temperature_kev=0.08, fraction=1.0),
    }


@pytest.fixture(scope="module")
def maxwellian_case():
    form_factor = FormFactor(
        [440.0, 630.0],
        npts=2048,
        lam_shift=0.0,
        scattering_angles={
            "sa": np.array([SCATTERING_ANGLE_DEG]),
            "weights": np.ones((1, 1)),
        },
        num_grad_points=1,
        ud_ang=None,
        va_ang=None,
        calc_gain={"calc": False},
    )
    params = _maxwellian_params()
    spectrum, wavelengths_cm = form_factor(params)
    return {
        "form_factor": form_factor,
        "params": params,
        "spectrum": np.squeeze(np.asarray(spectrum)),
        "wavelengths_nm": np.squeeze(np.asarray(wavelengths_cm)) * 1.0e7,
    }


def _plasma_dispersion(zeta):
    r"""Return ``Z(zeta) = i sqrt(pi) w(zeta)`` from SciPy's Faddeeva function."""

    return 1j * np.sqrt(np.pi) * wofz(zeta)


def _analytic_maxwellian_spectrum(wavelengths_nm, params):
    r"""Evaluate the collisionless Maxwellian Thomson spectrum independently.

    For each species the analytic susceptibility is

    ``chi_s = (1 + zeta_s Z(zeta_s)) / (k lambda_Ds)^2``,

    where ``Z(zeta) = i sqrt(pi) w(zeta)`` and ``w`` is the Faddeeva
    function.  The spectral density is assembled from

    ``(|1 + chi_i|^2 f_e/vTe + |chi_e|^2 Z^2 f_i/(Zbar vTi))``
    ``/ (k |1 + chi_e + chi_i|^2)``,

    followed by the production model's first-order scattered-frequency factor
    and the exact ``d omega / d lambda`` Jacobian.  Computing ``Z`` with
    :func:`scipy.special.wofz` makes this oracle independent of TSADAR's
    tabulated ``zprimeMaxw`` and 1-D rational-integral susceptibility paths.
    """

    wavelengths_nm = np.asarray(wavelengths_nm)
    electron = params["electron"]
    ion = params["ion-1"]
    general = params["general"]

    electron_density_cm3 = float(electron["ne"]) * 1.0e20
    electron_temperature_kev = float(electron["Te"])
    ion_temperature_kev = float(ion["Ti"])
    ion_mass = float(ion["A"]) * PROTON_MASS_KEV_S2_PER_CM2
    ion_charge = float(ion["Z"])
    ion_fraction = float(ion["fract"])
    z_bar = ion_charge * ion_fraction

    charge_squared_kev_cm = (
        ELECTRON_MASS_KEV_S2_PER_CM2 * C_CM_PER_S**2 * CLASSICAL_ELECTRON_RADIUS_CM
    )
    plasma_frequency_coefficient = np.sqrt(
        4.0 * np.pi * charge_squared_kev_cm / ELECTRON_MASS_KEV_S2_PER_CM2
    )
    electron_plasma_frequency = plasma_frequency_coefficient * np.sqrt(
        electron_density_cm3
    )

    scattered_frequency = 2.0 * np.pi * 1.0e7 * C_CM_PER_S / wavelengths_nm
    laser_frequency = 2.0 * np.pi * 1.0e7 * C_CM_PER_S / float(general["lam"])
    frequency_shift = scattered_frequency - laser_frequency
    scattered_wavenumber = (
        np.sqrt(scattered_frequency**2 - electron_plasma_frequency**2) / C_CM_PER_S
    )
    laser_wavenumber = (
        np.sqrt(laser_frequency**2 - electron_plasma_frequency**2) / C_CM_PER_S
    )
    angle = np.deg2rad(SCATTERING_ANGLE_DEG)
    scattering_wavenumber = np.sqrt(
        scattered_wavenumber**2
        + laser_wavenumber**2
        - 2.0 * scattered_wavenumber * laser_wavenumber * np.cos(angle)
    )

    electron_thermal_speed = np.sqrt(
        electron_temperature_kev / ELECTRON_MASS_KEV_S2_PER_CM2
    )
    k_lambda_de = (
        electron_thermal_speed * scattering_wavenumber / electron_plasma_frequency
    )
    electron_phase_speed = frequency_shift / (
        scattering_wavenumber * electron_thermal_speed
    )
    electron_zeta = electron_phase_speed / np.sqrt(2.0)
    electron_susceptibility = (
        1.0 + electron_zeta * _plasma_dispersion(electron_zeta)
    ) / k_lambda_de**2

    ion_density_cm3 = ion_fraction * electron_density_cm3 / z_bar
    ion_plasma_frequency = (
        plasma_frequency_coefficient
        * ion_charge
        * np.sqrt(ion_density_cm3 * ELECTRON_MASS_KEV_S2_PER_CM2 / ion_mass)
    )
    ion_thermal_speed = np.sqrt(ion_temperature_kev / ion_mass)
    k_lambda_di = ion_thermal_speed * scattering_wavenumber / ion_plasma_frequency
    ion_zeta = frequency_shift / (
        np.sqrt(2.0) * scattering_wavenumber * ion_thermal_speed
    )
    ion_susceptibility = (
        1.0 + ion_zeta * _plasma_dispersion(ion_zeta)
    ) / k_lambda_di**2

    dielectric = 1.0 + electron_susceptibility + ion_susceptibility
    electron_distribution = np.exp(-0.5 * electron_phase_speed**2) / np.sqrt(
        2.0 * np.pi
    )
    ion_distribution = np.exp(-(ion_zeta**2)) / np.sqrt(2.0 * np.pi)
    electron_component = (
        np.abs(1.0 + ion_susceptibility) ** 2
        * electron_distribution
        / electron_thermal_speed
    )
    ion_component = (
        ion_fraction
        * ion_charge**2
        / z_bar
        / ion_thermal_speed
        * np.abs(electron_susceptibility) ** 2
        * ion_distribution
    )
    spectrum_per_frequency = (
        (electron_component + ion_component)
        / scattering_wavenumber
        / np.abs(dielectric) ** 2
        * (1.0 + 2.0 * frequency_shift / laser_frequency)
        * CLASSICAL_ELECTRON_RADIUS_CM**2
        * electron_density_cm3
    )
    wavelengths_cm = wavelengths_nm * 1.0e-7
    return spectrum_per_frequency * 2.0 * np.pi * C_CM_PER_S / wavelengths_cm**2


def _relative_spectrum_errors(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    difference = np.abs(actual - expected)
    return (
        float(np.max(difference) / np.max(np.abs(expected))),
        float(np.sum(difference) / np.sum(np.abs(expected))),
    )


def _assert_species_split_invariant(actual, expected, *, case_id: str) -> None:
    relative_linf, relative_l1 = _relative_spectrum_errors(actual, expected)
    assert relative_linf < 1.0e-12, (
        f"{case_id}: splitting one physical ion population into identical numerical "
        "species must not change the spectrum; peak-normalized L-infinity error="
        f"{relative_linf:.3e}, allowed=1.000e-12"
    )
    assert relative_l1 < 1.0e-12, (
        f"{case_id}: splitting one physical ion population into identical numerical "
        f"species must conserve the spectrum; relative L1 error={relative_l1:.3e}, "
        "allowed=1.000e-12"
    )


def test_maxwellian_spectrum_matches_independent_faddeeva_reference(
    maxwellian_case,
):
    r"""P-MAXWELL-01: validate the complete Maxwellian ``S(k, omega)``.

    This checks both electron and ion terms against the analytic Faddeeva
    susceptibility ``chi_s = (1 + zeta_s Z(zeta_s))/(k lambda_Ds)^2``.  The
    5% peak-normalized and 2.5% integrated tolerances bound the intentionally
    different sampled-EDF/rational-quadrature discretization; the measured CPU
    errors are about 2.6% and 1.1%, respectively, at 2048 wavelengths.
    """

    spectrum = maxwellian_case["spectrum"]
    expected = _analytic_maxwellian_spectrum(
        maxwellian_case["wavelengths_nm"], maxwellian_case["params"]
    )
    assert_finite_nonnegative_spectrum(spectrum, case_id="P-MAXWELL-01/model")
    assert_finite_nonnegative_spectrum(expected, case_id="P-MAXWELL-01/reference")

    relative_linf, relative_l1 = _relative_spectrum_errors(spectrum, expected)
    assert relative_linf < 0.05, (
        "P-MAXWELL-01: the 1-D Maxwellian spectrum violates the independent "
        "Faddeeva reference; peak-normalized L-infinity error="
        f"{relative_linf:.3%}, allowed=5.000%"
    )
    assert relative_l1 < 0.025, (
        "P-MAXWELL-01: the 1-D Maxwellian spectrum violates the independent "
        f"Faddeeva reference; relative L1 error={relative_l1:.3%}, "
        "allowed=2.500%"
    )


def test_identical_ion_species_splitting_preserves_spectrum(maxwellian_case):
    r"""P-MULTI-01: ``chi_i`` and the ion numerator are species-additive.

    Replacing one ion population of fraction one by two populations with the
    same ``A``, ``Z``, ``Ti``, and ``Va`` and fractions 0.3/0.7 leaves
    ``Zbar``, total susceptibility, and spectral density invariant.
    """

    split_params = deepcopy(maxwellian_case["params"])
    split_params["ion-1"] = _ion(temperature_kev=0.08, fraction=0.3)
    split_params["ion-2"] = _ion(temperature_kev=0.08, fraction=0.7)
    split_spectrum, split_wavelengths_cm = maxwellian_case["form_factor"](split_params)
    split_spectrum = np.squeeze(np.asarray(split_spectrum))
    split_wavelengths_nm = np.squeeze(np.asarray(split_wavelengths_cm)) * 1.0e7

    assert_finite_nonnegative_spectrum(split_spectrum, case_id="P-MULTI-01/split")
    np.testing.assert_allclose(
        split_wavelengths_nm,
        maxwellian_case["wavelengths_nm"],
        rtol=0.0,
        atol=0.0,
        err_msg="P-MULTI-01: species bookkeeping changed the wavelength grid",
    )
    _assert_species_split_invariant(
        split_spectrum,
        maxwellian_case["spectrum"],
        case_id="P-MULTI-01",
    )


def test_species_split_oracle_rejects_temperature_mismatch(maxwellian_case):
    """P-ORACLE-02: reject a deliberately inconsistent species split.

    The second population is intentionally assigned twice the original ion
    temperature.  It is therefore not the same physical population, and the
    strict species-splitting oracle must detect the otherwise successful model
    evaluation.
    """

    inconsistent_params = deepcopy(maxwellian_case["params"])
    inconsistent_params["ion-1"] = _ion(temperature_kev=0.08, fraction=0.3)
    inconsistent_params["ion-2"] = _ion(temperature_kev=0.16, fraction=0.7)
    inconsistent_spectrum, _ = maxwellian_case["form_factor"](inconsistent_params)
    inconsistent_spectrum = np.squeeze(np.asarray(inconsistent_spectrum))
    assert_finite_nonnegative_spectrum(
        inconsistent_spectrum, case_id="P-ORACLE-02/inconsistent-split"
    )

    with pytest.raises(AssertionError, match="must not change the spectrum"):
        _assert_species_split_invariant(
            inconsistent_spectrum,
            maxwellian_case["spectrum"],
            case_id="P-ORACLE-02/deliberately-inconsistent",
        )
