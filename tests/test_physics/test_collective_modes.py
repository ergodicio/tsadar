"""Fast integrated checks of collective Thomson-scattering physics.

The cases use a directly specified Maxwellian plasma so the physical oracle is not
coupled to YAML merging, MLflow, plotting, detector calibration, or frozen spectra.
They exercise the production :class:`~tsadar.core.physics.form_factor.FormFactor`
from physical parameters through the returned wavelength-space spectrum.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from jax import config, jit, numpy as jnp

config.update("jax_enable_x64", True)

from tsadar.core.physics.form_factor import FormFactor

from .checks import (
    assert_finite_nonnegative_spectrum,
    assert_frequency_shifts_match,
    assert_outward_peak_motion,
    quadratic_peak_wavelength,
)

pytestmark = pytest.mark.physics

C_CM_PER_S = 2.99792458e10
ELECTRON_MASS_KEV_S2_PER_CM2 = 510.9896 / C_CM_PER_S**2
PROTON_MASS_KEV_S2_PER_CM2 = 1836.1 * ELECTRON_MASS_KEV_S2_PER_CM2
CLASSICAL_ELECTRON_RADIUS_CM = 2.8179e-13
LASER_WAVELENGTH_NM = 526.5
SCATTERING_ANGLE_DEG = 60.0
ELECTRON_TEMPERATURE_KEV = 0.6
ION_TEMPERATURE_KEV = 0.2


def _maxwellian_params(
    *,
    ne_1e20,
    ion_flow_1e6_cm_s=0.0,
    scattering_temperature_kev=ELECTRON_TEMPERATURE_KEV,
):
    velocity = jnp.linspace(-8.0, 8.0, 257)
    distribution = jnp.exp(-0.5 * velocity**2) / jnp.sqrt(2.0 * jnp.pi)
    return {
        "electron": {
            "ne": ne_1e20,
            "Te": scattering_temperature_kev,
            "fe": distribution,
            "v": velocity,
        },
        "ion-1": {
            "Ti": ION_TEMPERATURE_KEV,
            "Z": 1.0,
            "A": 1.0,
            "Va": ion_flow_1e6_cm_s,
            "fract": 1.0,
        },
        "general": {
            "lam": LASER_WAVELENGTH_NM,
            "amp1": 1.0,
            "amp2": 1.0,
            "amp3": 1.0,
            "ne_gradient": 0.0,
            "Te_gradient": 0.0,
            "ud": 0.0,
        },
    }


def _angular_frequency(wavelength_nm):
    return 2.0 * np.pi * C_CM_PER_S / (np.asarray(wavelength_nm) * 1.0e-7)


def _plasma_frequency(ne_1e20):
    # omega_pe^2 = 4 pi n_e e^2 / m_e and e^2 = m_e c^2 r_e in Gaussian units.
    coefficient = np.sqrt(4.0 * np.pi * C_CM_PER_S**2 * CLASSICAL_ELECTRON_RADIUS_CM)
    return coefficient * np.sqrt(ne_1e20 * 1.0e20)


def _scattering_wavenumbers(model_omegas, *, ne_1e20, angle_deg=SCATTERING_ANGLE_DEG):
    omega_laser = _angular_frequency(LASER_WAVELENGTH_NM)
    omega_pe = _plasma_frequency(ne_1e20)
    k_scattered = np.sqrt(np.asarray(model_omegas) ** 2 - omega_pe**2) / C_CM_PER_S
    k_laser = np.sqrt(omega_laser**2 - omega_pe**2) / C_CM_PER_S
    angle_rad = np.deg2rad(angle_deg)
    return np.sqrt(
        k_scattered**2 + k_laser**2 - 2.0 * k_scattered * k_laser * np.cos(angle_rad)
    )


def _peak_omegas(wavelengths_nm, spectrum, *, case_id, blue_window, red_window):
    peak_wavelengths = np.array(
        [
            quadratic_peak_wavelength(
                wavelengths_nm,
                spectrum,
                window_nm=blue_window,
                case_id=f"{case_id}/blue",
            ),
            quadratic_peak_wavelength(
                wavelengths_nm,
                spectrum,
                window_nm=red_window,
                case_id=f"{case_id}/red",
            ),
        ]
    )
    return _angular_frequency(peak_wavelengths)


@pytest.fixture(scope="module")
def epw_case():
    form_factor = FormFactor(
        [440.0, 620.0],
        npts=1024,
        lam_shift=0.0,
        scattering_angles={"sa": np.array([SCATTERING_ANGLE_DEG])},
        num_grad_points=1,
        ud_ang=None,
        va_ang=None,
        calc_gain={"calc": False},
    )
    calculate = jit(lambda density: form_factor(_maxwellian_params(ne_1e20=density))[0])

    @lru_cache(maxsize=None)
    def evaluate(density):
        return np.squeeze(np.asarray(calculate(jnp.asarray(density))))

    return np.asarray(form_factor.lambda_axis_nm), evaluate


@pytest.fixture(scope="module")
def iaw_case():
    form_factor = FormFactor(
        [525.0, 528.0],
        npts=1024,
        lam_shift=0.0,
        scattering_angles={"sa": np.array([SCATTERING_ANGLE_DEG])},
        num_grad_points=1,
        ud_ang=None,
        va_ang=None,
        calc_gain={"calc": False},
    )
    calculate = jit(
        lambda flow: form_factor(
            _maxwellian_params(ne_1e20=0.2, ion_flow_1e6_cm_s=flow)
        )[0]
    )

    @lru_cache(maxsize=None)
    def evaluate(flow):
        return np.squeeze(np.asarray(calculate(jnp.asarray(flow))))

    return np.asarray(form_factor.lambda_axis_nm), evaluate


def _epw_shifts(epw_case, density, *, case_id):
    wavelengths, evaluate = epw_case
    spectrum = evaluate(density)
    assert_finite_nonnegative_spectrum(spectrum, case_id=case_id)
    model_omegas = _peak_omegas(
        wavelengths,
        spectrum,
        case_id=case_id,
        blue_window=(450.0, 515.0),
        red_window=(540.0, 610.0),
    )
    return np.abs(model_omegas - _angular_frequency(LASER_WAVELENGTH_NM))


def _iaw_peak_omegas(iaw_case, flow, *, case_id):
    wavelengths, evaluate = iaw_case
    spectrum = evaluate(flow)
    assert_finite_nonnegative_spectrum(spectrum, case_id=case_id)
    return _peak_omegas(
        wavelengths,
        spectrum,
        case_id=case_id,
        blue_window=(525.3, 526.4),
        red_window=(526.6, 527.7),
    )


def test_epw_peaks_follow_bohm_gross_dispersion(epw_case):
    r"""P-EPW-01: ``Delta omega^2 = omega_pe^2 + 3 k^2 T_e/m_e``.

    The 5% detuning tolerance covers the known higher-order kinetic correction at
    ``k lambda_De = 0.46--0.51``; wavelength-grid refinement changes the result by
    less than 0.1%.  Comparing detuning is essential: a percent tolerance on the
    optical carrier can hide an order-unity plasma-wave error.
    """

    density = 0.2
    measured_shifts = _epw_shifts(epw_case, density, case_id="P-EPW-01")
    model_omegas = np.array(
        [
            _angular_frequency(LASER_WAVELENGTH_NM) + measured_shifts[0],
            _angular_frequency(LASER_WAVELENGTH_NM) - measured_shifts[1],
        ]
    )
    wave_numbers = _scattering_wavenumbers(model_omegas, ne_1e20=density)
    expected_shifts = np.sqrt(
        _plasma_frequency(density) ** 2
        + 3.0
        * wave_numbers**2
        * ELECTRON_TEMPERATURE_KEV
        / ELECTRON_MASS_KEV_S2_PER_CM2
    )

    assert_frequency_shifts_match(
        measured_shifts,
        expected_shifts,
        rtol=0.05,
        case_id="P-EPW-01",
        relation="the Bohm-Gross dispersion relation",
    )


def test_increasing_density_moves_both_epw_peaks_outward(epw_case):
    r"""P-EPW-02: ``omega_pe proportional to sqrt(n_e)`` moves both peaks outward."""

    lower = _epw_shifts(epw_case, 0.16, case_id="P-EPW-02/low-density")
    higher = _epw_shifts(epw_case, 0.30, case_id="P-EPW-02/high-density")

    # The reference plasma moves both detunings by more than 20%; 10% leaves a wide
    # numerical margin while still detecting a lost or sign-reversed density response.
    assert_outward_peak_motion(
        lower,
        higher,
        minimum_relative_increase=0.10,
        case_id="P-EPW-02",
    )


def test_density_oracle_rejects_deliberately_reversed_response(epw_case):
    """P-ORACLE-01: prove the suite rejects a result that runs but has bad physics."""

    lower = _epw_shifts(epw_case, 0.16, case_id="P-ORACLE-01/low-density")
    higher = _epw_shifts(epw_case, 0.30, case_id="P-ORACLE-01/high-density")

    # Swapping the two real model responses emulates a successfully executing model
    # whose density correction has the wrong sign.
    with pytest.raises(
        AssertionError, match="higher electron density must move both EPW peaks outward"
    ):
        assert_outward_peak_motion(
            higher,
            lower,
            minimum_relative_increase=0.10,
            case_id="P-ORACLE-01/deliberately-reversed",
        )


def test_iaw_peaks_follow_screened_warm_ion_dispersion(iaw_case):
    r"""P-IAW-01: validate the warm-fluid ion-acoustic detuning.

    The reference is
    ``Delta omega = k sqrt((Z T_e/(1+k^2 lambda_De^2) + 3 T_i)/m_i)``.
    A 3% tolerance covers the measured 2.4% kinetic departure from the warm-fluid
    approximation; grid refinement changes the peak shifts by less than 0.1%.
    """

    density = 0.2
    model_omegas = _iaw_peak_omegas(iaw_case, 0.0, case_id="P-IAW-01")
    omega_laser = _angular_frequency(LASER_WAVELENGTH_NM)
    measured_shifts = np.abs(model_omegas - omega_laser)
    wave_numbers = _scattering_wavenumbers(model_omegas, ne_1e20=density)
    debye_length = np.sqrt(
        ELECTRON_TEMPERATURE_KEV / ELECTRON_MASS_KEV_S2_PER_CM2
    ) / _plasma_frequency(density)
    sound_speed_squared = (
        ELECTRON_TEMPERATURE_KEV / (1.0 + (wave_numbers * debye_length) ** 2)
        + 3.0 * ION_TEMPERATURE_KEV
    ) / PROTON_MASS_KEV_S2_PER_CM2
    expected_shifts = wave_numbers * np.sqrt(sound_speed_squared)

    assert_frequency_shifts_match(
        measured_shifts,
        expected_shifts,
        rtol=0.03,
        case_id="P-IAW-01",
        relation="the Debye-screened warm-ion acoustic dispersion relation",
    )

    midpoint_error = abs(float(np.mean(model_omegas) - omega_laser))
    half_separation = float(np.ptp(model_omegas) / 2.0)
    assert midpoint_error < 0.002 * half_separation, (
        "P-IAW-01: a zero-flow IAW doublet must be centered on the probe in frequency; "
        f"midpoint error={midpoint_error:.6e} rad/s, "
        f"allowed={0.002 * half_separation:.6e} rad/s"
    )


def test_iaw_oracle_rejects_backscatter_wavevector_at_sixty_degrees(iaw_case):
    """P-ORACLE-03: reject the known ``k=2*kL`` finite-angle inconsistency.

    The retired IAW test used the 180-degree backscatter wavevector at 60 degrees and
    compared total optical frequencies with 1% tolerance.  That carrier-frequency
    comparison accepts the bad prediction even though its IAW detuning is wrong by
    more than a factor of two; the detuning oracle must reject it.
    """

    model_omegas = _iaw_peak_omegas(iaw_case, 0.0, case_id="P-ORACLE-03/model")
    omega_laser = _angular_frequency(LASER_WAVELENGTH_NM)
    measured_shifts = np.abs(model_omegas - omega_laser)
    omega_pe = _plasma_frequency(0.2)
    k_laser = np.sqrt(omega_laser**2 - omega_pe**2) / C_CM_PER_S
    bad_shift = (
        2.0
        * k_laser
        * np.sqrt((0.5 + 3.0 * ION_TEMPERATURE_KEV) / PROTON_MASS_KEV_S2_PER_CM2)
    )
    bad_shifts = np.full(2, bad_shift)
    bad_model_omegas = np.array([omega_laser + bad_shift, omega_laser - bad_shift])

    # Reproduce the old, vacuous comparison: the incorrect shifts are small relative
    # to the carrier, so both total frequencies appear to agree within 1%.
    np.testing.assert_allclose(model_omegas, bad_model_omegas, rtol=0.01, atol=0.0)

    with pytest.raises(AssertionError, match="backscatter wavevector at 60 degrees"):
        assert_frequency_shifts_match(
            measured_shifts,
            bad_shifts,
            rtol=0.03,
            case_id="P-ORACLE-03/deliberately-inconsistent",
            relation="the incorrect backscatter wavevector at 60 degrees",
        )


def test_ion_bulk_flow_doppler_shifts_iaw_doublet(iaw_case):
    r"""P-IAW-02: ion flow translates the doublet by ``Delta omega = k V_a``."""

    stationary = _iaw_peak_omegas(iaw_case, 0.0, case_id="P-IAW-02/stationary")
    flow_1e6_cm_s = 2.0
    flowing = _iaw_peak_omegas(iaw_case, flow_1e6_cm_s, case_id="P-IAW-02/flowing")
    measured_translation = float(np.mean(flowing) - np.mean(stationary))
    expected_translation = float(
        np.mean(_scattering_wavenumbers(stationary, ne_1e20=0.2))
        * flow_1e6_cm_s
        * 1.0e6
    )

    np.testing.assert_allclose(
        measured_translation,
        expected_translation,
        rtol=0.02,
        atol=0.0,
        err_msg=(
            "P-IAW-02: the IAW midpoint does not obey the k*Va Doppler shift; "
            "the sign and magnitude of the bulk-flow correction are both physical invariants"
        ),
    )


def test_noiseless_epw_spectrum_recovers_electron_density(epw_case):
    """P-INV-01: a scalar inverse pass recovers the density that generated the EPW."""

    wavelengths, evaluate = epw_case
    true_density = 0.24
    target = evaluate(true_density)
    epw_only = (wavelengths < 515.0) | (wavelengths > 540.0)
    scale = float(np.max(target[epw_only]))

    def loss(density):
        candidate = evaluate(float(density))
        return float(np.mean(((candidate[epw_only] - target[epw_only]) / scale) ** 2))

    result = minimize_scalar(
        loss,
        bounds=(0.16, 0.33),
        method="bounded",
        options={"xatol": 2.0e-4, "maxiter": 20},
    )
    assert (
        result.success
    ), f"P-INV-01: bounded density recovery failed: {result.message}"
    np.testing.assert_allclose(
        result.x,
        true_density,
        rtol=0.0,
        atol=2.0e-3,
        err_msg=(
            "P-INV-01: noiseless EPW recovery did not return the generating density; "
            "the 0.002 absolute tolerance is 10x the optimizer stopping scale"
        ),
    )
