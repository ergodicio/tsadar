"""Higher-resolution physics reference cases for the scheduled slow lane."""

from __future__ import annotations

import numpy as np
import pytest

from jax import config, jit, numpy as jnp

config.update("jax_enable_x64", True)

from tsadar.core.physics.form_factor import FormFactor

from .checks import (
    assert_finite_nonnegative_spectrum,
    assert_frequency_shifts_match,
    quadratic_peak_wavelength,
)
from .test_collective_modes import (
    ELECTRON_MASS_KEV_S2_PER_CM2,
    ELECTRON_TEMPERATURE_KEV,
    LASER_WAVELENGTH_NM,
    _angular_frequency,
    _maxwellian_params,
    _plasma_frequency,
    _scattering_wavenumbers,
)

pytestmark = [pytest.mark.physics, pytest.mark.slow]


def test_bohm_gross_density_angle_reference_sweep():
    """P-EPW-REF-01: Bohm-Gross remains valid over a small density/angle matrix.

    This nine-case, 4096-point sweep is deliberately kept out of pull-request CI.
    Its 6% detuning tolerance bounds the worst higher-order kinetic correction in
    the documented matrix; grid refinement contributes less than 0.1%.
    """

    densities = (0.16, 0.24, 0.32)
    angles = np.array([40.0, 60.0, 80.0])
    form_factor = FormFactor(
        [420.0, 680.0],
        npts=4096,
        lam_shift=0.0,
        scattering_angles={"sa": angles},
        num_grad_points=1,
        ud_ang=None,
        va_ang=None,
        calc_gain={"calc": False},
    )
    calculate = jit(lambda density: form_factor(_maxwellian_params(ne_1e20=density))[0])
    wavelengths = np.asarray(form_factor.lambda_axis_nm)
    omega_laser = _angular_frequency(LASER_WAVELENGTH_NM)

    for density in densities:
        spectra = np.squeeze(np.asarray(calculate(jnp.asarray(density))))
        for angle_index, angle in enumerate(angles):
            case_id = f"P-EPW-REF-01/ne={density:.2f}/angle={angle:.0f}"
            spectrum = spectra[:, angle_index]
            assert_finite_nonnegative_spectrum(spectrum, case_id=case_id)
            peak_wavelengths = np.array(
                [
                    quadratic_peak_wavelength(
                        wavelengths,
                        spectrum,
                        window_nm=(430.0, 515.0),
                        case_id=f"{case_id}/blue",
                    ),
                    quadratic_peak_wavelength(
                        wavelengths,
                        spectrum,
                        window_nm=(540.0, 660.0),
                        case_id=f"{case_id}/red",
                    ),
                ]
            )
            model_omegas = _angular_frequency(peak_wavelengths)
            measured_shifts = np.abs(model_omegas - omega_laser)
            wave_numbers = _scattering_wavenumbers(
                model_omegas,
                ne_1e20=density,
                angle_deg=float(angle),
            )
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
                rtol=0.06,
                case_id=case_id,
                relation="the Bohm-Gross density/angle reference sweep",
            )
